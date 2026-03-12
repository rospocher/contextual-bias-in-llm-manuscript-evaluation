#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import os
import json
import re
import sys
import time
import random
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import feedparser  # kept if you use it elsewhere; safe to leave
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import sleep

# ----------------------------
# Model (Transformers)
# ----------------------------
model_id = "allenai/Olmo-3-7B-Instruct"
model_path_name = "olmo"  # folder label


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python script.py <LANGUAGE>  (e.g., python script.py EN)")
    return sys.argv[1].strip()


language = parse_args()
base_dir = language

config_path = os.path.join(base_dir, f"config.{language}.json")
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

# Required config
names_by_group: Dict[str, List[str]] = cfg["names"]              # {"BF":[...], "WM":[...]}
inst_by_tier: Dict[str, List[str]] = cfg["institutions"]         # {"TOP":[...], "LOW":[...]}
surname: str = cfg["surname"]
metrics_by_level: Dict[str, Dict[str, int]] = cfg["metrics"]     # {"LOW":{"h_index":..,"citations":..}, "HIGH":{...}}

TEMPERATURE = cfg["temperature"]
NUMBER_OF_ITERATIONS = int(cfg["number_of_iterations"])
begin = cfg["start"]
end = cfg["end"]
data_folder = cfg["data_folder"]
text_data_folder = cfg["text_data_folder"]
LIKERT_MIN = int(cfg["likert_min"])
LIKERT_MAX = int(cfg["likert_max"])

# Input files
percorso_testi = os.path.join(base_dir, text_data_folder)
percorso_questionari = os.path.join(base_dir, f"questionnaire.{language}.txt")
percorso_questionari_noAttr = os.path.join(base_dir, f"questionnaire.{language}.noAttr.txt")
percorso_prompt = os.path.join(base_dir, f"prompt.{language}.txt")

# Output folders
output_dir = os.path.join(base_dir, data_folder, model_path_name)
json_out_dir = os.path.join(output_dir, "JSONoutput")
csv_out_dir = os.path.join(output_dir, "CSVoutput")
os.makedirs(json_out_dir, exist_ok=True)
os.makedirs(csv_out_dir, exist_ok=True)


# ----------------------------
# Deterministic sampling helpers (stable across runs)
# ----------------------------
def stable_rng(*parts: str) -> random.Random:
    s = "||".join(str(p) for p in parts)
    seed = int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)  # 64-bit seed
    return random.Random(seed)

def pick_one(text_id: str, tag: str, options: List[str]) -> str:
    if not options:
        raise ValueError(f"Empty options list for tag={tag}")
    rng = stable_rng(text_id, tag)
    return rng.choice(options)

def build_text_level_tokens(text_id: str) -> Dict[str, str]:
    """
    For each text, pick:
      - one BF first name, one WM first name
      - one TOP institution, one LOW institution
    These picks stay fixed across ALL conditions for that text.
    """
    bf_first = pick_one(text_id, "NAME_BF", names_by_group["BF"])
    wm_first = pick_one(text_id, "NAME_WM", names_by_group["WM"])
    top_inst = pick_one(text_id, "INST_TOP", inst_by_tier["TOP"])
    low_inst = pick_one(text_id, "INST_LOW", inst_by_tier["LOW"])

    return {
        "BF_name": f"{bf_first} {surname}",
        "WM_name": f"{wm_first} {surname}",
        "TOP_inst": top_inst,
        "LOW_inst": low_inst,
    }


# ----------------------------
# IO helpers
# ----------------------------
def leggiTesti(percorso: str) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for p in sorted(Path(percorso).glob("*.txt")):
        d[p.name.replace(".txt", "")] = p.read_text(encoding="utf-8")  # keep \n
    return d

def safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))
    return s.strip("_")[:200]

def write_jsonl(path: str, items: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


# ----------------------------
# Load stimuli + templates
# ----------------------------
dict_testi = leggiTesti(percorso_testi)

with open(percorso_questionari, "r", encoding="utf-8") as f:
    questionario_tpl = f.read()

with open(percorso_questionari_noAttr, "r", encoding="utf-8") as f:
    questionario_noAttr_tpl = f.read()

with open(percorso_prompt, "r", encoding="utf-8") as f:
    prompt_tpl = f.read()

# Replace likert bounds in prompt (questionnaires are filled per-condition below)
prompt_tpl = prompt_tpl.replace("{likert_min}", str(LIKERT_MIN)).replace("{likert_max}", str(LIKERT_MAX))


# ----------------------------
# Conditions (2×2×2 + NONE)
# ----------------------------
CONDITIONS: List[Tuple[str, str, str]] = [
    ("BF", "TOP", "LOW"),
    ("BF", "TOP", "HIGH"),
    ("BF", "LOW", "LOW"),
    ("BF", "LOW", "HIGH"),
    ("WM", "TOP", "LOW"),
    ("WM", "TOP", "HIGH"),
    ("WM", "LOW", "LOW"),
    ("WM", "LOW", "HIGH"),
]
NONE_COND_ID = "NONE"


def render_questionnaire_attributed(
    text_body: str,
    tokens: Dict[str, str],
    group: str,
    tier: str,
    metric: str,
) -> Tuple[str, dict]:
    """
    Returns (filled_questionnaire, metadata_used_for_profile)
    """
    name_val = tokens[f"{group}_name"]
    inst_val = tokens[f"{tier}_inst"]
    h_val = metrics_by_level[metric]["h_index"]
    c_val = metrics_by_level[metric]["citations"]

    q = questionario_tpl
    q = q.replace("{likert_min}", str(LIKERT_MIN)).replace("{likert_max}", str(LIKERT_MAX))
    q = q.replace("{name}", str(name_val))
    q = q.replace("{institution}", str(inst_val))
    q = q.replace("{h_index}", str(h_val))
    q = q.replace("{citations}", str(c_val))
    q = q.replace("[TEXT]", text_body)

    meta = {
        "group": group,
        "tier": tier,
        "metric": metric,
        "name": name_val,
        "institution": inst_val,
        "h_index": h_val,
        "citations": c_val,
    }
    return q, meta


def render_questionnaire_none(text_body: str) -> Tuple[str, dict]:
    q = questionario_noAttr_tpl
    q = q.replace("{likert_min}", str(LIKERT_MIN)).replace("{likert_max}", str(LIKERT_MAX))
    q = q.replace("[TEXT]", text_body)
    return q, {
        "group": "NONE",
        "tier": "NONE",
        "metric": "NONE",
        "name": "NONE",
        "institution": "NONE",
        "h_index": "NONE",
        "citations": "NONE",
    }


# ----------------------------
# Model download + load (Transformers)
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
llm = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
llm.eval()
print("Model loading complete.")


# ----------------------------
# LLM call + parsing/validation
# ----------------------------
def normalize_q_keys(s: str) -> str:
    def repl(m):
        prefix = m.group(1)
        num = int(m.group(2))
        return f"{prefix}{num:02d}="
    return re.sub(r"\b([qQ])0*([0-9]{1,4})=", repl, s)

def clean_output(text: str) -> str:
    t = (text or "").replace("\n", " ")
    t = t.replace("\\", "")
    t = t.replace('"', "''")
    t = t.strip()
    # remove trailing pipes/spaces
    t = t.rstrip(" |")
    return t

def questionario_LLM(
    questionario_autore_storia: str,
    prompt_base: str,
    llm: AutoModelForCausalLM,
    temperature: float,
    test_mode: bool = False,
) -> str:

    user_payload = f"{prompt_base}\n\n{questionario_autore_storia}\n\n"
    system_prompt = "Before emitting the output, enforce that it has the correct format (pipe separated answers) and that there are exactly 14 answers (q01..q14) and no other keys. Emit everything on a single line, without line breaks. ONLY the answers, do NOT repeat the question or add any commentary. Format: q01=.. | q02=.. | ... | q14=..  (with q01..q14 as keys)."
    print(f"[PROMPT]  Prepared.")

    if test_mode:
        sleep(1)
        return " | ".join([f"q{i:02d}=2" for i in range(1, 15)])

    # Use model's official chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    #print("#"*50+"\n"+prompt+"\n"+"#"*50)

    if test_mode:
        sleep(1)
        return " | ".join([f"q{i:02d}=2" for i in range(1, 15)])

        # Transformers generate
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(llm.device) for k, v in inputs.items()}

    do_sample = True if (temperature is not None and float(temperature) > 0) else False

    with torch.no_grad():
        gen_ids = llm.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            top_p=0.95 if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (after the prompt)
    prompt_len = inputs["input_ids"].shape[-1]
    raw = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)

    # Truncate if the model outputs extra role markers / template remnants
    STOP_MARKERS = ["<|im_end|>", "<|im_start|>", "</s>"]
    for m in STOP_MARKERS:
        if m in raw:
            raw = raw.split(m, 1)[0]

    raw = raw.strip()
    print(raw)
    return normalize_q_keys(clean_output(raw))

def parse_llm_answers(risposta_str: str) -> Optional[dict]:
    d: Dict[str, str] = {}
    try:
        for item in risposta_str.split("|"):
            item = item.strip()
            qn, val = item.split("=", maxsplit=1)
            d[qn.replace("\n", "").replace(" ", "").strip()] = val.replace("−", "-").strip() #to cope with \n in keys
        return d
    except Exception:
        return None

EXPECTED_KEYS = [f"q{i:02d}" for i in range(1, 15)]
EXPECTED_LEN = len(EXPECTED_KEYS)
EXPECTED_SET = set(EXPECTED_KEYS)

digits_re = re.compile(r"^[+−-]?\d+$")

def allowed_value_checker(risposta_dict: dict) -> bool:
    if set(risposta_dict.keys()) != EXPECTED_SET:
        print(f"[DEBUG] Keys mismatch. Expected: {EXPECTED_SET}, Got: {set(risposta_dict.keys())}")
        return False
    for v in risposta_dict.values():
        s = str(v).strip()
        if not digits_re.match(s):
            print(f"[DEBUG] Value '{v}' is not a valid integer.")
            return False
        x = int(s)
        if not (LIKERT_MIN <= x <= LIKERT_MAX):
            print(f"[DEBUG] Value {x} is out of allowed range [{LIKERT_MIN}, {LIKERT_MAX}].")
            return False
    return True

def correct_structure_checker(answer_str: str) -> bool:
    # expected exactly N items split by '|'
    return len([p for p in answer_str.split("|") if p.strip()]) == EXPECTED_LEN


# ----------------------------
# Main loop
# ----------------------------
rng = random.Random(42)
storie_items = list(dict_testi.items())
rng.shuffle(storie_items)

errori = 0
index = 0

for storia, testoStoria in storie_items:
    # Wrap manuscript text once
    text_body = f"[{begin}]\n{testoStoria}\n[{end}]"

    # Per-text deterministic tokens
    tokens = build_text_level_tokens(storia)

    # Build per-text questionnaires for 8 conditions + NONE
    attribuzione_text: Dict[str, str] = {}
    condition_meta: Dict[str, dict] = {}

    for group, tier, metric in CONDITIONS:
        cond_id = f"{group}_{tier}_{metric}"
        q_filled, meta = render_questionnaire_attributed(
            text_body=text_body,
            tokens=tokens,
            group=group,
            tier=tier,
            metric=metric,
        )
        attribuzione_text[cond_id] = q_filled
        condition_meta[cond_id] = meta

    q_none, meta_none = render_questionnaire_none(text_body=text_body)
    attribuzione_text[NONE_COND_ID] = q_none
    condition_meta[NONE_COND_ID] = meta_none

    # Shuffle conditions per text, reproducibly
    autori_items = list(attribuzione_text.items())
    rng.shuffle(autori_items)

    for cond_id, questAutore in autori_items:
        # Collect NUMBER_OF_ITERATIONS valid answers
        risposte_strutturate: List[dict] = []
        n_ok = 0
        n_try = 0

        json_path = os.path.join(json_out_dir, f"{safe_filename(storia)}__{safe_filename(cond_id)}.json")
        csv_path = os.path.join(csv_out_dir, f"{safe_filename(storia)}__{safe_filename(cond_id)}.csv")

        # If file exists, load and continue appending (optional). Here: overwrite each run.
        # risposte_strutturate = []

        if index >= 219:
            while n_ok < NUMBER_OF_ITERATIONS:
                n_try += 1
                print(f"\n{'='*80}\n")
                print(f"[RUN]     {model_path_name} - {language} | text={storia} | condition={cond_id} | ok={n_ok}/{NUMBER_OF_ITERATIONS} | try={n_try} | index={index}")
                meta = condition_meta.get(cond_id, {})
                if cond_id == "NONE":
                    print("[TOKENS]  NONE (no profile)")
                else:
                    print(
                        "[TOKENS]  "
                        f"name={meta.get('name')} | "
                        f"institution={meta.get('institution')} | "
                        f"h_index={meta.get('h_index')} | "
                        f"citations={meta.get('citations')} | "
                        f"group={meta.get('group')} | tier={meta.get('tier')} | metric={meta.get('metric')}"
                    )
                print(f"[TEXT]    {text_body[10:100].replace('\n', ' ')}\n...")
                risposta_str = questionario_LLM(questAutore, prompt_tpl, llm, TEMPERATURE)
                print(f"[LLM]     {risposta_str}")

                if not correct_structure_checker(risposta_str):
                    errori += 1
                    print("[ERROR01] Wrong pipe/field count.")
                    continue

                risposta_dict = parse_llm_answers(risposta_str)
                if risposta_dict is None:
                    errori += 1
                    print("[ERROR02] Parsing failed.")
                    continue

                if not allowed_value_checker(risposta_dict):
                    errori += 1
                    print("[ERRORR03] Values out of range or missing/extra keys.")
                    continue

                print("[VALID]   Answer accepted.")
                # Attach metadata for analysis/auditing
                rec = {
                    **risposta_dict,
                    "text_id": storia,
                    "condition_id": cond_id,
                    "iteration": n_ok,
                    "model": model_path_name,
                    "temperature": TEMPERATURE,
                    **condition_meta.get(cond_id, {}),
                }
                risposte_strutturate.append(rec)

                # Write incremental JSON
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(risposte_strutturate, f, ensure_ascii=False, indent=2)

                n_ok += 1

            # Write CSV at the end for this (text, condition)
            df = pd.read_json(json_path)
            df.to_csv(csv_path, index=False)

        index += 1

print(f"Numero totale di risposte NON VALIDE: {errori}")