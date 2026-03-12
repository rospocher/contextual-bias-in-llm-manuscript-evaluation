#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from scipy import stats


# -----------------------------------------------------------------------------
# Factor coding (NEW)
# -----------------------------------------------------------------------------

INST_HI = "HI"
INST_LI = "LI"

BIB_HI = "HP"  
BIB_LI = "LP"  

NONE_ID = "NONE"


# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------

QUESTION_LABELS = {
    "q01": "q01 (clear writing)",
    "q02": "q02 (novel contribution)",
    "q03": "q03 (scientifically negligible)",
    "q04": "q04 (important problem)",
    "q05": "q05 (sound methodology)",
    "q06": "q06 (sufficient evidence)",
    "q07": "q07 (acknowledges limitations)",
    "q08": "q08 (top-tier venue)",
    "q09": "q09 (unlikely important)",
    "q10": "q10 (academic award)",
    "q11": "q11 (worthy of funding)",
    "q12": "q12 (positive for hiring)",
    "q13": "q13 (field-changing)",
    "q14": "q14 (recommend acceptance)",
}


# -----------------------------------------------------------------------------
# Design detection
# -----------------------------------------------------------------------------

DesignKind = Literal["A_2x2x2", "B_2x2"]


@dataclass(frozen=True)
class DesignSpec:
    kind: DesignKind
    effect_order: List[str]
    effect_labels: Dict[str, str]
    interaction_order: List[str]
    loto_outcomes: List[str]


def infer_design_from_conditions(condition_ids: List[str], none_id: str = NONE_ID) -> DesignSpec:
    conds = [c for c in condition_ids if c != none_id]
    if not conds:
        raise ValueError("No non-NONE conditions found to infer design.")

    parts_lens = sorted({len(c.split("_")) for c in conds})

    # A: BF_HI_HP (3 parts) ; B: BF_HI (2 parts)
    if 3 in parts_lens:
        return DesignSpec(
            kind="A_2x2x2",
            effect_order=["HI_minus_LI", "HP_minus_LP", "BF_minus_WM"],
            effect_labels={
                "BF_minus_WM": "BF − WM (name cue)",
                "HI_minus_LI": "HI − LI (institution tier)",
                "HP_minus_LP": "HP − LP (bibliometrics)",
            },
            interaction_order=["Name_x_Inst", "Name_x_Bibliometrics", "Inst_x_Bibliometrics"],
            loto_outcomes=["QI", "PI", "AR"],
        )
    if parts_lens == [2]:
        return DesignSpec(
            kind="B_2x2",
            effect_order=["HI_minus_LI", "BF_minus_WM"],
            effect_labels={
                "BF_minus_WM": "BF − WM (name cue)",
                "HI_minus_LI": "HI − LI (institution tier)",
            },
            interaction_order=["Name_x_Inst"],
            loto_outcomes=["QI", "PI", "AR"],
        )

    # fallback
    if any(l == 3 for l in parts_lens):
        return DesignSpec(
            kind="A_2x2x2",
            effect_order=["HI_minus_LI", "HP_minus_LP", "BF_minus_WM"],
            effect_labels={
                "BF_minus_WM": "BF − WM (name cue)",
                "HI_minus_LI": "HI − LI (institution tier)",
                "HP_minus_LP": "HP − LP (bibliometrics)",
            },
            interaction_order=["Name_x_Inst", "Name_x_Bibliometrics", "Inst_x_Bibliometrics"],
            loto_outcomes=["QI", "PI", "AR"],
        )
    return DesignSpec(
        kind="B_2x2",
        effect_order=["HI_minus_LI", "BF_minus_WM"],
        effect_labels={
            "BF_minus_WM": "BF − WM (name cue)",
            "HI_minus_LI": "HI − LI (institution tier)",
        },
        interaction_order=["Name_x_Inst"],
        loto_outcomes=["PI", "AR"],
    )


# -----------------------------------------------------------------------------
# IO + basics
# -----------------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"text_id", "condition_id", "iteration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def qcols_expected() -> List[str]:
    return [f"q{i:02d}" for i in range(1, 15)]


def reverse_score(x: pd.Series, likert_min: float, likert_max: float) -> pd.Series:
    return (likert_min + likert_max) - x


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# -----------------------------------------------------------------------------
# Normalization of factor columns / condition_ids (NEW)
# -----------------------------------------------------------------------------

def normalize_factor_columns_and_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust normalization:
      - Rename meta column 'metric' -> 'bibliometrics' (if present)
      - Normalize meta factor levels:
          tier: TOP/LOW -> HI/LI
          bibliometrics: HIGH/LOW -> HP/LP  (also accept HB/LB if present)
      - Normalize condition_id by parsing:
          A design: BF_TOP_LOW -> BF_HI_LP  (tier->HI/LI; bib->HP/LP)
          B design: BF_TOP     -> BF_HI     (tier->HI/LI)
    This avoids the earlier bug where trailing _LOW in A was misread as institution LOW.
    """
    df = df.copy()

    # rename meta column
    if "metric" in df.columns and "bibliometrics" not in df.columns:
        df = df.rename(columns={"metric": "bibliometrics"})

    # normalize meta columns if present
    tier_map = {"TOP": "HI", "LOW": "LI", "HI": "HI", "LI": "LI"}
    bib_map = {
        "HIGH": "HP", "LOW": "LP",
        "HP": "HP", "LP": "LP",
        # tolerate HB/LB if they appear anywhere
        "HB": "HP", "LB": "LP",
    }

    if "tier" in df.columns:
        df["tier"] = df["tier"].astype(str).map(lambda x: tier_map.get(x, x))
    if "bibliometrics" in df.columns:
        df["bibliometrics"] = df["bibliometrics"].astype(str).map(lambda x: bib_map.get(x, x))

    def norm_condition(c: str) -> str:
        c = str(c)
        if c == "NONE":
            return c
        parts = c.split("_")

        # Design A: G_TIER_BIB
        if len(parts) == 3:
            g, t, b = parts
            t2 = tier_map.get(t, t)      # TOP/LOW -> HI/LI
            b2 = bib_map.get(b, b)       # HIGH/LOW -> HP/LP (or HB/LB -> HP/LP)
            return f"{g}_{t2}_{b2}"

        # Design B: G_TIER
        if len(parts) == 2:
            g, t = parts
            t2 = tier_map.get(t, t)
            return f"{g}_{t2}"

        # anything else: leave unchanged (or raise if you prefer strictness)
        return c

    if "condition_id" in df.columns:
        df["condition_id"] = df["condition_id"].astype(str).map(norm_condition)

    return df


# -----------------------------------------------------------------------------
# Composites + FDR
# -----------------------------------------------------------------------------

def compute_composites_row_level(df: pd.DataFrame, likert_min: float, likert_max: float) -> pd.DataFrame:
    qcols = qcols_expected()
    for c in qcols:
        if c not in df.columns:
            raise ValueError(f"Missing {c} in input CSV.")
    df = coerce_numeric(df, qcols)

    qi_cols = ["q01", "q02", "q03", "q04", "q05", "q06", "q07"]
    qi = df[qi_cols].copy()
    qi["q03"] = reverse_score(qi["q03"], likert_min, likert_max)
    df["QI"] = qi.mean(axis=1)

    pi_cols = ["q08", "q09", "q10", "q11", "q12", "q13"]
    pi = df[pi_cols].copy()
    pi["q09"] = reverse_score(pi["q09"], likert_min, likert_max)
    df["PI"] = pi.mean(axis=1)

    df["AR"] = df["q14"]
    return df


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    p_ok = p[ok]
    if p_ok.size == 0:
        return out
    order = np.argsort(p_ok)
    ranked = p_ok[order]
    m = len(ranked)
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out_ok = np.empty_like(q)
    out_ok[order] = q
    out[ok] = out_ok
    return out


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

def aggregate_iterations(df: pd.DataFrame) -> pd.DataFrame:
    qcols = qcols_expected()
    score_cols = qcols + ["QI", "PI", "AR"]

    # UPDATED meta column set: tier + bibliometrics (but still tolerate metric)
    meta_cols = [c for c in [
        "model", "temperature", "group", "tier", "bibliometrics",
        "name", "institution", "h_index", "citations",
        "clean_textID", "clean_domain", "clean_len"
    ] if c in df.columns]

    group_keys = ["text_id", "condition_id"]
    if "model" in df.columns:
        group_keys.append("model")

    return (
        df.groupby(group_keys, as_index=False)
          .agg({**{c: "mean" for c in score_cols}, **{c: "first" for c in meta_cols}})
    )


def aggregate_models_equal_weight(agg: pd.DataFrame, scores: List[str]) -> pd.DataFrame:
    """Pool models equally within (text_id, condition_id). Used for Δ vs NONE outputs."""
    if "model" not in agg.columns:
        return agg

    meta_cols = [c for c in [
        "temperature", "group", "tier", "bibliometrics",
        "name", "institution", "h_index", "citations",
        "clean_textID", "clean_domain", "clean_len"
    ] if c in agg.columns]

    return (
        agg.groupby(["text_id", "condition_id"], as_index=False)
           .agg({**{c: "mean" for c in scores}, **{c: "first" for c in meta_cols}})
    )


# -----------------------------------------------------------------------------
# Wide tables
# -----------------------------------------------------------------------------

def wide_scores(agg: pd.DataFrame, scores: List[str]) -> pd.DataFrame:
    if "model" in agg.columns:
        return agg.pivot(index=["model", "text_id"], columns="condition_id", values=scores)
    return agg.pivot(index="text_id", columns="condition_id", values=scores)


def condition_order(conds: List[str]) -> List[str]:
    rest = sorted([c for c in conds if c != NONE_ID])
    return ([NONE_ID] + rest) if (NONE_ID in conds) else rest


# -----------------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------------

def bootstrap_mean_ci(d: np.ndarray, B: int = 10000, seed: int = 1) -> Tuple[float, float]:
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def sign_flip_pvalue(d: np.ndarray, mc: int = 200_000, seed: int = 1) -> float:
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = float(np.mean(d))
    signs = rng.choice([-1, 1], size=(mc, len(d)))
    sims = (signs * d[None, :]).mean(axis=1)
    return float((np.abs(sims) >= abs(obs)).mean())


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def tost_equivalence_test(d: np.ndarray, delta: float, alpha: float = 0.05) -> Dict[str, float]:
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return {"n": n, "mean": np.nan, "tost_p": np.nan, "p_lo": np.nan, "p_hi": np.nan}

    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = sd / np.sqrt(n)

    if se == 0:
        equivalent = (-delta <= mean <= delta)
        return {
            "n": n, "mean": mean, "delta": float(delta),
            "p_lo": 0.0 if mean > -delta else 1.0,
            "p_hi": 0.0 if mean <  delta else 1.0,
            "tost_p": 0.0 if equivalent else 1.0,
            "equivalent": bool(equivalent),
        }

    t_lo = (mean - (-delta)) / se
    p_lo = 1 - stats.t.cdf(t_lo, df=n-1)

    t_hi = (mean - (delta)) / se
    p_hi = stats.t.cdf(t_hi, df=n-1)

    tost_p = max(p_lo, p_hi)
    eq = (p_lo < alpha) and (p_hi < alpha)
    return {
        "n": n, "mean": mean, "delta": float(delta),
        "p_lo": float(p_lo), "p_hi": float(p_hi),
        "tost_p": float(tost_p), "equivalent": bool(eq),
    }


def bootstrap_mean_ci_1d(x: np.ndarray, B: int = 10000, seed: int = 1) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def sign_flip_pvalue_1d(x: np.ndarray, mc: int = 200_000, seed: int = 1) -> float:
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = float(np.mean(x))
    signs = rng.choice([-1, 1], size=(mc, n))
    sims = (signs * x[None, :]).mean(axis=1)
    return float((np.abs(sims) >= abs(obs)).mean())


def domain_heterogeneity_perm_pvalues(
    per_text_df: pd.DataFrame,
    text_to_domain: pd.Series,
    *,
    outcomes: List[str],
    effects: List[str],
    B: int = 20_000,
    seed: int = 1,
    domain_name: str = "clean_domain",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = per_text_df.copy()
    if "text_id" not in df.columns or "outcome" not in df.columns or "effect" not in df.columns or "value" not in df.columns:
        raise ValueError("per_text_df must have columns: text_id, outcome, effect, value")

    df = df.merge(
        text_to_domain.rename(domain_name),
        left_on="text_id",
        right_index=True,
        how="inner",
    )

    rows = []
    for out in outcomes:
        for eff in effects:
            sub = df[(df["outcome"] == out) & (df["effect"] == eff)].copy()
            x = sub["value"].to_numpy(float)
            dom = sub[domain_name].astype(str).to_numpy()

            ok = np.isfinite(x) & pd.notna(dom)
            x = x[ok]
            dom = dom[ok]
            n = int(len(x))
            k = int(len(np.unique(dom)))

            if n < 8 or k < 2:
                rows.append({
                    "outcome": out,
                    "effect": eff,
                    "p_perm": np.nan,
                    "stat_obs": np.nan,
                    "B": int(B),
                    "n_texts": n,
                    "k_domains": k,
                })
                continue

            mu = float(np.mean(x))

            # observed BSS
            stat_obs = 0.0
            for d in np.unique(dom):
                xd = x[dom == d]
                if len(xd) == 0:
                    continue
                stat_obs += len(xd) * (float(np.mean(xd)) - mu) ** 2

            # permutations
            ge = 0
            # precompute unique labels to keep groups stable in loop (speed)
            uniq = np.unique(dom)
            for _ in range(int(B)):
                dom_perm = rng.permutation(dom)
                stat = 0.0
                for d in uniq:
                    xd = x[dom_perm == d]
                    if len(xd) == 0:
                        continue
                    stat += len(xd) * (float(np.mean(xd)) - mu) ** 2
                if stat >= stat_obs - 1e-12:
                    ge += 1

            # +1 smoothing
            p = (ge + 1) / (int(B) + 1)

            rows.append({
                "outcome": out,
                "effect": eff,
                "p_perm": float(p),
                "stat_obs": float(stat_obs),
                "B": int(B),
                "n_texts": n,
                "k_domains": k,
            })

    return pd.DataFrame(rows)


def attach_domain_perm_pvalues_to_domain_effects_csv(
    *,
    per_text_path: str,
    aggregated_path: str,
    domain_effects_csv_path: str,
    outcomes: List[str],
    effects: List[str],
    B: int = 20_000,
    seed: int = 1,
    domain_col: str = "clean_domain",
    text_col: str = "text_id",
    overwrite: bool = True,
) -> pd.DataFrame:

    per_text = pd.read_csv(per_text_path)
    agg = pd.read_csv(aggregated_path)

    if domain_col not in agg.columns:
        raise ValueError(f"'{domain_col}' not found in aggregated file. Available columns: {list(agg.columns)}")

    text_to_domain = (
        agg[[text_col, domain_col]]
        .drop_duplicates(text_col)
        .set_index(text_col)[domain_col]
    )

    pdom = domain_heterogeneity_perm_pvalues(
        per_text,
        text_to_domain,
        outcomes=outcomes,
        effects=effects,
        B=B,
        seed=seed,
        domain_name=domain_col,
    ).rename(columns={
        "B": "B_perm",
        "n_texts": "n_texts_perm",
        "k_domains": "k_domains_perm",
    })

    dom_df = pd.read_csv(domain_effects_csv_path)

    # merge and broadcast p-values to all rows for that (outcome,effect)
    dom_df = dom_df.merge(
        pdom[["outcome", "effect", "p_perm", "stat_obs", "B_perm", "n_texts_perm", "k_domains_perm"]],
        on=["outcome", "effect"],
        how="left",
    )

    if overwrite:
        dom_df.to_csv(domain_effects_csv_path, index=False)

    return dom_df

# -----------------------------------------------------------------------------
# Condition deltas vs NONE (compute only; plot reads table)
# -----------------------------------------------------------------------------

def cluster_bootstrap_ci_deltas_vs_none(
    wide: pd.DataFrame,
    conds: List[str],
    score_col: str,
    none_id: str = NONE_ID,
    B: int = 10000,
    seed: int = 1,
) -> Dict[str, Tuple[float, float, float, float, int]]:
    rng = np.random.default_rng(seed)
    texts = wide.index.to_numpy()
    n = len(texts)
    if n == 0:
        raise ValueError("No texts in wide table.")
    if none_id not in wide.columns.get_level_values(1):
        raise ValueError(f"NONE condition '{none_id}' not found in wide columns.")

    none = wide[(score_col, none_id)].to_numpy(dtype=float)
    out: Dict[str, Tuple[float, float, float, float, int]] = {}

    for c in conds:
        if c == none_id:
            out[c] = (0.0, 0.0, 0.0, np.nan, int(np.isfinite(none).sum()))
            continue

        v = wide[(score_col, c)].to_numpy(dtype=float)
        d = v - none
        d = d[np.isfinite(d)]
        n_eff = int(len(d))
        mean_d = float(np.mean(d)) if n_eff else np.nan

        boot = np.empty(B, dtype=float)
        base_v = wide[(score_col, c)].to_numpy(dtype=float)
        base_none = wide[(score_col, none_id)].to_numpy(dtype=float)
        for b in range(B):
            samp = rng.integers(0, n, size=n)
            d_b = base_v[samp] - base_none[samp]
            boot[b] = np.nanmean(d_b)

        ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])
        p = sign_flip_pvalue(v - none, mc=200_000, seed=seed)

        out[c] = (mean_d, float(ci_lo), float(ci_hi), float(p), n_eff)

    return out


def compute_condition_deltas_table(
    wide_comp: pd.DataFrame,
    composites: List[str],
    all_conds: List[str],
    boot: int,
    seed: int,
    none_id: str = NONE_ID,
) -> pd.DataFrame:
    rows = []
    for score in composites:
        est_all = cluster_bootstrap_ci_deltas_vs_none(
            wide=wide_comp,
            conds=all_conds,
            score_col=score,
            none_id=none_id,
            B=boot,
            seed=seed,
        )
        for cond, (m, lo, hi, p, n_eff) in est_all.items():
            rows.append({
                "score": score,
                "condition": cond,
                "mean": m,
                "ci_lo": lo,
                "ci_hi": hi,
                "p": p,
                "n_texts": n_eff,
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Per-text main effects & interactions (design-specific) — UPDATED LEVELS
# -----------------------------------------------------------------------------

def per_text_main_effects_no_none(wide: pd.DataFrame, score: str, design: DesignSpec) -> pd.DataFrame:
    if design.kind == "A_2x2x2":
        def cid(g, t, b): return f"{g}_{t}_{b}"
        out = pd.DataFrame(index=wide.index, columns=["BF_minus_WM", "HI_minus_LI", "HP_minus_LP"], dtype=float)

        bfwm, hilo, hplp = [], [], []
        for tx in wide.index:
            # BF − WM averaged over inst × bib
            vals = []
            for t in [INST_HI, INST_LI]:
                for b in [BIB_LI, BIB_HI]:
                    a = wide.loc[tx, (score, cid("BF", t, b))]
                    c = wide.loc[tx, (score, cid("WM", t, b))]
                    if pd.notna(a) and pd.notna(c):
                        vals.append(a - c)
            bfwm.append(np.mean(vals) if vals else np.nan)

            # HI − LI averaged over name × bib
            vals = []
            for g in ["BF", "WM"]:
                for b in [BIB_LI, BIB_HI]:
                    a = wide.loc[tx, (score, cid(g, INST_HI, b))]
                    c = wide.loc[tx, (score, cid(g, INST_LI, b))]
                    if pd.notna(a) and pd.notna(c):
                        vals.append(a - c)
            hilo.append(np.mean(vals) if vals else np.nan)

            # HP − LP averaged over name × inst
            vals = []
            for g in ["BF", "WM"]:
                for t in [INST_HI, INST_LI]:
                    a = wide.loc[tx, (score, cid(g, t, BIB_HI))]
                    c = wide.loc[tx, (score, cid(g, t, BIB_LI))]
                    if pd.notna(a) and pd.notna(c):
                        vals.append(a - c)
            hplp.append(np.mean(vals) if vals else np.nan)

        out["BF_minus_WM"] = bfwm
        out["HI_minus_LI"] = hilo
        out["HP_minus_LP"] = hplp
        return out

    # B_2x2
    def cid(g, t): return f"{g}_{t}"
    out = pd.DataFrame(index=wide.index, columns=["BF_minus_WM", "HI_minus_LI"], dtype=float)

    bfwm, hilo = [], []
    for tx in wide.index:
        vals = []
        for t in [INST_HI, INST_LI]:
            a = wide.loc[tx, (score, cid("BF", t))]
            c = wide.loc[tx, (score, cid("WM", t))]
            if pd.notna(a) and pd.notna(c):
                vals.append(a - c)
        bfwm.append(np.mean(vals) if vals else np.nan)

        vals = []
        for g in ["BF", "WM"]:
            a = wide.loc[tx, (score, cid(g, INST_HI))]
            c = wide.loc[tx, (score, cid(g, INST_LI))]
            if pd.notna(a) and pd.notna(c):
                vals.append(a - c)
        hilo.append(np.mean(vals) if vals else np.nan)

    out["BF_minus_WM"] = bfwm
    out["HI_minus_LI"] = hilo
    return out


def per_text_interactions_no_none(wide: pd.DataFrame, score: str, design: DesignSpec) -> pd.DataFrame:
    if design.kind == "A_2x2x2":
        def cid(g, t, b): return f"{g}_{t}_{b}"
        out = pd.DataFrame(index=wide.index, columns=["Name_x_Inst", "Name_x_Bibliometrics", "Inst_x_Bibliometrics"], dtype=float)

        name_x_inst, name_x_bib, inst_x_bib = [], [], []
        for tx in wide.index:
            # Name×Inst: [(HI-LI)_BF - (HI-LI)_WM] avg over bib
            vals = []
            for b in [BIB_LI, BIB_HI]:
                bf = wide.loc[tx, (score, cid("BF", INST_HI, b))] - wide.loc[tx, (score, cid("BF", INST_LI, b))]
                wm = wide.loc[tx, (score, cid("WM", INST_HI, b))] - wide.loc[tx, (score, cid("WM", INST_LI, b))]
                if pd.notna(bf) and pd.notna(wm):
                    vals.append(bf - wm)
            name_x_inst.append(np.mean(vals) if vals else np.nan)

            # Name×Bib: [(HP-LP)_BF - (HP-LP)_WM] avg over inst
            vals = []
            for t in [INST_HI, INST_LI]:
                bf = wide.loc[tx, (score, cid("BF", t, BIB_HI))] - wide.loc[tx, (score, cid("BF", t, BIB_LI))]
                wm = wide.loc[tx, (score, cid("WM", t, BIB_HI))] - wide.loc[tx, (score, cid("WM", t, BIB_LI))]
                if pd.notna(bf) and pd.notna(wm):
                    vals.append(bf - wm)
            name_x_bib.append(np.mean(vals) if vals else np.nan)

            # Inst×Bib: [(HI-LI)@HP - (HI-LI)@LP] avg over names
            vals = []
            for g in ["BF", "WM"]:
                hi = wide.loc[tx, (score, cid(g, INST_HI, BIB_HI))] - wide.loc[tx, (score, cid(g, INST_LI, BIB_HI))]
                lo = wide.loc[tx, (score, cid(g, INST_HI, BIB_LI))] - wide.loc[tx, (score, cid(g, INST_LI, BIB_LI))]
                if pd.notna(hi) and pd.notna(lo):
                    vals.append(hi - lo)
            inst_x_bib.append(np.mean(vals) if vals else np.nan)

        out["Name_x_Inst"] = name_x_inst
        out["Name_x_Bibliometrics"] = name_x_bib
        out["Inst_x_Bibliometrics"] = inst_x_bib
        return out

    # B_2x2: Name×Inst only
    def cid(g, t): return f"{g}_{t}"
    out = pd.DataFrame(index=wide.index, columns=["Name_x_Inst"], dtype=float)

    vals_all = []
    for tx in wide.index:
        bf_hi = wide.loc[tx, (score, cid("BF", INST_HI))]
        bf_li = wide.loc[tx, (score, cid("BF", INST_LI))]
        wm_hi = wide.loc[tx, (score, cid("WM", INST_HI))]
        wm_li = wide.loc[tx, (score, cid("WM", INST_LI))]
        if all(pd.notna(x) for x in [bf_hi, bf_li, wm_hi, wm_li]):
            vals_all.append(float((bf_hi - bf_li) - (wm_hi - wm_li)))
        else:
            vals_all.append(np.nan)

    out["Name_x_Inst"] = vals_all
    return out


# -----------------------------------------------------------------------------
# Effect & interaction tables (compute-only)
# -----------------------------------------------------------------------------

def compute_effect_table_no_none(wide: pd.DataFrame, outcomes: List[str], design: DesignSpec, B: int, seed: int) -> pd.DataFrame:
    rows = []
    for out in outcomes:
        per = per_text_main_effects_no_none(wide, out, design)
        for eff in design.effect_order:
            if eff not in per.columns:
                continue
            d = per[eff].to_numpy(dtype=float)
            est = float(np.nanmean(d))
            ci_lo, ci_hi = bootstrap_mean_ci(d, B=B, seed=seed)
            p = sign_flip_pvalue(d, mc=200_000, seed=seed)
            rows.append({
                "outcome": out,
                "effect": eff,
                "estimate": est,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p": p,
                "n_texts": int(np.isfinite(d).sum()),
            })
    return pd.DataFrame(rows)


def compute_interaction_table_no_none(wide: pd.DataFrame, outcomes: List[str], design: DesignSpec, B: int, seed: int) -> pd.DataFrame:
    rows = []
    for out in outcomes:
        per = per_text_interactions_no_none(wide, out, design)
        for eff in design.interaction_order:
            if eff not in per.columns:
                continue
            d = per[eff].to_numpy(dtype=float)
            est = float(np.nanmean(d))
            ci_lo, ci_hi = bootstrap_mean_ci(d, B=B, seed=seed)
            p = sign_flip_pvalue(d, mc=200_000, seed=seed)
            rows.append({
                "outcome": out,
                "interaction": eff,
                "estimate": est,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p": p,
                "n_texts": int(np.isfinite(d).sum()),
            })
    return pd.DataFrame(rows)


def export_per_text_effects(wide: pd.DataFrame, outcomes: List[str], design: DesignSpec, out_path: str) -> None:
    rows = []
    is_multi = isinstance(wide.index, pd.MultiIndex)

    for out in outcomes:
        per = per_text_main_effects_no_none(wide, out, design)
        inter = per_text_interactions_no_none(wide, out, design)

        def emit(df_eff: pd.DataFrame):
            if is_multi:
                tmp = df_eff.reset_index()  # model, text_id, ...
                for eff in [c for c in tmp.columns if c not in ("model", "text_id")]:
                    g = tmp.groupby("text_id", as_index=False)[eff].mean()
                    for _, r in g.iterrows():
                        rows.append({"text_id": r["text_id"], "outcome": out, "effect": eff, "value": float(r[eff]) if pd.notna(r[eff]) else np.nan})
            else:
                for eff in df_eff.columns:
                    for tx, val in df_eff[eff].items():
                        rows.append({"text_id": tx, "outcome": out, "effect": eff, "value": float(val) if pd.notna(val) else np.nan})

        emit(per)
        emit(inter)

    pd.DataFrame(rows).to_csv(out_path, index=False)


# -----------------------------------------------------------------------------
# Meta-analysis across models (compute-only)
# -----------------------------------------------------------------------------

def meta_analysis_random_effects(effects: np.ndarray, ses: np.ndarray) -> Dict[str, float]:
    effects = np.asarray(effects, dtype=float)
    ses = np.asarray(ses, dtype=float)
    m = np.isfinite(effects) & np.isfinite(ses) & (ses > 0)
    effects = effects[m]
    ses = ses[m]
    k = len(effects)
    if k == 0:
        return {"k": 0, "mu": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "tau2": np.nan, "Q": np.nan, "I2": np.nan}

    w = 1.0 / (ses ** 2)
    mu_fe = np.sum(w * effects) / np.sum(w)
    Q = np.sum(w * (effects - mu_fe) ** 2)
    df = max(k - 1, 1)
    C = np.sum(w) - (np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    w_re = 1.0 / (ses ** 2 + tau2)
    mu = np.sum(w_re * effects) / np.sum(w_re)
    se = np.sqrt(1.0 / np.sum(w_re))
    ci_lo = mu - 1.96 * se
    ci_hi = mu + 1.96 * se
    I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    return {"k": int(k), "mu": float(mu), "se": float(se), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "tau2": float(tau2), "Q": float(Q), "I2": float(I2)}


def modelwise_effects_for_outcomes(
    df_row: pd.DataFrame,
    outcomes: List[str],
    design: DesignSpec,
    likert_min: float,
    likert_max: float,
    B: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "model" not in df_row.columns:
        return pd.DataFrame(), pd.DataFrame()

    models = sorted([m for m in df_row["model"].dropna().unique()])
    if not models:
        return pd.DataFrame(), pd.DataFrame()

    model_rows = []
    for model_name in models:
        sub = df_row[df_row["model"] == model_name].copy()
        sub = compute_composites_row_level(sub, likert_min, likert_max)
        agg = aggregate_iterations(sub)
        wide = wide_scores(agg, outcomes)

        eff = compute_effect_table_no_none(wide, outcomes, design, B=B, seed=seed)
        eff["model"] = model_name

        eff["se_model"] = np.nan
        for out in outcomes:
            per = per_text_main_effects_no_none(wide, out, design)
            for effect in eff.loc[eff["outcome"] == out, "effect"].unique():
                if effect not in per.columns:
                    continue
                d = per[effect].to_numpy(float)
                d = d[np.isfinite(d)]
                se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) >= 2 else np.nan
                eff.loc[(eff["outcome"] == out) & (eff["effect"] == effect), "se_model"] = se

        model_rows.append(eff)

    model_df = pd.concat(model_rows, ignore_index=True)

    meta_rows = []
    if len(models) >= 2:
        for out in outcomes:
            for effect in model_df["effect"].unique():
                sub = model_df[(model_df["outcome"] == out) & (model_df["effect"] == effect)].copy()
                ma = meta_analysis_random_effects(sub["estimate"].to_numpy(float), sub["se_model"].to_numpy(float))
                meta_rows.append({"outcome": out, "effect": effect, **ma})

    return model_df, pd.DataFrame(meta_rows)


# -----------------------------------------------------------------------------
# LOTO token extraction (UPDATED patterns)
# -----------------------------------------------------------------------------

def extract_text_level_tokens_A(agg: pd.DataFrame) -> pd.DataFrame:
    hi_rows = agg[agg["condition_id"].str.contains(r"_HI_")][["text_id", "institution"]].drop_duplicates("text_id")
    li_rows = agg[agg["condition_id"].str.contains(r"_LI_")][["text_id", "institution"]].drop_duplicates("text_id")
    bf_rows = agg[agg["condition_id"].str.startswith("BF_")][["text_id", "name"]].drop_duplicates("text_id")
    wm_rows = agg[agg["condition_id"].str.startswith("WM_")][["text_id", "name"]].drop_duplicates("text_id")

    tok = pd.DataFrame({"text_id": agg["text_id"].unique()})
    tok = tok.merge(hi_rows.rename(columns={"institution": "hi_institution"}), on="text_id", how="left")
    tok = tok.merge(li_rows.rename(columns={"institution": "li_institution"}), on="text_id", how="left")
    tok = tok.merge(bf_rows.rename(columns={"name": "bf_name"}), on="text_id", how="left")
    tok = tok.merge(wm_rows.rename(columns={"name": "wm_name"}), on="text_id", how="left")

    lp = agg[agg["condition_id"].str.endswith("_LP")][["text_id", "h_index", "citations"]].drop_duplicates("text_id")
    hp = agg[agg["condition_id"].str.endswith("_HP")][["text_id", "h_index", "citations"]].drop_duplicates("text_id")
    tok = tok.merge(lp.rename(columns={"h_index": "lp_h_index", "citations": "lp_citations"}), on="text_id", how="left")
    tok = tok.merge(hp.rename(columns={"h_index": "hp_h_index", "citations": "hp_citations"}), on="text_id", how="left")
    return tok


def extract_text_level_tokens_B(agg: pd.DataFrame) -> pd.DataFrame:
    hi_rows = agg[agg["condition_id"].str.endswith("_HI")][["text_id", "institution"]].drop_duplicates("text_id")
    li_rows = agg[agg["condition_id"].str.endswith("_LI")][["text_id", "institution"]].drop_duplicates("text_id")
    bf_rows = agg[agg["condition_id"].str.startswith("BF_")][["text_id", "name"]].drop_duplicates("text_id")
    wm_rows = agg[agg["condition_id"].str.startswith("WM_")][["text_id", "name"]].drop_duplicates("text_id")

    tok = pd.DataFrame({"text_id": agg["text_id"].unique()})
    tok = tok.merge(hi_rows.rename(columns={"institution": "hi_institution"}), on="text_id", how="left")
    tok = tok.merge(li_rows.rename(columns={"institution": "li_institution"}), on="text_id", how="left")
    tok = tok.merge(bf_rows.rename(columns={"name": "bf_name"}), on="text_id", how="left")
    tok = tok.merge(wm_rows.rename(columns={"name": "wm_name"}), on="text_id", how="left")
    return tok


def loto_tokens(
    agg: pd.DataFrame,
    design: DesignSpec,
    out_dir: str,
    B: int,
    seed: int,
) -> None:
    composites = list(design.loto_outcomes)

    if design.kind == "A_2x2x2":
        tok = extract_text_level_tokens_A(agg)

        # institution tokens for HI_minus_LI
        rows = []
        for col in ["hi_institution", "li_institution"]:
            for token in sorted(tok[col].dropna().unique()):
                keep_texts = tok.loc[tok[col] != token, "text_id"].dropna().unique()
                sub = agg[agg["text_id"].isin(keep_texts)].copy()
                w = wide_scores(sub, composites)
                eff = compute_effect_table_no_none(w, composites, design, B=B, seed=seed)
                eff = eff[eff["effect"] == "HI_minus_LI"].copy()
                for _, r in eff.iterrows():
                    rows.append({
                        "dropped_from": col,
                        "dropped_token": token,
                        "outcome": r["outcome"],
                        "estimate": r["estimate"],
                        "ci_lo": r["ci_lo"],
                        "ci_hi": r["ci_hi"],
                        "p": r["p"],
                        "n_texts": int(len(keep_texts)),
                    })
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "loto_institution_QI_PI_AR.csv"), index=False)

        # name tokens for BF_minus_WM
        rows = []
        for col in ["bf_name", "wm_name"]:
            for token in sorted(tok[col].dropna().unique()):
                keep_texts = tok.loc[tok[col] != token, "text_id"].dropna().unique()
                sub = agg[agg["text_id"].isin(keep_texts)].copy()
                w = wide_scores(sub, composites)
                eff = compute_effect_table_no_none(w, composites, design, B=B, seed=seed)
                eff = eff[eff["effect"] == "BF_minus_WM"].copy()
                for _, r in eff.iterrows():
                    rows.append({
                        "dropped_from": col,
                        "dropped_token": token,
                        "outcome": r["outcome"],
                        "estimate": r["estimate"],
                        "ci_lo": r["ci_lo"],
                        "ci_hi": r["ci_hi"],
                        "p": r["p"],
                        "n_texts": int(len(keep_texts)),
                    })
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "loto_name_QI_PI_AR.csv"), index=False)
        return

    # B_2x2
    tok = extract_text_level_tokens_B(agg)

    # institution tokens for HI_minus_LI
    rows = []
    for col in ["hi_institution", "li_institution"]:
        for token in sorted(tok[col].dropna().unique()):
            keep = tok.loc[tok[col] != token, "text_id"].dropna().unique()
            sub = agg[agg["text_id"].isin(keep)].copy()
            w = wide_scores(sub, composites)
            eff = compute_effect_table_no_none(w, composites, design, B=B, seed=seed)
            eff = eff[eff["effect"] == "HI_minus_LI"]
            for _, r in eff.iterrows():
                rows.append({
                    "dropped_from": col, "dropped_token": token,
                    "outcome": r["outcome"],
                    "estimate": r["estimate"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p"],
                    "n_texts": int(len(keep)),
                })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "loto_institution_PI_AR.csv"), index=False)

    # name tokens for BF_minus_WM
    rows = []
    for col in ["bf_name", "wm_name"]:
        for token in sorted(tok[col].dropna().unique()):
            keep = tok.loc[tok[col] != token, "text_id"].dropna().unique()
            sub = agg[agg["text_id"].isin(keep)].copy()
            w = wide_scores(sub, composites)
            eff = compute_effect_table_no_none(w, composites, design, B=B, seed=seed)
            eff = eff[eff["effect"] == "BF_minus_WM"]
            for _, r in eff.iterrows():
                rows.append({
                    "dropped_from": col, "dropped_token": token,
                    "outcome": r["outcome"],
                    "estimate": r["estimate"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p"],
                    "n_texts": int(len(keep)),
                })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "loto_name_PI_AR.csv"), index=False)


# domain only (cached)
def compute_domain_effects_table(
    *,
    per_text_path: str,
    aggregated_path: str,
    outcomes: List[str],
    effects: List[str],
    out_csv: str,
    B: int = 10000,
    seed: int = 1,
    domain_col: str = "clean_domain",
    text_col: str = "text_id",
    effect_col: str = "effect",
    outcome_col: str = "outcome",
    value_col: str = "value",
) -> pd.DataFrame:
    per = pd.read_csv(per_text_path)
    agg = pd.read_csv(aggregated_path)

    if domain_col not in agg.columns:
        raise ValueError(f"'{domain_col}' not found in aggregated file. Available: {list(agg.columns)}")

    # map text -> domain (one row per text)
    td = agg[[text_col, domain_col]].drop_duplicates(text_col).copy()

    per = per.merge(td, on=text_col, how="inner")
    per = per[per[outcome_col].isin(outcomes) & per[effect_col].isin(effects)].copy()

    if per.empty:
        raise ValueError("No rows after filtering by outcomes/effects and merging domains.")

    domains = sorted(per[domain_col].dropna().unique().tolist())

    rows = []
    for out in outcomes:
        for eff in effects:
            sub = per[(per[outcome_col] == out) & (per[effect_col] == eff)]
            if sub.empty:
                continue

            # overall (all domains pooled)
            x_all = sub[value_col].to_numpy(float)
            est_all = float(np.nanmean(x_all))
            lo_all, hi_all = bootstrap_mean_ci_1d(x_all, B=B, seed=seed)
            p_all = sign_flip_pvalue_1d(x_all, mc=200_000, seed=seed)
            rows.append({
                "scope": "overall",
                "domain": "ALL",
                "outcome": out,
                "effect": eff,
                "estimate": est_all,
                "ci_lo": lo_all,
                "ci_hi": hi_all,
                "p": p_all,
                "n_texts": int(np.isfinite(x_all).sum()),
            })

            # per-domain
            for d in domains:
                sd = sub[sub[domain_col] == d]
                x = sd[value_col].to_numpy(float)
                est = float(np.nanmean(x))
                lo, hi = bootstrap_mean_ci_1d(x, B=B, seed=seed)
                p = sign_flip_pvalue_1d(x, mc=200_000, seed=seed)
                rows.append({
                    "scope": "domain",
                    "domain": str(d),
                    "outcome": out,
                    "effect": eff,
                    "estimate": est,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "p": p,
                    "n_texts": int(np.isfinite(x).sum()),
                })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out





# -----------------------------------------------------------------------------
# Plotting only (reads cached tables)
# -----------------------------------------------------------------------------

def _ncols_for_handles(n: int, max_cols: int = 4) -> int:
    if n <= 0:
        return 1
    return min(max_cols, n)

def add_legend_above(
    fig: plt.Figure,
    ax: plt.Axes,
    handles,
    labels,
    *,
    title: str | None = None,
    y: float = 1.02,
    max_cols: int = 4,
    fontsize: int = 9,
    title_fontsize: int = 9,
    frameon: bool = False,
) -> Legend:
    n = len(handles)
    ncol = _ncols_for_handles(n, max_cols=max_cols)
    leg = ax.legend(
        handles=handles,
        labels=labels,
        title=title,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=frameon,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
        borderaxespad=0.0,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    return leg

def add_two_legends_above(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    handles1,
    labels1,
    title1: str,
    handles2,
    labels2,
    title2: str,
    y_top: float = 1.16,
    y_bottom: float = 1.06,
    max_cols1: int = 4,
    max_cols2: int = 4,
    fontsize: int = 9,
    title_fontsize: int = 9,
    frameon: bool = False,
) -> tuple[Legend, Legend]:
    leg1 = add_legend_above(
        fig, ax, handles1, labels1,
        title=title1, y=y_top, max_cols=max_cols1,
        fontsize=fontsize, title_fontsize=title_fontsize, frameon=frameon
    )
    ax.add_artist(leg1)
    leg2 = add_legend_above(
        fig, ax, handles2, labels2,
        title=title2, y=y_bottom, max_cols=max_cols2,
        fontsize=fontsize, title_fontsize=title_fontsize, frameon=frameon
    )
    return leg1, leg2

def reserve_top_for_legends(fig: plt.Figure, *, n_legend_rows: int = 1) -> None:
    # Conservative: each legend row ~7% of height + title space
    top_margin = 0.08 * n_legend_rows + 0.02
    top_margin = min(0.30, max(0.12, top_margin))
    fig.tight_layout(rect=[0, 0, 1, 1 - top_margin])



def forest_plot_conditions_grouped_from_table(
    summary_df: pd.DataFrame,
    *,
    scores: List[str],
    conds: List[str],
    title: str,
    out_path: str,
    score_labels: Optional[Dict[str, str]] = None,
    none_id: str = "NONE",
    figsize: Optional[Tuple[float, float]] = None,
    # NEW:
    draw_separators: bool = True,
) -> None:
    score_labels = score_labels or {s: s for s in scores}
    conds_ord = [c for c in condition_order(list(conds)) if c != none_id]

    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H", "d", "p"]
    marker_map = {c: markers[i % len(markers)] for i, c in enumerate(conds_ord)}

    x = np.arange(len(scores))
    nC = len(conds_ord)
    offsets = np.array([0.0]) if nC == 1 else np.linspace(-0.25, 0.25, nC)

    if figsize is None:
        figsize = (10, (5 + 0.12 * max(0, nC - 6)) + 0.8)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, linewidth=1)

    for j, c in enumerate(conds_ord):
        ys, ylos, yhis, pvals = [], [], [], []
        for sc in scores:
            sub = summary_df[(summary_df["score"] == sc) & (summary_df["condition"] == c)]
            if sub.empty:
                ys.append(np.nan); ylos.append(np.nan); yhis.append(np.nan); pvals.append(np.nan)
            else:
                r = sub.iloc[0]
                ys.append(float(r["mean"]))
                ylos.append(float(r["ci_lo"]))
                yhis.append(float(r["ci_hi"]))
                pvals.append(float(r["p"]) if "p" in sub.columns else np.nan)

        ys = np.array(ys, dtype=float)
        ylos = np.array(ylos, dtype=float)
        yhis = np.array(yhis, dtype=float)
        ok = np.isfinite(ys) & np.isfinite(ylos) & np.isfinite(yhis)
        if not ok.any():
            continue

        yerr = np.vstack([ys[ok] - ylos[ok], yhis[ok] - ys[ok]])
        ax.errorbar(
            (x + offsets[j])[ok],
            ys[ok],
            yerr=yerr,
            fmt=marker_map[c],
            capsize=3,
            linewidth=1.5,
            markersize=6,
            label=str(c),
        )
        for xi, hi, p in zip((x + offsets[j])[ok], yhis[ok], np.array(pvals)[ok]):
            st = stars(p)
            if st:
                #ax.text(xi, hi + 0.02, st, ha="center", va="bottom")
                ax.text(xi, hi, st, ha="center", va="bottom")

    # NEW: separator boundaries between score groups
    if draw_separators and len(scores) > 1:
        bounds = x[:-1] + 0.5  # between ticks
        y0, y1 = ax.get_ylim()
        ax.vlines(bounds, y0, y1, linewidth=0.8, alpha=0.25, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([score_labels.get(sc, sc) for sc in scores])
    #ax.set_ylabel(f"Δ score vs {none_id} (paired within text; 95% bootstrap CI)")
    #ax.set_ylabel("Likert score")
    #ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        add_legend_above(fig, ax, handles, labels, title="Condition", y=1.02, max_cols=8, frameon=False)
        reserve_top_for_legends(fig, n_legend_rows=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def forest_plot_grouped_transposed(
    effects_table: pd.DataFrame,
    out_path: str,
    title: str,
    outcome_order: List[str],
    effect_order: List[str],
    effect_labels: Dict[str, str],
    outcome_labels: Dict[str, str] | None = None,
    p_col: str = "p",
    figsize: Optional[Tuple[float, float]] = None,
    # NEW:
    draw_separators: bool = True,
    rotate_xticks: int = 0,
    align_xticks: str = "center",
) -> None:
    df = effects_table.copy()
    df["effect"] = pd.Categorical(df["effect"], categories=effect_order, ordered=True)
    df["outcome"] = pd.Categorical(df["outcome"], categories=outcome_order, ordered=True)
    df = df.sort_values(["outcome", "effect"])

    outcomes = [o for o in outcome_order if (df["outcome"] == o).any()]
    x = np.arange(len(outcomes), dtype=float)

    nE = len(effect_order)
    offsets = np.array([0.0]) if nE == 1 else np.linspace(-0.25, 0.25, nE)

    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H", "d", "p"]
    marker_map = {e: markers[i % len(markers)] for i, e in enumerate(effect_order)}

    if figsize is None:
        figsize = (max(10, 0.45 * len(outcomes)), 6.8)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, linewidth=1)

    for j, eff in enumerate(effect_order):
        ys = np.full(len(outcomes), np.nan, dtype=float)
        ylos = np.full(len(outcomes), np.nan, dtype=float)
        yhis = np.full(len(outcomes), np.nan, dtype=float)
        pvals = np.full(len(outcomes), np.nan, dtype=float)

        for i, out in enumerate(outcomes):
            sub = df[(df["outcome"] == out) & (df["effect"] == eff)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            ys[i] = float(r["estimate"])
            ylos[i] = float(r["ci_lo"])
            yhis[i] = float(r["ci_hi"])
            pvals[i] = float(r[p_col]) if p_col in sub.columns else np.nan

        ok = np.isfinite(ys) & np.isfinite(ylos) & np.isfinite(yhis)
        if not ok.any():
            continue

        yerr = np.vstack([ys[ok] - ylos[ok], yhis[ok] - ys[ok]])
        ax.errorbar(
            (x + offsets[j])[ok],
            ys[ok],
            yerr=yerr,
            fmt=marker_map[eff],
            capsize=3,
            linewidth=1.5,
            markersize=6,
            label=effect_labels.get(str(eff), str(eff)),
        )

        for xi, hi, p in zip((x + offsets[j])[ok], yhis[ok], pvals[ok]):
            st = stars(p)
            if st:
                #ax.text(xi, hi + 0.02, st, ha="center", va="bottom")
                ax.text(xi, hi, st, ha="center", va="bottom")

    # NEW: separator boundaries between outcome groups
    if draw_separators and len(outcomes) > 1:
        bounds = x[:-1] + 0.5
        y0, y1 = ax.get_ylim()
        ax.vlines(bounds, y0, y1, linewidth=0.8, alpha=0.25, zorder=0)

    xtlabs = [outcome_labels.get(str(o), str(o)) for o in outcomes] if outcome_labels else [str(o) for o in outcomes]
    ax.set_xticks(x)
    ax.set_xticklabels(xtlabs, rotation=rotate_xticks, ha=align_xticks)
    #ax.set_ylabel("Effect size (paired within text; 95% bootstrap CI)")
    #ax.set_ylabel("Likert score")
    #ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        add_legend_above(fig, ax, handles, labels, title="Effect", y=1.02, max_cols=4, frameon=False)
        reserve_top_for_legends(fig, n_legend_rows=1)
        
    fig.tight_layout()  
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def forest_plot_model_heterogeneity(
    model_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    outcome: str,
    effect: str,
    out_path: str,
    title: str,
    model_col: str = "model",
    est_col: str = "estimate",
    se_col_candidates: List[str] = ["se_model", "se", "se_approx"],
    ci_cols: Tuple[str, str] = ("ci_lo", "ci_hi"),
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    sub = model_df[(model_df["outcome"] == outcome) & (model_df["effect"] == effect)].copy()
    if sub.empty:
        return

    # choose SE column if present
    se_col = None
    for c in se_col_candidates:
        if c in sub.columns and np.isfinite(sub[c]).any():
            se_col = c
            break

    # build CI if missing
    lo_name, hi_name = ci_cols
    if (lo_name not in sub.columns) or (hi_name not in sub.columns) or (not np.isfinite(sub[lo_name]).any()):
        if se_col is not None:
            sub[lo_name] = sub[est_col] - 1.96 * sub[se_col]
            sub[hi_name] = sub[est_col] + 1.96 * sub[se_col]
        else:
            # cannot plot without CI information
            return

    # sort models alphabetically
    sub = sub.sort_values(model_col)

    # meta summary row (if available)
    meta = meta_df[(meta_df["outcome"] == outcome) & (meta_df["effect"] == effect)].copy()
    has_meta = (not meta.empty) and all(c in meta.columns for c in ["mu", "ci_lo", "ci_hi"])
    tau2 = float(meta.iloc[0].get("tau2", np.nan)) if has_meta else np.nan
    I2 = float(meta.iloc[0].get("I2", np.nan)) if has_meta else np.nan

    labels = sub[model_col].astype(str).tolist()
    y = np.arange(len(labels), dtype=float)

    # Append summary row at bottom
    if has_meta:
        labels.append("Random-effects summary")
        y = np.arange(len(labels), dtype=float)

    fig_h = max(5, 0.35 * len(labels))
    if figsize is None:
        figsize = (10, fig_h)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, linewidth=1)

    # plot model rows
    for i, r in enumerate(sub.itertuples(index=False)):
        e = float(getattr(r, est_col))
        lo = float(getattr(r, lo_name))
        hi = float(getattr(r, hi_name))
        if np.isfinite(e) and np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], linewidth=2)
            ax.plot(e, i, marker="o")

    # summary row
    if has_meta:
        i = len(labels) - 1
        mu = float(meta.iloc[0]["mu"])
        lo = float(meta.iloc[0]["ci_lo"])
        hi = float(meta.iloc[0]["ci_hi"])
        ax.plot([lo, hi], [i, i], linewidth=3)
        ax.plot(mu, i, marker="D")

        # annotate heterogeneity
        txt = []
        if np.isfinite(tau2):
            txt.append(f"τ²={tau2:.3g}")
        if np.isfinite(I2):
            txt.append(f"I²={I2*100:.1f}%")
        if txt:
            ax.text(hi + 0.02, i, "  ".join(txt), va="center")

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    #ax.set_title(title)
    ax.set_xlabel("Effect size (paired within text; 95% CI)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def forest_plot_loto(
    loto_df: pd.DataFrame,
    *,
    baseline_df: Optional[pd.DataFrame],
    baseline_outcome: str,
    baseline_effect: str,
    out_path: str,
    title: str,
    token_col: str = "dropped_token",
    outcome_col: str = "outcome",
    effect_col: str = "effect",
    est_col: str = "estimate",
    lo_col: str = "ci_lo",
    hi_col: str = "ci_hi",
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    sub = loto_df[(loto_df[outcome_col] == baseline_outcome)].copy()
    if effect_col in sub.columns:
        sub = sub[sub[effect_col] == baseline_effect].copy()
    if sub.empty:
        return

    # baseline estimate (full sample)
    baseline = np.nan
    if baseline_df is not None:
        b = baseline_df[(baseline_df["outcome"] == baseline_outcome) & (baseline_df["effect"] == baseline_effect)]
        if not b.empty:
            baseline = float(b.iloc[0]["estimate"])

    # sort by estimate
    sub = sub.sort_values(est_col)

    labels = sub[token_col].astype(str).tolist()
    y = np.arange(len(labels), dtype=float)

    fig_h = max(6, 0.22 * len(labels))
    if figsize is None:
        figsize = (10, fig_h)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, linewidth=1)
    if np.isfinite(baseline):
        ax.axvline(baseline, linestyle="--", linewidth=1)

    for i, r in enumerate(sub.itertuples(index=False)):
        e = float(getattr(r, est_col))
        lo = float(getattr(r, lo_col))
        hi = float(getattr(r, hi_col))
        if np.isfinite(e) and np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], linewidth=2)
            ax.plot(e, i, marker="o")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    #ax.set_title(title)
    ax.set_xlabel("Re-estimated effect (95% CI) after dropping token")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_interactions_transposed(
    inter_df: pd.DataFrame,
    *,
    outcomes: List[str],
    interaction_order: List[str],
    out_path: str,
    title: str,
    figsize: Optional[Tuple[float, float]] = None,
    draw_separators: bool = True,
) -> None:
    df = inter_df.copy()
    # standardize expected columns
    if "interaction" not in df.columns:
        raise ValueError("inter_df must have column 'interaction'")
    df["interaction"] = pd.Categorical(df["interaction"], categories=interaction_order, ordered=True)
    df["outcome"] = pd.Categorical(df["outcome"], categories=outcomes, ordered=True)
    df = df.sort_values(["outcome", "interaction"])

    outs = [o for o in outcomes if (df["outcome"] == o).any()]
    x = np.arange(len(outs), dtype=float)

    nI = len(interaction_order)
    offsets = np.array([0.0]) if nI == 1 else np.linspace(-0.25, 0.25, nI)

    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]
    marker_map = {e: markers[i % len(markers)] for i, e in enumerate(interaction_order)}

    if figsize is None:
        figsize = (max(10, 0.45 * len(outs)), 6.8)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, linewidth=1)

    for j, inter in enumerate(interaction_order):
        ys = np.full(len(outs), np.nan, dtype=float)
        ylos = np.full(len(outs), np.nan, dtype=float)
        yhis = np.full(len(outs), np.nan, dtype=float)
        pvals = np.full(len(outs), np.nan, dtype=float)

        for i, out in enumerate(outs):
            sub = df[(df["outcome"] == out) & (df["interaction"] == inter)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            ys[i] = float(r["estimate"])
            ylos[i] = float(r["ci_lo"])
            yhis[i] = float(r["ci_hi"])
            pvals[i] = float(r["p"]) if "p" in sub.columns else np.nan

        ok = np.isfinite(ys) & np.isfinite(ylos) & np.isfinite(yhis)
        if not ok.any():
            continue

        yerr = np.vstack([ys[ok] - ylos[ok], yhis[ok] - ys[ok]])
        ax.errorbar(
            (x + offsets[j])[ok],
            ys[ok],
            yerr=yerr,
            fmt=marker_map[inter],
            capsize=3,
            linewidth=1.5,
            markersize=6,
            label=str(inter),
        )

        for xi, hi, p in zip((x + offsets[j])[ok], yhis[ok], pvals[ok]):
            st = stars(p)
            if st:
                #ax.text(xi, hi + 0.02, st, ha="center", va="bottom")
                ax.text(xi, hi, st, ha="center", va="bottom")

    if draw_separators and len(outs) > 1:
        bounds = x[:-1] + 0.5
        y0, y1 = ax.get_ylim()
        ax.vlines(bounds, y0, y1, linewidth=0.8, alpha=0.25, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(outs)
    #ax.set_ylabel("Interaction effect (paired within text; 95% CI)")
    #ax.set_ylabel("Likert score")
    #ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        add_legend_above(fig, ax, handles, labels, title="Interaction", y=1.02, max_cols=4, frameon=False)
        reserve_top_for_legends(fig, n_legend_rows=1)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def heterogeneity_grid_plot(
    model_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    outcomes: List[str],
    effects: List[str],
    effect_labels: Dict[str, str],
    out_path: str,
    title: str,
    model_col: str = "model",
    est_col: str = "estimate",
    se_col_candidates: List[str] = ["se_model", "se", "se_approx"],
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    # choose SE col globally (fallback if needed)
    se_col = None
    for c in se_col_candidates:
        if c in model_df.columns and np.isfinite(model_df[c]).any():
            se_col = c
            break

    nR = len(outcomes)
    nC = len(effects)

    if figsize is None:
        # heuristic: wider with cols, taller with number of models
        figsize = (4.5 * nC, 2.2 * nR)

    fig, axes = plt.subplots(nR, nC, figsize=figsize, squeeze=False)
    fig.suptitle(title)

    for r, out in enumerate(outcomes):
        for c, eff in enumerate(effects):
            ax = axes[r][c]
            sub = model_df[(model_df["outcome"] == out) & (model_df["effect"] == eff)].copy()
            ax.axvline(0, linewidth=1)

            if sub.empty:
                ax.set_axis_off()
                continue

            # CI columns or compute from SE
            if "ci_lo" not in sub.columns or "ci_hi" not in sub.columns or (not np.isfinite(sub["ci_lo"]).any()):
                if se_col is None:
                    ax.set_axis_off()
                    continue
                sub["ci_lo"] = sub[est_col] - 1.96 * sub[se_col]
                sub["ci_hi"] = sub[est_col] + 1.96 * sub[se_col]

            sub = sub.sort_values(model_col)
            labels = sub[model_col].astype(str).tolist()
            y = np.arange(len(labels), dtype=float)

            # plot models
            for i, rr in enumerate(sub.itertuples(index=False)):
                e = float(getattr(rr, est_col))
                lo = float(getattr(rr, "ci_lo"))
                hi = float(getattr(rr, "ci_hi"))
                if np.isfinite(e) and np.isfinite(lo) and np.isfinite(hi):
                    ax.plot([lo, hi], [i, i], linewidth=1.8)
                    ax.plot(e, i, marker="o")

            # meta summary
            meta = meta_df[(meta_df["outcome"] == out) & (meta_df["effect"] == eff)]
            if not meta.empty and all(k in meta.columns for k in ["mu", "ci_lo", "ci_hi"]):
                i = len(labels)
                mu = float(meta.iloc[0]["mu"])
                lo = float(meta.iloc[0]["ci_lo"])
                hi = float(meta.iloc[0]["ci_hi"])
                ax.plot([lo, hi], [i, i], linewidth=2.6)
                ax.plot(mu, i, marker="D")
                labels.append("RE")
                y = np.arange(len(labels), dtype=float)

                # annotate I2 if available
                I2 = float(meta.iloc[0].get("I2", np.nan))
                if np.isfinite(I2):
                    ax.text(hi + 0.02, i, f"I²={I2*100:.0f}%", va="center")

            ax.set_yticks(y)
            # Only show ytick labels on first column to reduce clutter
            if c == 0:
                ax.set_yticklabels(labels)
            else:
                ax.set_yticklabels([])

            ax.invert_yaxis()

            # panel title
            panel_title = effect_labels.get(eff, eff)
            if r == 0:
                ax.set_title(panel_title)

            # only bottom row shows x-label
            if r == nR - 1:
                ax.set_xlabel("Effect size")
            else:
                ax.set_xlabel("")

            # only first column shows outcome label (as y-axis label)
            if c == 0:
                ax.set_ylabel(out)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def heterogeneity_summary_prediction_intervals(
    meta_df: pd.DataFrame,
    *,
    outcomes: List[str],
    effects: List[str],
    effect_labels: Dict[str, str],
    out_path: str,
    title: str,
    figsize: Optional[Tuple[float, float]] = None,
    # if meta_df doesn't have mu/se, you can pass fallbacks; but with your pipeline it should.
    mu_col: str = "mu",
    se_col: str = "se",
    tau2_col: str = "tau2",
    ci_lo_col: str = "ci_lo",
    ci_hi_col: str = "ci_hi",
    show_separators: bool = True,
) -> None:
    df = meta_df.copy()

    # Keep only requested combos in the specified order
    rows = []
    for out in outcomes:
        for eff in effects:
            sub = df[(df["outcome"] == out) & (df["effect"] == eff)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            mu = float(r.get(mu_col, np.nan))
            se = float(r.get(se_col, np.nan))
            tau2 = float(r.get(tau2_col, np.nan))

            # CI: prefer stored, else compute from mu,se
            ci_lo = r.get(ci_lo_col, np.nan)
            ci_hi = r.get(ci_hi_col, np.nan)
            ci_lo = float(ci_lo) if np.isfinite(ci_lo) else (mu - 1.96 * se if np.isfinite(mu) and np.isfinite(se) else np.nan)
            ci_hi = float(ci_hi) if np.isfinite(ci_hi) else (mu + 1.96 * se if np.isfinite(mu) and np.isfinite(se) else np.nan)

            # Prediction interval (conservative; uses se_mu + tau^2)
            if np.isfinite(mu) and np.isfinite(se) and np.isfinite(tau2):
                pi_half = 1.96 * np.sqrt(max(0.0, tau2) + se**2)
                pi_lo = mu - pi_half
                pi_hi = mu + pi_half
            else:
                pi_lo = np.nan
                pi_hi = np.nan

            rows.append({
                "outcome": out,
                "effect": eff,
                "label": f"{out} · {effect_labels.get(eff, eff)}",
                "mu": mu,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "pi_lo": pi_lo,
                "pi_hi": pi_hi,
                "tau2": tau2,
                "I2": float(r.get("I2", np.nan)),
                "k": int(r.get("k", np.nan)) if pd.notna(r.get("k", np.nan)) else np.nan,
            })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    # y positions (one per outcome×effect)
    y = np.arange(len(plot_df), dtype=float)

    # figure size heuristic
    if figsize is None:
        figsize = (11, max(6, 0.32 * len(plot_df)))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, linewidth=1)

    # Draw prediction intervals first (lighter)
    for yi, r in zip(y, plot_df.itertuples(index=False)):
        if np.isfinite(r.pi_lo) and np.isfinite(r.pi_hi):
            ax.plot([r.pi_lo, r.pi_hi], [yi, yi], linewidth=3, alpha=0.25)

    # Draw CI + point estimate on top
    for yi, r in zip(y, plot_df.itertuples(index=False)):
        if np.isfinite(r.ci_lo) and np.isfinite(r.ci_hi) and np.isfinite(r.mu):
            ax.plot([r.ci_lo, r.ci_hi], [yi, yi], linewidth=2.2)
            ax.plot(r.mu, yi, marker="o")

            # annotate tau2/I2 to the right of CI
            ann = []
            if np.isfinite(r.tau2):
                ann.append(f"τ²={r.tau2:.3g}")
            if np.isfinite(r.I2):
                ann.append(f"I²={r.I2*100:.0f}%")
            if np.isfinite(r.k):
                ann.append(f"k={int(r.k)}")
            if ann:
                ax.text(r.ci_hi + 0.02, yi, "  ".join(ann), va="center")

    # Optional separators between outcome blocks
    if show_separators and len(outcomes) > 1:
        # find last index of each outcome block
        last_idx = []
        for out in outcomes:
            idxs = plot_df.index[plot_df["outcome"] == out].tolist()
            if idxs:
                last_idx.append(max(idxs))
        # boundaries after each block except last
        bounds = [i + 0.5 for i in last_idx[:-1]]
        y0, y1 = ax.get_ylim()
        ax.hlines(bounds, *ax.get_xlim(), linewidth=0.8, alpha=0.2)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"].tolist())
    ax.invert_yaxis()
    #ax.set_title(title)
    ax.set_xlabel("Random-effects mean (dot) with 95% CI; 95% prediction interval (thick pale line)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def heterogeneity_summary_prediction_intervals_transposed(
    meta_df: pd.DataFrame,
    *,
    outcomes: List[str],
    effects: List[str],
    effect_labels: Dict[str, str],
    out_path: str,
    title: str,
    figsize: Optional[Tuple[float, float]] = None,
    mu_col: str = "mu",
    se_col: str = "se",
    tau2_col: str = "tau2",
    ci_lo_col: str = "ci_lo",
    ci_hi_col: str = "ci_hi",
    # controls
    show_ann: bool = True,
    ann_fields: Tuple[str, ...] = ("I2", "tau2"),  # can include "k"
    ann_fontsize: int = 8,
    legend_outside: bool = True,
    show_separators: bool = True,
) -> None:
    df = meta_df.copy()

    # collect rows in requested order
    rows = []
    for out in outcomes:
        for eff in effects:
            sub = df[(df["outcome"] == out) & (df["effect"] == eff)]
            if sub.empty:
                continue
            r = sub.iloc[0]

            mu = float(r.get(mu_col, np.nan))
            se = float(r.get(se_col, np.nan))
            tau2 = float(r.get(tau2_col, np.nan))

            # CI: prefer stored, else compute
            ci_lo = r.get(ci_lo_col, np.nan)
            ci_hi = r.get(ci_hi_col, np.nan)
            ci_lo = float(ci_lo) if np.isfinite(ci_lo) else (mu - 1.96 * se if np.isfinite(mu) and np.isfinite(se) else np.nan)
            ci_hi = float(ci_hi) if np.isfinite(ci_hi) else (mu + 1.96 * se if np.isfinite(mu) and np.isfinite(se) else np.nan)

            # PI: mu ± 1.96*sqrt(tau^2 + se^2)
            if np.isfinite(mu) and np.isfinite(se) and np.isfinite(tau2):
                pi_half = 1.96 * np.sqrt(max(0.0, tau2) + se**2)
                pi_lo = mu - pi_half
                pi_hi = mu + pi_half
            else:
                pi_lo, pi_hi = np.nan, np.nan

            rows.append({
                "outcome": out,
                "effect": eff,
                "effect_label": effect_labels.get(eff, eff),
                "mu": mu,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "pi_lo": pi_lo,
                "pi_hi": pi_hi,
                "tau2": tau2,
                "I2": float(r.get("I2", np.nan)),
                "k": float(r.get("k", np.nan)),
            })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    # lock to requested order but only keep present categories
    outcomes_present = [o for o in outcomes if o in set(plot_df["outcome"])]
    effects_present  = [e for e in effects  if e in set(plot_df["effect"])]

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3"])
    color_for = {eff: prop_cycle[i % len(prop_cycle)] for i, eff in enumerate(effects_present)}

    n_out = len(outcomes_present)
    n_eff = max(1, len(effects_present))

    # fig size
    if figsize is None:
        figsize = (max(8.5, 1.7 * n_out), 6.6)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, linewidth=1)

    # x positions and dodging
    x_base = np.arange(n_out, dtype=float)
    dodge = 0.75 / n_eff
    offsets = (np.arange(n_eff) - (n_eff - 1) / 2.0) * dodge

    # marker map for effects (no colors)
    marker_cycle = ["o", "s", "D", "^", "v", "P", "X", "<", ">"]
    marker_for = {eff: marker_cycle[i % len(marker_cycle)] for i, eff in enumerate(effects_present)}

    # --- draw PI (behind) then CI then mu for each (outcome,effect)
    for j, eff in enumerate(effects_present):
        d = plot_df[plot_df["effect"] == eff].set_index("outcome").reindex(outcomes_present)

        for i, out in enumerate(outcomes_present):
            xx = x_base[i] + offsets[j]

            mu = float(d.loc[out, "mu"]) if out in d.index else np.nan
            ci_lo = float(d.loc[out, "ci_lo"]) if out in d.index else np.nan
            ci_hi = float(d.loc[out, "ci_hi"]) if out in d.index else np.nan
            pi_lo = float(d.loc[out, "pi_lo"]) if out in d.index else np.nan
            pi_hi = float(d.loc[out, "pi_hi"]) if out in d.index else np.nan

            col = color_for[eff]

            # PI
            if np.isfinite(pi_lo) and np.isfinite(pi_hi):
                ax.plot([xx, xx], [pi_lo, pi_hi],
                        linewidth=6, alpha=0.20, solid_capstyle="round",
                        color=col)

            # CI
            if np.isfinite(ci_lo) and np.isfinite(ci_hi):
                ax.plot([xx, xx], [ci_lo, ci_hi],
                        linewidth=2.2, solid_capstyle="round",
                        color=col)

            # mu
            if np.isfinite(mu):
                ax.plot(xx, mu,
                        marker=marker_for[eff],
                        linestyle="None",
                        color=col,
                        markerfacecolor=col,
                        markeredgecolor=col)

                # compact annotation above marker
                # compact annotation (vertical) slightly LEFT of the bar, centered on mu
                if show_ann:
                    ann = []
                    if "tau2" in ann_fields and np.isfinite(float(d.loc[out, "tau2"])):
                        ann.append(f"τ²={float(d.loc[out, 'tau2']):.2g}")
                    if "I2" in ann_fields and np.isfinite(float(d.loc[out, "I2"])):
                        ann.append(f"I²={float(d.loc[out, 'I2'])*100:.0f}%")
                    if "k" in ann_fields and np.isfinite(float(d.loc[out, "k"])):
                        ann.append(f"k={int(float(d.loc[out, 'k']))}")

                    if ann:
                        # x offset: a bit left of the drawn interval
                        # tweak factor 0.55–0.80 depending on how tight you want it
                        x_ann = xx - 0.10 * dodge

                        # y anchor: align to mu (or you could use 0.5*(ci_lo+ci_hi) if preferred)
                        y_ann = mu

                        ax.text(
                            x_ann, y_ann,
                            " ".join(ann),          # vertical stacking (line breaks)
                            rotation=90,             # rotate the whole block
                            ha="right", va="center", # "right" looks better when text sits left of the bar
                            fontsize=ann_fontsize,
                        )
    # x-axis
    ax.set_xticks(x_base)
    ax.set_xticklabels(outcomes_present, rotation=0)
    #ax.set_title(title)
    #ax.set_ylabel("Random-effects mean (μ) with 95% CI; PI shown as thick pale bar")
    #ax.set_ylabel("Likert score")
    #ax.set_xlabel("Outcome")

        # legends ABOVE (two rows)
    effect_handles = [
        plt.Line2D([0], [0],
                marker=marker_for[eff],
                linestyle="None",
                color=color_for[eff],
                markerfacecolor=color_for[eff],
                markeredgecolor=color_for[eff],
                label=effect_labels.get(eff, eff))
        for eff in effects_present
    ]
    style_handles = [
        plt.Line2D([0], [0], linewidth=6, alpha=0.20, label="95% prediction interval"),
        plt.Line2D([0], [0], linewidth=2.2, label="95% CI (μ)"),
#        plt.Line2D([0], [0], marker="o", linewidth=0, label="μ"),
    ]

    add_two_legends_above(
        fig, ax,
        handles1=effect_handles, labels1=[h.get_label() for h in effect_handles], title1="Effect",
        handles2=style_handles,  labels2=[h.get_label() for h in style_handles],  title2="Encoding",
        y_top=1.14, y_bottom=1.01,
        max_cols1=4, max_cols2=4,
        frameon=False,
    )
    # separators between outcomes
    if show_separators and len(outcomes_present) > 1:
        bounds = x_base[:-1] + 0.5
        y0, y1 = ax.get_ylim()
        ax.vlines(bounds, y0, y1, linewidth=0.8, alpha=0.25, zorder=0)

    reserve_top_for_legends(fig, n_legend_rows=2)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# domain plots

# ---------- plotting option A: grid (domains as "studies") ----------

def domain_heterogeneity_grid_plot(
    domain_df: pd.DataFrame,
    *,
    outcomes: List[str],
    effects: List[str],
    effect_labels: Dict[str, str],
    out_path: str,
    title: str,
    figsize: Optional[Tuple[float, float]] = None,
    draw_zero: bool = True,
    show_ci: bool = True,
    show_overall: bool = True,
) -> None:
    df = domain_df.copy()

    # domains (exclude ALL unless show_overall)
    domains = sorted([d for d in df["domain"].unique().tolist() if d != "ALL"])
    if show_overall:
        domains_plus = domains + ["ALL"]
    else:
        domains_plus = domains

    nR = len(outcomes)
    nC = len(effects)
    if figsize is None:
        figsize = (4.8 * nC, 2.4 * nR)

    fig, axes = plt.subplots(nR, nC, figsize=figsize, squeeze=False)
    fig.suptitle(title)

    for r, out in enumerate(outcomes):
        for c, eff in enumerate(effects):
            ax = axes[r][c]
            sub = df[(df["outcome"] == out) & (df["effect"] == eff)].copy()
            if sub.empty:
                ax.set_axis_off()
                continue

            if draw_zero:
                ax.axvline(0, linewidth=1)

            # build rows in desired order
            rows = []
            for d in domains:
                s = sub[sub["domain"] == d]
                if not s.empty:
                    rows.append(s.iloc[0])
            if show_overall:
                s = sub[sub["domain"] == "ALL"]
                if not s.empty:
                    rows.append(s.iloc[0])

            labels = [str(rr["domain"]) for rr in rows]
            y = np.arange(len(rows), dtype=float)

            for i, rr in enumerate(rows):
                e = float(rr["estimate"])
                lo = float(rr.get("ci_lo", np.nan))
                hi = float(rr.get("ci_hi", np.nan))

                if show_ci and np.isfinite(lo) and np.isfinite(hi):
                    ax.plot([lo, hi], [i, i], linewidth=2.4 if rr["domain"] == "ALL" else 1.8)
                ax.plot(e, i, marker="D" if rr["domain"] == "ALL" else "o")

            ax.set_yticks(y)
            if c == 0:
                ax.set_yticklabels(labels)
            else:
                ax.set_yticklabels([])
            ax.invert_yaxis()

            if r == 0:
                ax.set_title(effect_labels.get(eff, eff))
            if r == nR - 1:
                ax.set_xlabel("Effect size")
            if c == 0:
                ax.set_ylabel(out)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------- plotting option C: compact summary with "domain spread" ----------

from typing import Optional, Tuple, List, Dict

def domain_heterogeneity_summary_spread_transposed(
    domain_df: pd.DataFrame,
    *,
    outcomes: List[str],
    effects: List[str],
    effect_labels: Dict[str, str],
    out_path: str,
    title: str,
    figsize: Optional[Tuple[float, float]] = None,
    # column names (expected from compute_domain_effects_table output)
    outcome_col: str = "outcome",
    effect_col: str = "effect",
    domain_col: str = "domain",
    est_col: str = "estimate",
    ci_lo_col: str = "ci_lo",
    ci_hi_col: str = "ci_hi",
    p_col: str = "p_perm",  # NEW (added via merge in B)
    # controls
    show_ann: bool = True,
    ann_fields: Tuple[str, ...] = ("k", "p"),  # NEW: include "p"
    ann_fontsize: int = 8,
    show_separators: bool = True,
) -> None:
    df = domain_df.copy()

    # Collect rows in requested order
    rows = []
    for out in outcomes:
        for eff in effects:
            sub = df[(df[outcome_col] == out) & (df[effect_col] == eff)]
            if sub.empty:
                continue

            all_row = sub[sub[domain_col] == "ALL"]
            if all_row.empty:
                continue
            r_all = all_row.iloc[0]

            mu = float(r_all.get(est_col, np.nan))
            ci_lo = float(r_all.get(ci_lo_col, np.nan))
            ci_hi = float(r_all.get(ci_hi_col, np.nan))

            dom = sub[sub[domain_col] != "ALL"]
            dom_est = dom[est_col].to_numpy(float)
            dom_est = dom_est[np.isfinite(dom_est)]
            spread_lo = float(np.min(dom_est)) if dom_est.size else np.nan
            spread_hi = float(np.max(dom_est)) if dom_est.size else np.nan
            k = int(dom[domain_col].nunique())

            # permutation p-value broadcasted by merge (same for all rows within outcome×effect)
            p_perm = float(sub[p_col].iloc[0]) if (p_col in sub.columns and len(sub) > 0) else np.nan

            rows.append({
                "outcome": out,
                "effect": eff,
                "effect_label": effect_labels.get(eff, eff),
                "mu": mu,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "spread_lo": spread_lo,
                "spread_hi": spread_hi,
                "k": k,
                "p_perm": p_perm,
            })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    outcomes_present = [o for o in outcomes if o in set(plot_df["outcome"])]
    effects_present  = [e for e in effects  if e in set(plot_df["effect"])]

    # --- add right after marker_for is defined ---
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3"])
    color_for = {eff: prop_cycle[i % len(prop_cycle)] for i, eff in enumerate(effects_present)}

    n_out = len(outcomes_present)
    n_eff = max(1, len(effects_present))

    if figsize is None:
        figsize = (max(8.5, 1.7 * n_out), 6.6)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, linewidth=1)

    # x positions and dodging
    x_base = np.arange(n_out, dtype=float)
    dodge = 0.75 / n_eff
    offsets = (np.arange(n_eff) - (n_eff - 1) / 2.0) * dodge

    marker_cycle = ["o", "s", "D", "^", "v", "P", "X", "<", ">"]
    marker_for = {eff: marker_cycle[i % len(marker_cycle)] for i, eff in enumerate(effects_present)}

    for j, eff in enumerate(effects_present):
        d = plot_df[plot_df["effect"] == eff].set_index("outcome").reindex(outcomes_present)

        for i, out in enumerate(outcomes_present):
            xx = x_base[i] + offsets[j]
            if out not in d.index:
                continue

            mu = float(d.loc[out, "mu"])
            ci_lo = float(d.loc[out, "ci_lo"])
            ci_hi = float(d.loc[out, "ci_hi"])
            sp_lo = float(d.loc[out, "spread_lo"])
            sp_hi = float(d.loc[out, "spread_hi"])
            k = float(d.loc[out, "k"])
            p_perm = float(d.loc[out, "p_perm"]) if "p_perm" in d.columns else np.nan

            col = color_for[eff]

            # domain spread (behind)
            if np.isfinite(sp_lo) and np.isfinite(sp_hi):
                ax.plot([xx, xx], [sp_lo, sp_hi],
                        linewidth=6, alpha=0.20, solid_capstyle="round",
                        color=col)

            # overall CI (thin)
            if np.isfinite(ci_lo) and np.isfinite(ci_hi):
                ax.plot([xx, xx], [ci_lo, ci_hi],
                        linewidth=2.2, solid_capstyle="round",
                        color=col)

            # overall mean
            if np.isfinite(mu):
                ax.plot(xx, mu,
                        marker=marker_for[eff],
                        linestyle="None",
                        color=col,
                        markerfacecolor=col,
                        markeredgecolor=col)

                if show_ann:
                    ann = []
                    if "k" in ann_fields and np.isfinite(k):
                        ann.append(f"k={int(k)}")
                    if "p" in ann_fields and np.isfinite(p_perm):
                        ann.append(f"p={p_perm:.3g}")
                    if "spread" in ann_fields and np.isfinite(sp_lo) and np.isfinite(sp_hi):
                        ann.append(f"[{sp_lo:.2g},{sp_hi:.2g}]")

                    if ann:
                        x_ann = xx - 0.10 * dodge
                        y_ann = mu
                        ax.text(
                            x_ann, y_ann,
                            " ".join(ann),
                            rotation=90,
                            ha="right", va="center",
                            fontsize=ann_fontsize,
                        )

    # x-axis
    ax.set_xticks(x_base)
    ax.set_xticklabels(outcomes_present, rotation=0)
    #ax.set_title(title)
    #ax.set_ylabel("Overall mean (dot) with 95% CI; domain spread shown as thick pale bar")
    #ax.set_ylabel("Likert score")
    #ax.set_xlabel("Outcome")

    effect_handles = [
        plt.Line2D(
            [0], [0],
            marker=marker_for[eff],
            linestyle="None",
            color=color_for[eff],
            markerfacecolor=color_for[eff],
            markeredgecolor=color_for[eff],
            label=effect_labels.get(eff, eff),
        )
        for eff in effects_present
    ]
    style_handles = [
        plt.Line2D([0], [0], linewidth=6, alpha=0.20, label="Domain spread (min–max)"),
        plt.Line2D([0], [0], linewidth=2.2, label="95% CI (overall)"),
#        plt.Line2D([0], [0], marker="o", linewidth=0, label="Overall mean"),
    ]

    add_two_legends_above(
        fig, ax,
        handles1=effect_handles, labels1=[h.get_label() for h in effect_handles], title1="Effect",
        handles2=style_handles,  labels2=[h.get_label() for h in style_handles],  title2="Encoding",
        y_top=1.14, y_bottom=1.01,
        max_cols1=4, max_cols2=4,
        frameon=False,
    )

    # separators between outcomes
    if show_separators and len(outcomes_present) > 1:
        bounds = x_base[:-1] + 0.5
        y0, y1 = ax.get_ylim()
        ax.vlines(bounds, y0, y1, linewidth=0.8, alpha=0.25, zorder=0)

    reserve_top_for_legends(fig, n_legend_rows=2)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



# -----------------------------------------------------------------------------
# Cache utilities
# -----------------------------------------------------------------------------

def cache_dir(out_dir: str) -> str:
    d = os.path.join(out_dir, "analysis_cache")
    os.makedirs(d, exist_ok=True)
    return d


def save_meta_json(path: str, meta: Dict) -> None:
    pd.Series(meta).to_json(path, indent=2)


# -----------------------------------------------------------------------------
# COMPUTE pipeline
# -----------------------------------------------------------------------------

def run_compute(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    cdir = cache_dir(args.out_dir)

    df = load_data(args.csv)
    df = normalize_factor_columns_and_conditions(df)  # NEW
    print("[DEBUG] condition_id sample:", sorted(df["condition_id"].unique())[:20])

    qcols = qcols_expected()
    df = coerce_numeric(df, qcols)
    df = compute_composites_row_level(df, args.likert_min, args.likert_max)

    agg = aggregate_iterations(df)
    agg.to_csv(os.path.join(cdir, "aggregated_by_text_condition.csv"), index=False)
    agg.to_csv(os.path.join(args.out_dir, "aggregated_by_text_condition.csv"), index=False)

    all_conds = sorted(agg["condition_id"].unique())
    design = infer_design_from_conditions(all_conds, none_id=NONE_ID)

    print("[INFO] inferred design:", design.kind)
    if "model" in agg.columns:
        print("[DEBUG] unique models:", sorted(agg["model"].dropna().unique()))
        print("[DEBUG] rows per model:", agg["model"].value_counts().to_dict())

    composites = ["QI", "PI", "AR"]
    outcomes_all = composites + qcols

    # A) Δ vs NONE on composites (model-pooled equal weight) -> summary table
    agg_pooled = aggregate_models_equal_weight(agg, scores=composites)
    wide_comp = wide_scores(agg_pooled, composites)

    summary_df = compute_condition_deltas_table(
        wide_comp=wide_comp,
        composites=composites,
        all_conds=all_conds,
        boot=args.boot,
        seed=args.seed,
        none_id=NONE_ID,
    )
    summary_df.to_csv(os.path.join(args.out_dir, "summary_conditions_composites.csv"), index=False)
    summary_df.to_csv(os.path.join(cdir, "summary_conditions_composites.csv"), index=False)

    # B/C) Main effects (no NONE) on composites + questions
    wide_all = wide_scores(agg, outcomes_all)

    eff_comp = compute_effect_table_no_none(wide_all, composites, design, B=args.boot, seed=args.seed)
    eff_comp.to_csv(os.path.join(args.out_dir, "main_effects_composites.csv"), index=False)
    eff_comp.to_csv(os.path.join(cdir, "main_effects_composites.csv"), index=False)

    eff_q = compute_effect_table_no_none(wide_all, qcols, design, B=args.boot, seed=args.seed)
    eff_q = eff_q.copy()
    eff_q["p_fdr"] = np.nan
    for eff in eff_q["effect"].unique():
        mask = eff_q["effect"] == eff
        eff_q.loc[mask, "p_fdr"] = bh_fdr(eff_q.loc[mask, "p"].to_numpy())

    eff_q.to_csv(os.path.join(args.out_dir, "main_effects_questions_fdr.csv"), index=False)
    eff_q.to_csv(os.path.join(cdir, "main_effects_questions_fdr.csv"), index=False)

    # per-text effects export
    export_per_text_effects(wide_all, outcomes_all, design, os.path.join(args.out_dir, "per_text_effects_long.csv"))
    export_per_text_effects(wide_all, outcomes_all, design, os.path.join(cdir, "per_text_effects_long.csv"))

    # interactions on composites (design-specific)
    inter_comp = compute_interaction_table_no_none(wide_all, composites, design, B=args.boot, seed=args.seed)
    inter_comp.to_csv(os.path.join(args.out_dir, "interaction_effects_composites.csv"), index=False)
    inter_comp.to_csv(os.path.join(cdir, "interaction_effects_composites.csv"), index=False)

    # equivalence tests for BF−WM on composites
    eq_rows = []
    for out in composites:
        per = per_text_main_effects_no_none(wide_all, out, design)
        if "BF_minus_WM" in per.columns:
            d = per["BF_minus_WM"].to_numpy(float)
            eq_rows.append({"outcome": out, **tost_equivalence_test(d, delta=args.sesoi)})
        else:
            eq_rows.append({"outcome": out, "n": int(np.isfinite(per.to_numpy()).sum()), "mean": np.nan, "tost_p": np.nan, "p_lo": np.nan, "p_hi": np.nan})

    eq_df = pd.DataFrame(eq_rows)
    eq_df.to_csv(os.path.join(args.out_dir, "main_effects_composites_equivalence.csv"), index=False)
    eq_df.to_csv(os.path.join(cdir, "main_effects_composites_equivalence.csv"), index=False)

    # model-wise + meta across models
    model_df_c, meta_df_c = modelwise_effects_for_outcomes(
        df_row=df,
        outcomes=composites,
        design=design,
        likert_min=args.likert_min,
        likert_max=args.likert_max,
        B=args.boot,
        seed=args.seed,
    )
    if not model_df_c.empty:
        model_df_c.to_csv(os.path.join(args.out_dir, "modelwise_effects_composites.csv"), index=False)
        model_df_c.to_csv(os.path.join(cdir, "modelwise_effects_composites.csv"), index=False)
    if not meta_df_c.empty:
        meta_df_c.to_csv(os.path.join(args.out_dir, "meta_effects_composites.csv"), index=False)
        meta_df_c.to_csv(os.path.join(cdir, "meta_effects_composites.csv"), index=False)

    model_df_q, meta_df_q = modelwise_effects_for_outcomes(
        df_row=df,
        outcomes=qcols,
        design=design,
        likert_min=args.likert_min,
        likert_max=args.likert_max,
        B=args.boot,
        seed=args.seed,
    )
    if not model_df_q.empty:
        model_df_q.to_csv(os.path.join(args.out_dir, "modelwise_effects_questions.csv"), index=False)
        model_df_q.to_csv(os.path.join(cdir, "modelwise_effects_questions.csv"), index=False)
    if not meta_df_q.empty:
        meta_df_q.to_csv(os.path.join(args.out_dir, "meta_effects_questions.csv"), index=False)
        meta_df_q.to_csv(os.path.join(cdir, "meta_effects_questions.csv"), index=False)

    # LOTO robustness
    if "institution" in agg.columns and "name" in agg.columns:
        loto_tokens(agg, design, out_dir=args.out_dir, B=args.boot, seed=args.seed)
        loto_tokens(agg, design, out_dir=cdir, B=args.boot, seed=args.seed)

    save_meta_json(
        os.path.join(cdir, "run_meta.json"),
        {
            "csv": args.csv,
            "likert_min": args.likert_min,
            "likert_max": args.likert_max,
            "boot": args.boot,
            "seed": args.seed,
            "sesoi": args.sesoi,
            "design": design.kind,
        },
    )

    # domain tables (as in your original)
    per_text_path = os.path.join(args.out_dir, "per_text_effects_long.csv")
    aggregated_path = os.path.join(args.out_dir, "aggregated_by_text_condition.csv")
    out_dom_csv = os.path.join(cache_dir(args.out_dir), "domain_effects_composites.csv")

    compute_domain_effects_table(
        per_text_path=per_text_path,
        aggregated_path=aggregated_path,
        outcomes=["QI", "PI", "AR"],
        effects=design.effect_order,
        out_csv=out_dom_csv,
        B=args.boot,
        seed=args.seed,
    )

    attach_domain_perm_pvalues_to_domain_effects_csv(
        per_text_path=os.path.join(args.out_dir, "per_text_effects_long.csv"),
        aggregated_path=os.path.join(args.out_dir, "aggregated_by_text_condition.csv"),
        domain_effects_csv_path=out_dom_csv,
        outcomes=["QI", "PI", "AR"],
        effects=design.effect_order,
        B=args.boot,
        seed=args.seed,
    )

    print("[DONE] Computations cached in:", cdir)
    print("[DONE] Outputs written to:", args.out_dir)


# -----------------------------------------------------------------------------
# PLOT pipeline (UPDATED baseline_effect names + titles)
# -----------------------------------------------------------------------------

def run_plot(args) -> None:
    cdir = cache_dir(args.out_dir)

    path_summary = os.path.join(cdir, "summary_conditions_composites.csv")
    path_eff_comp = os.path.join(cdir, "main_effects_composites.csv")
    path_eff_q = os.path.join(cdir, "main_effects_questions_fdr.csv")
    path_meta = os.path.join(cdir, "run_meta.json")

    for p in [path_summary, path_eff_comp, path_eff_q, path_meta]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing cached file: {p}. Run 'compute' first.")

    summary_df = pd.read_csv(path_summary)
    eff_comp = pd.read_csv(path_eff_comp)
    eff_q = pd.read_csv(path_eff_q)
    meta = pd.read_json(path_meta, typ="series").to_dict()

    all_conds = sorted(summary_df["condition"].unique().tolist())
    design = infer_design_from_conditions(all_conds, none_id=NONE_ID)

    composites = ["QI", "PI", "AR"]
    qcols = qcols_expected()

    plot_conds = [c for c in all_conds if c != NONE_ID]
    print(f"[INFO] conditions to plot (excluding {NONE_ID}): {plot_conds}")

    figsize = (9, 4) if len(plot_conds) > 4 else (6, 4)

    forest_plot_conditions_grouped_from_table(
        summary_df=summary_df,
        scores=composites,
        conds=plot_conds,
        title=f"Δ vs {NONE_ID} across texts (design={design.kind}; pooled equally over available models)",
        out_path=os.path.join(args.out_dir, "forest_conditions_grouped_QI_PI_AR.png"),
        score_labels={"QI": "QI", "PI": "PI", "AR": "AR"},
        none_id=NONE_ID,
        figsize=figsize,
    )

    figsize = (6, 4) if len(plot_conds) > 4 else (4.5, 4)

    forest_plot_grouped_transposed(
        eff_comp,
        os.path.join(args.out_dir, "forest_main_effects_composites_transposed.png"),
        f"Main effects (no {NONE_ID}) — Composites (grouped by outcome; design={design.kind})",
        outcome_order=composites,
        effect_order=design.effect_order,
        effect_labels=design.effect_labels,
        outcome_labels=None,
        p_col="p",
        figsize=figsize,
        rotate_xticks=0,
        align_xticks="center"
    )

    figsize = (15, 6) if len(plot_conds) > 4 else (13, 6)

    forest_plot_grouped_transposed(
        eff_q,
        os.path.join(args.out_dir, "forest_main_effects_questions_transposed.png"),
        f"Main effects (no {NONE_ID}) — Questions (grouped by question; design={design.kind}; BH-FDR stars)",
        outcome_order=qcols,
        effect_order=design.effect_order,
        effect_labels=design.effect_labels,
        outcome_labels=QUESTION_LABELS,
        p_col="p_fdr",
        figsize=figsize,
        rotate_xticks=45,
        align_xticks="right"
    )

    # Interaction effects on composites
    path_inter = os.path.join(cdir, "interaction_effects_composites.csv")
    if os.path.exists(path_inter):
        inter_df = pd.read_csv(path_inter)
        plot_interactions_transposed(
            inter_df,
            outcomes=["QI", "PI", "AR"],
            interaction_order=design.interaction_order,
            out_path=os.path.join(args.out_dir, "supp_interactions_composites_transposed.png"),
            title=f"Interaction effects (no {NONE_ID}) — Composites (design={design.kind})",
            draw_separators=True,
        )

    # LOTO robustness
    path_loto_inst_A = os.path.join(cdir, "loto_institution_QI_PI_AR.csv")
    path_loto_name_A = os.path.join(cdir, "loto_name_QI_PI_AR.csv")
    path_loto_inst_B = os.path.join(cdir, "loto_institution_PI_AR.csv")
    path_loto_name_B = os.path.join(cdir, "loto_name_PI_AR.csv")

    if os.path.exists(path_loto_inst_A):
        loto_inst = pd.read_csv(path_loto_inst_A)
        for out in ["QI", "PI", "AR"]:
            forest_plot_loto(
                loto_inst,
                baseline_df=eff_comp,
                baseline_outcome=out,
                baseline_effect="HI_minus_LI",  # UPDATED
                out_path=os.path.join(args.out_dir, f"supp_loto_institution_{out}.png"),
                title=f"LOTO (institution tokens): HI−LI on {out} (design={design.kind})",
            )

    if os.path.exists(path_loto_name_A):
        loto_name = pd.read_csv(path_loto_name_A)
        for out in ["QI", "PI", "AR"]:
            forest_plot_loto(
                loto_name,
                baseline_df=eff_comp,
                baseline_outcome=out,
                baseline_effect="BF_minus_WM",
                out_path=os.path.join(args.out_dir, f"supp_loto_name_{out}.png"),
                title=f"LOTO (name tokens): BF−WM on {out} (design={design.kind})",
            )

    if os.path.exists(path_loto_inst_B):
        loto_inst = pd.read_csv(path_loto_inst_B)
        for out in ["PI", "AR"]:
            forest_plot_loto(
                loto_inst,
                baseline_df=eff_comp,
                baseline_outcome=out,
                baseline_effect="HI_minus_LI",  # UPDATED
                out_path=os.path.join(args.out_dir, f"supp_loto_institution_{out}.png"),
                title=f"LOTO (institution tokens): HI−LI on {out} (design={design.kind})",
            )

    if os.path.exists(path_loto_name_B):
        loto_name = pd.read_csv(path_loto_name_B)
        for out in ["PI", "AR"]:
            forest_plot_loto(
                loto_name,
                baseline_df=eff_comp,
                baseline_outcome=out,
                baseline_effect="BF_minus_WM",
                out_path=os.path.join(args.out_dir, f"supp_loto_name_{out}.png"),
                title=f"LOTO (name tokens): BF−WM on {out} (design={design.kind})",
            )

    # Heterogeneity plots (unchanged; uses design.effect_order/labels)
    path_model_comp = os.path.join(cdir, "modelwise_effects_composites.csv")
    path_meta_comp = os.path.join(cdir, "meta_effects_composites.csv")
    if os.path.exists(path_model_comp) and os.path.exists(path_meta_comp):
        model_df_c = pd.read_csv(path_model_comp)
        meta_df_c = pd.read_csv(path_meta_comp)
        heterogeneity_grid_plot(
            model_df_c, meta_df_c,
            outcomes=["QI", "PI", "AR"],
            effects=design.effect_order,
            effect_labels=design.effect_labels,
            out_path=os.path.join(args.out_dir, "supp_heterogeneity_grid_composites.png"),
            title=f"Heterogeneity across evaluator models (design={design.kind})",
        )
        
        heterogeneity_summary_prediction_intervals(
            meta_df_c,
            outcomes=["QI", "PI", "AR"],
            effects=design.effect_order,
            effect_labels=design.effect_labels,
            out_path=os.path.join(args.out_dir, "supp_heterogeneity_summary_PI_composites.png"),
            title=f"Heterogeneity summary (random-effects mean + prediction intervals; design={design.kind})",
        )
        figsize = (6, 4) if len(plot_conds) > 4 else (5, 4)
        heterogeneity_summary_prediction_intervals_transposed(
            meta_df_c,
            outcomes=["QI", "PI", "AR"],
            figsize=figsize,
            effects=design.effect_order,
            effect_labels=design.effect_labels,
            out_path=os.path.join(args.out_dir, "supp_heterogeneity_summary_PI_composites_transposed.png"),
            title=f"Heterogeneity summary (random-effects mean + prediction intervals; design={design.kind})",
        )

    # Domain heterogeneity plots (unchanged; uses design.effect_order/labels)
    path_dom = os.path.join(cdir, "domain_effects_composites.csv")
    if os.path.exists(path_dom):
        dom_df = pd.read_csv(path_dom)
        domain_heterogeneity_grid_plot(
            dom_df,
            outcomes=["QI", "PI", "AR"],
            effects=design.effect_order,
            effect_labels=design.effect_labels,
            out_path=os.path.join(args.out_dir, "supp_domain_heterogeneity_grid_composites.png"),
            title=f"Domain heterogeneity (composites; design={design.kind})",
        )
        figsize = (6, 4) if len(plot_conds) > 4 else (5, 4)
        domain_heterogeneity_summary_spread_transposed(
            dom_df,
            outcomes=["QI", "PI", "AR"],
            figsize=figsize,
            effects=design.effect_order,
            effect_labels=design.effect_labels,
            out_path=os.path.join(args.out_dir, "supp_domain_heterogeneity_summary_transposed.png"),
            title=f"Domain heterogeneity summary (overall CI + domain spread; design={design.kind})",
            ann_fields=("k", "p"),
        )

    print("[DONE] Plots rendered from cache into:", args.out_dir)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_cli():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_c = sub.add_parser("compute", help="Run computations and cache results")
    ap_c.add_argument("--csv", required=True)
    ap_c.add_argument("--out_dir", required=True)
    ap_c.add_argument("--likert_min", type=float, default=-3)
    ap_c.add_argument("--likert_max", type=float, default=3)
    ap_c.add_argument("--boot", type=int, default=20000)
    ap_c.add_argument("--seed", type=int, default=1)
    ap_c.add_argument("--sesoi", type=float, default=0.10, help="Equivalence margin for BF−WM (TOST), in score units")

    ap_p = sub.add_parser("plot", help="Plot only (reads cached computation outputs)")
    ap_p.add_argument("--out_dir", required=True)

    return ap


def main():
    ap = build_cli()
    args = ap.parse_args()

    if args.cmd == "compute":
        run_compute(args)
    elif args.cmd == "plot":
        run_plot(args)
    else:
        raise ValueError(f"Unknown cmd={args.cmd}")


if __name__ == "__main__":
    main()