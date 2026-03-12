#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download arXiv papers (PDF) + extract text into per-paper folders, driven by one or
more ArxivBucket specs.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import random
import re
import time
from typing import Any, List, Optional, Sequence

import requests
import feedparser
from tqdm import tqdm
import fitz  # PyMuPDF

try:
    from langdetect import detect as _langdetect_detect
except Exception:
    _langdetect_detect = None

UTC = dt.timezone.utc
WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
PAGES_RE = re.compile(r"\b4\s*(pages?|pp)\b", re.IGNORECASE)


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_rmdir_if_empty(dir_path: str) -> None:
    try:
        if os.path.isdir(dir_path) and not os.listdir(dir_path):
            os.rmdir(dir_path)
    except Exception:
        pass

def safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def write_text_utf8(path: str, text: str) -> None:
    """
    Write text as valid UTF-8 no matter what (replace unencodable/surrogate chars).
    """
    b = (text or "").encode("utf-8", errors="replace")
    with open(path, "wb") as f:
        f.write(b)

def write_jsonl(path: str, items: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s.strip("_")[:150]

def normalize_ws(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # De-hyphenate "inter-\nface" -> "interface"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def words_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))

def is_english(text: str) -> bool:
    """
    Best-effort English detection:
      - langdetect if installed
      - otherwise: heuristic based on ASCII ratio + common stopwords
    """
    text = (text or "").strip()
    if len(text) < 200:
        return False

    if _langdetect_detect is not None:
        try:
            return _langdetect_detect(text) == "en"
        except Exception:
            pass

    ascii_letters = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
    if ascii_letters / max(1, len(text)) < 0.55:
        return False

    lower = text.lower()
    stopwords = [" the ", " and ", " of ", " to ", " in ", " we ", " with ", " for "]
    return sum(1 for sw in stopwords if sw in lower) >= 3


def http_get_with_backoff(
    session: requests.Session,
    url: str,
    params: dict,
    headers: dict,
    timeout: int = 60,
    max_retries: int = 12,
    base_sleep: float = 5.0,
    max_sleep: float = 180.0,
) -> requests.Response:
    """
    Retry on 429/5xx and on transient network errors, with exponential backoff + jitter.
    Honors Retry-After when present.
    """
    attempt = 0
    while True:
        try:
            r = session.get(url, params=params, headers=headers, timeout=timeout)
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = min(max_sleep, base_sleep * (2 ** (attempt - 1))) * (0.8 + 0.4 * random.random())
            print(f"[arXiv] network error {type(e).__name__}, retry {attempt}/{max_retries} in {sleep_s:.1f}s")
            time.sleep(sleep_s)
            continue

        if r.status_code not in (429, 500, 502, 503, 504):
            return r

        attempt += 1
        if attempt > max_retries:
            r.raise_for_status()

        retry_after = r.headers.get("Retry-After")
        if retry_after:
            try:
                sleep_s = float(retry_after)
            except ValueError:
                sleep_s = base_sleep * (2 ** (attempt - 1))
        else:
            sleep_s = base_sleep * (2 ** (attempt - 1))

        sleep_s = min(max_sleep, sleep_s) * (0.8 + 0.4 * random.random())
        print(f"[arXiv] HTTP {r.status_code} (start={params.get('start')}), retry {attempt}/{max_retries} in {sleep_s:.1f}s")
        time.sleep(sleep_s)


def comment_has_4_pages(entry: Any) -> bool:
    comment = getattr(entry, "arxiv_comment", None)
    if comment is None:
        try:
            comment = entry.get("arxiv_comment", "")
        except Exception:
            comment = ""
    comment = comment or ""
    return bool(PAGES_RE.search(comment))


# ----------------------------
# arXiv bucket
# ----------------------------

@dataclasses.dataclass
class ArxivBucket:
    """
    A retrieval "bucket" = a set of arXiv categories + optional filters, saved under
    out_root/papers/<domain>/...
    """
    domain: str
    n: int
    categories: List[str]

    days_back: int = 60
    min_words: int = 10
    max_words: int = 4000

    oversample_factor: int = 3
    request_delay_s: float = 3.0

    # If you use comment_phrase (API-side co:"..."), do NOT also local-filter.
    require_comment_4_pages: bool = False
    comment_phrase: Optional[str] = None

    use_submittedDate_filter: bool = True
    user_agent: str = "LLMer"


def _fmt_submitted_date_window(days_back: int) -> str:
    """
    arXiv submittedDate range in query syntax (NOT URL-encoded):
      [YYYYMMDDHHMM TO YYYYMMDDHHMM]
    """
    now = dt.datetime.now(UTC)
    start = now - dt.timedelta(days=days_back)
    return f"[{start:%Y%m%d%H%M} TO {now:%Y%m%d%H%M}]"


def arxiv_query(cfg: ArxivBucket) -> str:
    """
    Build arXiv API search_query with:
      - category constraint(s) via cat:
      - optional comments phrase via co:"..."
      - optional submittedDate window: submittedDate:[... TO ...]
    """
    if not cfg.categories:
        raise ValueError("ArxivBucket.categories must be a non-empty list of category strings (e.g., ['cs.AI']).")

    cats = [f"cat:{c.strip()}" for c in cfg.categories if str(c).strip()]
    if not cats:
        raise ValueError("ArxivBucket.categories contained only empty/blank entries.")

    cat_clause = cats[0] if len(cats) == 1 else "(" + " OR ".join(cats) + ")"
    clauses = [cat_clause]

    if cfg.comment_phrase:
        phrase = str(cfg.comment_phrase).replace('"', r"\"")
        clauses.append(f'co:"{phrase}"')

    if cfg.use_submittedDate_filter:
        clauses.append(f"submittedDate:{_fmt_submitted_date_window(cfg.days_back)}")

    return " AND ".join(clauses)


def fetch_arxiv_entries(cfg: ArxivBucket, session: requests.Session) -> List[dict]:
    base = "https://export.arxiv.org/api/query"
    q = arxiv_query(cfg)

    # Oversample to have enough candidates after filtering (language/word-count).
    target_candidates = cfg.n * max(1, min(int(cfg.oversample_factor), 20))

    per_page = 50
    start = 0
    cutoff = dt.datetime.now(UTC) - dt.timedelta(days=cfg.days_back)
    headers = {"User-Agent": cfg.user_agent}

    entries: List[dict] = []
    seen_ids: set[str] = set()

    # Cap the number of API pages we will scan (prevents runaway).
    max_start = 2000

    while len(entries) < target_candidates and start <= max_start:
        params = {
            "search_query": q,
            "start": start,
            "max_results": per_page,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        r = http_get_with_backoff(
            session=session,
            url=base,
            params=params,
            headers=headers,
            timeout=60,
            max_retries=12,
            base_sleep=max(cfg.request_delay_s, 3.0),
        )
        r.raise_for_status()

        feed = feedparser.parse(r.text)
        page_entries = feed.entries or []
        if not page_entries:
            break

        reached_end = len(page_entries) < per_page
        stop_due_to_cutoff = False

        for e in page_entries:
            try:
                published = dt.datetime.fromisoformat(str(e.published).replace("Z", "+00:00"))
            except Exception:
                published = None

            if published is not None and published < cutoff:
                stop_due_to_cutoff = True
                break

            # Avoid double filtering if comment_phrase is already in the API query.
            if cfg.require_comment_4_pages and cfg.comment_phrase is None:
                if not comment_has_4_pages(e):
                    continue

            comment = getattr(e, "arxiv_comment", None)
            if comment is None:
                try:
                    comment = e.get("arxiv_comment", "")
                except Exception:
                    comment = ""
            comment = comment or ""

            # Pull pdf link when available.
            pdf_url = None
            for link in getattr(e, "links", []) or []:
                if getattr(link, "type", "") == "application/pdf":
                    pdf_url = link.href
                    break
            if not pdf_url:
                pdf_url = str(e.id).replace("/abs/", "/pdf/") + ".pdf"

            arxiv_id = str(e.id).split("/abs/")[-1].strip()
            if not arxiv_id or arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)

            try:
                tags = [t.term for t in getattr(e, "tags", [])] if hasattr(e, "tags") else []
            except Exception:
                tags = []

            try:
                primary_cat = getattr(e, "arxiv_primary_category", None)
                primary_cat = primary_cat.get("term") if isinstance(primary_cat, dict) else None
            except Exception:
                primary_cat = None

            entries.append({
                "arxiv_id": arxiv_id,
                "title": getattr(e, "title", "").strip().replace("\n", " "),
                "published_utc": getattr(e, "published", None),
                "updated_utc": getattr(e, "updated", None),
                "authors": [a.name for a in getattr(e, "authors", [])] if hasattr(e, "authors") else [],
                "primary_category": primary_cat,
                "categories": tags,
                "pdf_url": pdf_url,
                "abs_url": str(e.id),
                "comment": comment,
            })

            if len(entries) >= target_candidates:
                break

        if stop_due_to_cutoff or reached_end:
            break

        start += per_page
        time.sleep(cfg.request_delay_s)

    return entries[:target_candidates]


def download_file(url: str, out_path: str, session: requests.Session, headers: dict, timeout: int = 120) -> None:
    with session.get(url, stream=True, headers=headers, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz) and normalize whitespace.
    """
    doc = None
    try:
        doc = fitz.open(pdf_path)
        parts: List[str] = []
        for page in doc:
            parts.append(page.get_text("text"))
        raw = "".join(parts)
        return normalize_ws(raw or "")
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def collect_arxiv_bucket(cfg: ArxivBucket, out_root: str) -> List[dict]:
    """
    Download + extract + filter papers for a single bucket.
    Writes:
      out_root/papers/<domain>/<arxiv_id>/paper.pdf
      out_root/papers/<domain>/<arxiv_id>/text.txt   (only if kept)
    Returns:
      List of metadata dicts for kept papers.
    """
    session = requests.Session()
    headers = {"User-Agent": cfg.user_agent}

    candidates = fetch_arxiv_entries(cfg, session)
    random.shuffle(candidates)

    kept: List[dict] = []
    pbar = tqdm(candidates, desc=f"Papers arXiv [{cfg.domain}]", total=len(candidates))

    for c in pbar:
        if len(kept) >= cfg.n:
            break

        paper_id = safe_filename(c["arxiv_id"])
        item_dir = os.path.join(out_root, "papers", cfg.domain, f"{cfg.min_words}-{cfg.max_words}", paper_id)
        ensure_dir(item_dir)

        pdf_path = os.path.join(item_dir, "paper.pdf")
        txt_path = os.path.join(item_dir, "text.txt")

        try:
            if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) < 10_000:
                download_file(c["pdf_url"], pdf_path, session=session, headers=headers)
                time.sleep(cfg.request_delay_s)

            text = extract_pdf_text(pdf_path)
            wc = words_count(text)

            if not (cfg.min_words <= wc <= cfg.max_words):
                # Delete PDF when rejected due to length (behavior preserved)
                safe_remove(pdf_path)
                safe_remove(txt_path)
                safe_rmdir_if_empty(item_dir)
                continue

            if not is_english(text):
                # Only delete on length failures (behavior preserved)
                continue

            write_text_utf8(txt_path, text + "\n")

            kept.append({
                "kind": "paper",
                "source": "arxiv",
                "domain": cfg.domain,
                "id": paper_id,
                "word_count": wc,
                "title": c["title"],
                "arxiv_id": c["arxiv_id"],
                "abs_url": c["abs_url"],
                "pdf_url": c["pdf_url"],
                "published_utc": c["published_utc"],
                "updated_utc": c["updated_utc"],
                "paths": {"pdf": pdf_path, "text": txt_path},
                "categories": c["categories"],
                "primary_category": c["primary_category"],
            })
            pbar.set_postfix(kept=len(kept))
        except Exception:
            # Intentionally swallow per-item errors to keep collecting.
            continue

    return kept


# ----------------------------
# Multi-bucket runner + CLI
# ----------------------------

def _parse_categories(values: List[str]) -> List[str]:
    """
    Accept either:
      --categories cs.AI cs.LG
    or:
      --categories cs.AI,cs.LG
    """
    out: List[str] = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def load_buckets_from_json(path: str) -> List[ArxivBucket]:
    """
    JSON formats supported:

    1) {"buckets": [ {...}, {...} ]}
    2) [ {...}, {...} ]

    Each bucket object maps to ArxivBucket fields.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "buckets" in data:
        data = data["buckets"]

    if not isinstance(data, list):
        raise ValueError("Config JSON must be a list of bucket objects or {'buckets': [...]}")

    buckets: List[ArxivBucket] = []
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise ValueError(f"Bucket #{i} must be an object/dict, got {type(obj).__name__}")
        if "categories" in obj and isinstance(obj["categories"], str):
            obj = dict(obj)
            obj["categories"] = _parse_categories([obj["categories"]])
        buckets.append(ArxivBucket(**obj))
    return buckets


def run_arxiv_buckets(buckets: Sequence[ArxivBucket], out_root: str, seed: int = 42) -> List[dict]:
    random.seed(seed)
    ensure_dir(out_root)

    all_items: List[dict] = []
    for b in buckets:
        items = collect_arxiv_bucket(b, out_root)
        all_items.extend(items)
        write_jsonl(os.path.join(out_root, f"metadata_papers_{safe_filename(b.domain)}-{b.min_words}-{b.max_words}.jsonl"), items)

    write_jsonl(os.path.join(out_root, "metadata_papers_all.jsonl"), all_items)

    print("\nDone.")
    print(f"Total papers written this run: {len(all_items)}")
    print(f"Output folder: {os.path.abspath(out_root)}")
    return all_items


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download arXiv papers (PDF + extracted text) for one or more buckets.")
    ap.add_argument("--out", default="stimuli", help="Output root folder")
    ap.add_argument("--seed", type=int, default=42)

    # Option A: JSON config with multiple buckets
    ap.add_argument("--config", help="Path to JSON file containing bucket specs (supports multiple buckets)")

    # Option B: single bucket via CLI flags (used if --config is not provided)
    ap.add_argument("--domain", default="default", help="Bucket name / domain label (folder name)")
    ap.add_argument("--n", type=int, default=50, help="Number of papers to keep")
    ap.add_argument("--categories", nargs="+", default=[], help="arXiv category codes (space or comma separated)")

    ap.add_argument("--days-back", type=int, default=60)
    ap.add_argument("--min-words", type=int, default=10)
    ap.add_argument("--max-words", type=int, default=4000)
    ap.add_argument("--oversample-factor", type=int, default=3)
    ap.add_argument("--request-delay-s", type=float, default=3.0)

    ap.add_argument("--comment-phrase", default=None, help='API-side comment phrase filter (co:"...")')
    ap.add_argument("--require-comment-4-pages", action="store_true", help="Local filter: require '4 pages' in comment")
    ap.add_argument("--no-submittedDate-filter", action="store_true", help="Disable submittedDate:[... TO ...] filter")

    ap.add_argument("--user-agent", default="LLMer")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        buckets = load_buckets_from_json(args.config)
    else:
        cats = _parse_categories(args.categories)
        if not cats:
            raise SystemExit("Error: provide --categories (or use --config with bucket specs).")
        buckets = [
            ArxivBucket(
                domain=args.domain,
                n=args.n,
                categories=cats,
                days_back=args.days_back,
                min_words=args.min_words,
                max_words=args.max_words,
                oversample_factor=args.oversample_factor,
                request_delay_s=args.request_delay_s,
                require_comment_4_pages=args.require_comment_4_pages,
                comment_phrase=args.comment_phrase,
                use_submittedDate_filter=not args.no_submittedDate_filter,
                user_agent=args.user_agent,
            )
        ]

    run_arxiv_buckets(buckets, args.out, seed=args.seed)


if __name__ == "__main__":
    main()