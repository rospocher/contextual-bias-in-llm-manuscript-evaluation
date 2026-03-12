#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

# -----------------------------
# CONFIG (EDIT THIS)
# -----------------------------

OUTPUT_PREFIX = "output-emailOnly"
#OUTPUT_PREFIX = "output"
CSV_SOURCES: Dict[str, str] = {
    # "label": "/absolute/or/relative/path/to/file.csv",
    "gemma": f"EN/{OUTPUT_PREFIX}/gemma/merged.csv",
    "llama": f"EN/{OUTPUT_PREFIX}/llama/merged.csv",
    "qwen": f"EN/{OUTPUT_PREFIX}/qwen/merged.csv",
    "olmo": f"EN/{OUTPUT_PREFIX}/olmo/merged.csv",
    "gpt": f"EN/{OUTPUT_PREFIX}/gpt/merged.csv"
}

OUTPUT_CSV: str = f"EN/{OUTPUT_PREFIX}/merged.csv"

STRICT: bool = True          # True = error on mismatch; False = skip mismatches
ENCODING: Optional[str] = None  # e.g. "utf-8", "utf-8-sig", "latin1"
# -----------------------------


def _resolve_paths(sources: Dict[str, str]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for label, p in sources.items():
        rp = os.path.abspath(os.path.expanduser(p))
        items.append((label, rp))
    return items


def merge_csvs_from_dict(
    sources: Dict[str, str],
    out_path: str,
    strict: bool = True,
    encoding: Optional[str] = None,
) -> None:
    items = _resolve_paths(sources)
    if not items:
        raise RuntimeError("CSV_SOURCES is empty.")

    # Validate existence
    missing = [(label, p) for label, p in items if not os.path.isfile(p)]
    if missing:
        msg = "\n".join([f"  - {label}: {p}" for label, p in missing])
        raise FileNotFoundError(f"Some CSV paths do not exist:\n{msg}")

    # Use first CSV as canonical schema
    first_label, first_path = items[0]
    first_df = pd.read_csv(first_path, encoding=encoding)
    canonical_cols = list(first_df.columns)

    frames = [first_df]
    used = [(first_label, first_path)]
    skipped: List[Tuple[str, str]] = []

    for label, path in items[1:]:
        df = pd.read_csv(path, encoding=encoding)
        if list(df.columns) != canonical_cols:
            msg = (
                f"[WARN] Column mismatch in '{label}' ({path})\n"
                f"  expected: {canonical_cols}\n"
                f"  got:      {list(df.columns)}"
            )
            if strict:
                raise RuntimeError(msg)
            print(msg, file=sys.stderr)
            skipped.append((label, path))
            continue

        frames.append(df)
        used.append((label, path))

    merged = pd.concat(frames, ignore_index=True)

    out_path = os.path.abspath(os.path.expanduser(out_path))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"[OK] Wrote merged CSV: {out_path}")
    print(f"[OK] Included {len(used)} file(s):")
    for label, path in used:
        print(f"  - {label}: {path}")

    if skipped:
        print(f"[INFO] Skipped {len(skipped)} file(s) due to schema mismatch:")
        for label, path in skipped:
            print(f"  - {label}: {path}")


def main() -> None:
    merge_csvs_from_dict(
        sources=CSV_SOURCES,
        out_path=OUTPUT_CSV,
        strict=STRICT,
        encoding=ENCODING,
    )


if __name__ == "__main__":
    main()