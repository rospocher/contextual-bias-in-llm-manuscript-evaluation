#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import pandas as pd


def find_csvs(root: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(root.glob(pattern))


def add_clean_columns(df: pd.DataFrame, text_id_col: str = "text_id") -> pd.DataFrame:
    if text_id_col not in df.columns:
        raise KeyError(f"Missing required column '{text_id_col}'")

    # Split on "-" and take first 3 parts
    parts = df[text_id_col].astype(str).str.split("-", n=3, expand=True)

    # parts[0], parts[1], parts[2] correspond to the first three splits
    df = df.copy()
    df["clean_textID"] = parts[0]
    df["clean_domain"] = parts[1]
    df["clean_len"] = parts[2]
    return df


def merge_csvs(
    input_dir: Path,
    output_csv: Path,
    recursive: bool = True,
    require_same_columns: bool = True,
    text_id_col: str = "text_id",
) -> None:
    csv_files = find_csvs(input_dir, recursive=recursive)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {input_dir}")

    dfs: list[pd.DataFrame] = []
    reference_cols: list[str] | None = None

    for f in csv_files:
        df = pd.read_csv(f)

        if reference_cols is None:
            reference_cols = list(df.columns)
        else:
            if require_same_columns and list(df.columns) != reference_cols:
                raise ValueError(
                    "CSV structure mismatch.\n"
                    f"Reference columns (from first file): {reference_cols}\n"
                    f"Columns in {f}: {list(df.columns)}"
                )

        #df["__source_file"] = str(f)  # optional provenance column
        dfs.append(df)

    if require_same_columns:
        merged = pd.concat(dfs, ignore_index=True)
    else:
        # union of columns, missing values become NaN
        merged = pd.concat(dfs, ignore_index=True, sort=False)

    merged = add_clean_columns(merged, text_id_col=text_id_col)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)


def main():
    ap = argparse.ArgumentParser(
        description="Merge CSV files under a folder and add clean_* columns from text_id."
    )
    ap.add_argument("input_dir", help="Path containing CSVs (in subfolders).")
    ap.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: <input_dir>/merged.csv).",
    )
    ap.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not scan subfolders; only merge CSVs in the given folder.",
    )
    ap.add_argument(
        "--allow-column-union",
        action="store_true",
        help="Allow different columns across CSVs (will take union of columns).",
    )
    ap.add_argument(
        "--text-id-col",
        default="text_id",
        help="Name of the column containing text IDs (default: text_id).",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    output = Path(args.output).expanduser().resolve() if args.output else (input_dir / "merged.csv")

    merge_csvs(
        input_dir=input_dir/"CSVoutput",
        output_csv=output,
        recursive=not args.no_recursive,
        require_same_columns=not args.allow_column_union,
        text_id_col=args.text_id_col,
    )

    print(f"Done. Wrote merged CSV to: {output}")


if __name__ == "__main__":
    main()