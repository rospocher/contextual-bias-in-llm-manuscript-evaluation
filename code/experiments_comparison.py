#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Legend helpers (NEW)
# -----------------------------------------------------------------------------

def _ncols_for_handles(n: int, max_cols: int = 4) -> int:
    if n <= 0:
        return 1
    return min(max_cols, n)

def add_legend_above(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    title: str,
    y: float = 1.02,
    max_cols: int = 4,
    fontsize: int = 9,
    title_fontsize: int = 9,
    frameon: bool = False,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ncol = _ncols_for_handles(len(handles), max_cols=max_cols)
    ax.legend(
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

def reserve_top_for_legends(fig: plt.Figure, *, n_legend_rows: int = 1) -> None:
    top_margin = 0.08 * n_legend_rows + 0.02
    top_margin = min(0.30, max(0.12, top_margin))
    fig.tight_layout(rect=[0, 0, 1, 1 - top_margin])


# -----------------------------------------------------------------------------
# Factor coding
# -----------------------------------------------------------------------------

INST_HI = "HI"
INST_LI = "LI"
BIB_HI = "HP"
BIB_LI = "LP"


def normalize_effect_name(eff: str) -> str:
    eff = str(eff)
    if eff == "TOP_minus_LOW":
        return "HI_minus_LI"
    if eff == "HIGH_minus_LOW":
        return "HP_minus_LP"
    return eff


# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------

def sign_flip_pvalue(d: np.ndarray, mc: int = 200_000, seed: int = 1) -> float:
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = float(np.mean(d))
    signs = rng.choice([-1, 1], size=(mc, len(d)))
    sims = (signs * d[None, :]).mean(axis=1)
    return float((np.abs(sims) >= abs(obs)).mean())


def bootstrap_mean_ci(d: np.ndarray, B: int = 10000, seed: int = 1) -> Tuple[float, float]:
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# -----------------------------------------------------------------------------
# Cache utilities
# -----------------------------------------------------------------------------

def cache_dir(out_dir: str) -> str:
    d = os.path.join(out_dir, "analysis_cache")
    os.makedirs(d, exist_ok=True)
    return d


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def forest_plot_diff_grouped(
    df: pd.DataFrame,
    title: str,  # kept for signature compatibility; NOT USED (title is dropped)
    out_path: str,
    *,
    outcome_order: Optional[List[str]] = None,
    effect_order: Optional[List[str]] = None,
    effect_labels: Optional[Dict[str, str]] = None,
    p_col: str = "p",
    figsize: Optional[Tuple[float, float]] = None,
    draw_separators: bool = True,
) -> None:
    """
    TRANSPOSED + GROUPED:
      x-axis: outcomes
      y-axis: A−B estimate
      grouping: multiple effects per outcome (offsets)
      legend: effects (markers) ABOVE plot
    """
    effect_order = effect_order or ["HI_minus_LI", "BF_minus_WM"]
    effect_labels = effect_labels or {
        "HI_minus_LI": "HI − LI (institution tier)",
        "BF_minus_WM": "BF − WM (name cue)",
    }

    df = df.copy()
    df["effect"] = pd.Categorical(df["effect"], categories=effect_order, ordered=True)

    if outcome_order is None:
        outcomes = list(df["outcome"].dropna().unique())
    else:
        outcomes = [o for o in outcome_order if (df["outcome"] == o).any()]

    x = np.arange(len(outcomes), dtype=float)

    nE = len(effect_order)
    offsets = np.array([0.0]) if nE == 1 else np.linspace(-0.25, 0.25, nE)

    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H", "d", "p"]
    marker_map = {e: markers[i % len(markers)] for i, e in enumerate(effect_order)}

    if figsize is None:
        # +0.8 height to accommodate legend above
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
            label=effect_labels.get(eff, eff),
        )

        for xi, hi, p in zip((x + offsets[j])[ok], yhis[ok], pvals[ok]):
            st = stars(p)
            if st:
                ax.text(xi, hi + 0.02, st, ha="center", va="bottom")

    if draw_separators and len(outcomes) > 1:
        bounds = x[:-1] + 0.5
        y0, y1 = ax.get_ylim()
        ax.vlines(bounds, y0, y1, linewidth=0.8, alpha=0.25, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, rotation=0, ha="center")
    #ax.set_ylabel("Difference in effect size: A − B (paired across texts; 95% bootstrap CI)")


    # Legend above + reserve top space
    add_legend_above(fig, ax, title="Effect", y=1.02, max_cols=4, frameon=False)
    reserve_top_for_legends(fig, n_legend_rows=1)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)





# -----------------------------------------------------------------------------
# COMPUTE
# -----------------------------------------------------------------------------

def run_compute(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    cdir = cache_dir(args.out_dir)

    A = pd.read_csv(args.effects_A)
    B = pd.read_csv(args.effects_B)

    A = A.copy()
    B = B.copy()
    A["effect"] = A["effect"].map(normalize_effect_name)
    B["effect"] = B["effect"].map(normalize_effect_name)

    keep_effects = {"HI_minus_LI", "BF_minus_WM"}
    A = A[A["effect"].isin(keep_effects)].copy()
    B = B[B["effect"].isin(keep_effects)].copy()

    if args.only_outcomes.strip():
        wanted = {s.strip() for s in args.only_outcomes.split(",") if s.strip()}
        A = A[A["outcome"].isin(wanted)]
        B = B[B["outcome"].isin(wanted)]

    M = A.merge(B, on=["text_id", "outcome", "effect"], suffixes=("_A", "_B"), how="inner")
    if M.empty:
        raise SystemExit("No overlapping (text_id,outcome,effect) rows between A and B. Check inputs.")

    rows = []
    for (outcome, effect), g in M.groupby(["outcome", "effect"], dropna=False):
        d = (g["value_A"] - g["value_B"]).to_numpy(float)
        est = float(np.nanmean(d))
        lo, hi = bootstrap_mean_ci(d, B=args.boot, seed=args.seed)
        p = sign_flip_pvalue(d, mc=200_000, seed=args.seed)
        rows.append({
            "outcome": outcome,
            "effect": effect,
            "estimate": est,
            "ci_lo": lo,
            "ci_hi": hi,
            "p": p,
            "n_texts": int(np.isfinite(d).sum()),
        })

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out_dir, "A_minus_B_effect_differences.csv"), index=False)
    res.to_csv(os.path.join(cdir, "A_minus_B_effect_differences.csv"), index=False)

    pd.Series({
        "effects_A": args.effects_A,
        "effects_B": args.effects_B,
        "boot": args.boot,
        "seed": args.seed,
        "only_outcomes": args.only_outcomes,
        "keep_effects": sorted(list(keep_effects)),
    }).to_json(os.path.join(cdir, "run_meta.json"), indent=2)

    print("[DONE] Computed and cached results in:", cdir)
    print("[DONE] Wrote to", args.out_dir)


# -----------------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------------

def run_plot(args) -> None:
    cdir = cache_dir(args.out_dir)
    path_res = os.path.join(cdir, "A_minus_B_effect_differences.csv")
    if not os.path.exists(path_res):
        raise FileNotFoundError(f"Missing cached file: {path_res}. Run 'compute' first.")

    res = pd.read_csv(path_res)
    res = res.copy()
    res["effect"] = res["effect"].map(normalize_effect_name)

    composites = ["QI", "PI", "AR"]
    qcols = [f"q{i:02d}" for i in range(1, 15)]

    effect_order = ["HI_minus_LI", "BF_minus_WM"]
    effect_labels = {
        "HI_minus_LI": "HI − LI (institution tier)",
        "BF_minus_WM": "BF − WM (name cue)",
    }

    comp_df = res[res["outcome"].isin(composites)].copy()
    q_df = res[res["outcome"].isin(qcols)].copy()

    if not comp_df.empty:
        figsize = (4.5, 4)
        forest_plot_diff_grouped(
            comp_df,
            title="",  # ignored
            out_path=os.path.join(args.out_dir, "forest_A_minus_B_composites_transposed.png"),
            outcome_order=composites,
            effect_order=effect_order,
            effect_labels=effect_labels,
            p_col="p",
            draw_separators=True,
            figsize=figsize,
        )

    if not q_df.empty:
        figsize = (12, 6)
        forest_plot_diff_grouped(
            q_df,
            title="",  # ignored
            out_path=os.path.join(args.out_dir, "forest_A_minus_B_questions_transposed.png"),
            outcome_order=qcols,
            effect_order=effect_order,
            effect_labels=effect_labels,
            p_col="p",
            draw_separators=True,
            figsize=figsize,
        )

    print("[DONE] Plots rendered from cache into:", args.out_dir)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_cli():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_c = sub.add_parser("compute", help="Compute A−B effect differences and cache results")
    ap_c.add_argument("--effects_A", required=True)
    ap_c.add_argument("--effects_B", required=True)
    ap_c.add_argument("--out_dir", required=True)
    ap_c.add_argument("--boot", type=int, default=20000)
    ap_c.add_argument("--seed", type=int, default=1)
    ap_c.add_argument("--only_outcomes", default="", help="Comma-separated outcomes (e.g., QI,PI,AR). Default=all in file.")

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