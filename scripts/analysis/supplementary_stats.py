#!/usr/bin/env python3
"""Compute supplementary statistics for BioDesignBench paper (Sections B & C).

Outputs LaTeX-ready numbers for:
  B-1: Kruskal-Wallis omnibus test
  B-2: Bootstrap 95% CI for mean scores
  B-3: Pairwise Mann-Whitney U with Bonferroni correction
  B-4: Rank-biserial effect sizes (user vs benchmark mode)
  C-1: Sample size for PCA / correlation
  C-2: PCA loadings verification
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.load_results import load_all, CONDITION_MAP

np.random.seed(42)


def bootstrap_ci(data: np.ndarray, n_boot: int = 10_000, ci: float = 0.95) -> tuple:
    """Bootstrap mean and percentile CI."""
    means = np.array([
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return np.mean(data), lo, hi


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation r_rb from Mann-Whitney U.

    r_rb = 2U / (n1*n2) - 1
    where U is the Mann-Whitney U statistic for x vs y.
    Positive means x > y (user > benchmark).
    """
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    n1, n2 = len(x), len(y)
    return 2 * u_stat / (n1 * n2) - 1


def main():
    df = load_all()
    conditions = list(CONDITION_MAP.keys())
    n_tasks = df["task_id"].nunique()
    n_conds = df["condition"].nunique()
    print(f"Data: {len(df)} rows = {n_tasks} tasks × {n_conds} conditions\n")

    # ── B-1: Kruskal-Wallis ─────────────────────────────────────────
    print("=" * 60)
    print("B-1: Kruskal-Wallis omnibus test")
    print("=" * 60)
    groups = [grp["total"].values for _, grp in df.groupby("condition")]
    H, p = stats.kruskal(*groups)
    if p == 0:
        # Use log10 approximation from chi2 survival function
        log_p = stats.chi2.logsf(H, df=n_conds - 1) / np.log(10)
        print(f"H = {H:.1f}, p < 10^{{{int(np.floor(log_p))}}}")
        print(f"  LaTeX: $H = {H:.1f},\\; p < 10^{{{int(np.floor(log_p))}}}$")
    else:
        print(f"H = {H:.1f}, p = {p:.2e}")
        if p < 1e-10:
            exp = int(np.floor(np.log10(p)))
            print(f"  LaTeX: $H = {H:.1f},\\; p < 10^{{{exp}}}$")
        else:
            print(f"  LaTeX: $H = {H:.1f},\\; p = {p:.2e}$")

    # ── B-2: Bootstrap 95% CI ───────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("B-2: Bootstrap 95% CI for mean total score (10,000 resamples)")
    print("=" * 60)
    for cond in conditions:
        scores = df[df["condition"] == cond]["total"].values
        mean_val, lo, hi = bootstrap_ci(scores)
        print(f"  {cond:30s}: mean = {mean_val:.1f} (95% CI: {lo:.1f}--{hi:.1f})")

    # Highlight key ones
    print("\n  Key LaTeX values:")
    for key_cond in ["Oracle", "Human Expert", "Gemini 2.5 Pro benchmark"]:
        scores = df[df["condition"] == key_cond]["total"].values
        mean_val, lo, hi = bootstrap_ci(scores)
        print(f"    {key_cond}: mean $= {mean_val:.1f}$ (95\\% CI: {lo:.1f}--{hi:.1f})")

    # ── B-3: Pairwise Mann-Whitney U with Bonferroni ────────────────
    print(f"\n{'=' * 60}")
    print("B-3: Pairwise Mann-Whitney U tests (Bonferroni-corrected)")
    print("=" * 60)
    n_pairs = n_conds * (n_conds - 1) // 2
    alpha_bonf = 0.05 / n_pairs
    print(f"  {n_conds} conditions → {n_pairs} pairs, Bonferroni α = {alpha_bonf:.6f}")

    # Store all pairwise results
    pairwise_results = []
    for i, c1 in enumerate(conditions):
        for j, c2 in enumerate(conditions):
            if j <= i:
                continue
            x = df[df["condition"] == c1]["total"].values
            y = df[df["condition"] == c2]["total"].values
            u_stat, p_val = stats.mannwhitneyu(x, y, alternative="two-sided")
            sig = "***" if p_val < alpha_bonf else "ns"
            pairwise_results.append((c1, c2, u_stat, p_val, sig))

    # Print key comparisons
    print("\n  Key comparisons:")
    key_pairs = [
        ("Human Expert", "Hardcoded Pipeline"),
        ("Gemini 2.5 Pro benchmark", "Gemini 2.5 Pro user"),
    ]
    # Also Gemini vs all LLM conditions
    gemini_conds = ["Gemini 2.5 Pro benchmark", "Gemini 2.5 Pro user"]
    llm_conds = [c for c in conditions if c not in ["Oracle", "Human Expert", "Hardcoded Pipeline"]
                 and c not in gemini_conds]

    for c1, c2, u, p_val, sig in pairwise_results:
        show = False
        if (c1, c2) in key_pairs or (c2, c1) in key_pairs:
            show = True
        if c1 in gemini_conds and c2 in llm_conds:
            show = True
        if c2 in gemini_conds and c1 in llm_conds:
            show = True
        if show:
            if p_val < 1e-10:
                exp = int(np.floor(np.log10(p_val))) if p_val > 0 else -99
                p_str = f"p < 10^{{{exp}}}"
            elif p_val < 0.001:
                p_str = f"p = {p_val:.2e}"
            else:
                p_str = f"p = {p_val:.4f}"
            print(f"    {c1:30s} vs {c2:30s}: U = {u:.0f}, {p_str} [{sig}]")

    # Full table for reference
    print("\n  Full pairwise table (significant only):")
    for c1, c2, u, p_val, sig in pairwise_results:
        if sig != "ns":
            if p_val < 1e-10:
                exp = int(np.floor(np.log10(p_val))) if p_val > 0 else -99
                p_str = f"p < 10^{{{exp}}}"
            else:
                p_str = f"p = {p_val:.2e}"
            print(f"    {c1:30s} vs {c2:30s}: {p_str}")

    print(f"\n  Non-significant pairs (p > {alpha_bonf:.6f}):")
    for c1, c2, u, p_val, sig in pairwise_results:
        if sig == "ns":
            print(f"    {c1:30s} vs {c2:30s}: p = {p_val:.4f}")

    # ── B-4: Rank-biserial effect sizes (user vs benchmark) ─────────
    print(f"\n{'=' * 60}")
    print("B-4: Rank-biserial effect size r_rb (user - benchmark mode)")
    print("=" * 60)
    mode_pairs = [
        ("DeepSeek V3", "DeepSeek V3 user", "DeepSeek V3 benchmark"),
        ("GPT-5", "GPT-5 user", "GPT-5 benchmark"),
        ("Sonnet 4.5", "Sonnet 4.5 user", "Sonnet 4.5 benchmark"),
        ("Gemini 2.5 Pro", "Gemini 2.5 Pro user", "Gemini 2.5 Pro benchmark"),
    ]
    for label, user_cond, bench_cond in mode_pairs:
        x_user = df[df["condition"] == user_cond]["total"].values
        x_bench = df[df["condition"] == bench_cond]["total"].values
        r_rb = rank_biserial(x_user, x_bench)
        # Also get p-value
        _, p_val = stats.mannwhitneyu(x_user, x_bench, alternative="two-sided")
        u_mean = np.mean(x_user)
        b_mean = np.mean(x_bench)
        delta = u_mean - b_mean
        if p_val < 0.001:
            p_str = f"p = {p_val:.2e}"
        else:
            p_str = f"p = {p_val:.4f}"
        print(f"  {label:20s}: r_rb = {r_rb:.3f}  (user={u_mean:.1f}, bench={b_mean:.1f}, Δ={delta:+.1f}, {p_str})")

    print("\n  LaTeX:")
    for label, user_cond, bench_cond in mode_pairs:
        x_user = df[df["condition"] == user_cond]["total"].values
        x_bench = df[df["condition"] == bench_cond]["total"].values
        r_rb = rank_biserial(x_user, x_bench)
        print(f"    {label}: $r_{{rb}} = {r_rb:.2f}$")

    # ── C-1: Sample size ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("C-1: Sample size for correlation / PCA")
    print("=" * 60)
    n_all = len(df)
    n_excl_oracle = len(df[df["condition"] != "Oracle"])
    print(f"  Including Oracle:  n = {n_all} ({n_tasks} tasks × {n_conds} conditions)")
    print(f"  Excluding Oracle:  n = {n_excl_oracle} ({n_tasks} tasks × {n_conds - 1} conditions)")

    # ── C-2: PCA on 6 component scores ──────────────────────────────
    print(f"\n{'=' * 60}")
    print("C-2: PCA on 6 rubric components")
    print("=" * 60)

    components = ["approach", "orchestration", "quality", "feasibility", "novelty", "diversity"]

    # Excluding Oracle for PCA (as they note)
    df_pca = df[df["condition"] != "Oracle"].copy()
    X = df_pca[components].values

    # Standardize
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    pca = PCA()
    pca.fit(X_std)

    print(f"  Variance explained:")
    cumulative = 0
    for i, (var, ratio) in enumerate(zip(pca.explained_variance_, pca.explained_variance_ratio_)):
        cumulative += ratio * 100
        print(f"    PC{i+1}: {ratio * 100:.1f}%  (cumulative: {cumulative:.1f}%)")

    print(f"\n  PC loadings:")
    for i in range(min(3, len(components))):
        loadings = pca.components_[i]
        load_str = ", ".join(f"{c}={v:.3f}" for c, v in zip(components, loadings))
        print(f"    PC{i+1}: {load_str}")

    # Also run with all conditions
    print(f"\n  --- PCA with ALL conditions (including Oracle) ---")
    X_all = df[components].values
    X_all_std = (X_all - X_all.mean(axis=0)) / X_all.std(axis=0)
    pca_all = PCA()
    pca_all.fit(X_all_std)

    cumulative = 0
    for i, ratio in enumerate(pca_all.explained_variance_ratio_):
        cumulative += ratio * 100
        print(f"    PC{i+1}: {ratio * 100:.1f}%  (cumulative: {cumulative:.1f}%)")

    # ── Summary table ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY: LaTeX-ready values")
    print("=" * 60)

    # B-1
    if p == 0:
        log_p = stats.chi2.logsf(H, df=n_conds - 1) / np.log(10)
        print(f"B-1: $H = {H:.1f},\\; p < 10^{{{int(np.floor(log_p))}}}$")

    # B-2
    for key_cond in ["Oracle", "Human Expert", "Gemini 2.5 Pro benchmark"]:
        scores = df[df["condition"] == key_cond]["total"].values
        mean_val, lo, hi = bootstrap_ci(scores)
        print(f"B-2: {key_cond} mean $= {mean_val:.1f}$ (95\\% CI: {lo:.1f}--{hi:.1f})")

    # B-3 key
    for c1, c2, u, p_val, sig in pairwise_results:
        if (c1 == "Human Expert" and c2 == "Hardcoded Pipeline") or \
           (c1 == "Hardcoded Pipeline" and c2 == "Human Expert"):
            if p_val >= alpha_bonf:
                print(f"B-3: Human Expert vs Hardcoded Pipeline: $p = {p_val:.3f}$ (ns, Bonferroni $\\alpha = {alpha_bonf:.4f}$)")
            else:
                print(f"B-3: Human Expert vs Hardcoded Pipeline: $p = {p_val:.2e}$ (sig)")

    # B-4
    for label, user_cond, bench_cond in mode_pairs:
        x_user = df[df["condition"] == user_cond]["total"].values
        x_bench = df[df["condition"] == bench_cond]["total"].values
        r_rb = rank_biserial(x_user, x_bench)
        print(f"B-4: {label} $r_{{rb}} = {r_rb:.2f}$")

    # C-1
    print(f"C-1: $n = {n_excl_oracle}$ ({n_tasks} tasks $\\times$ {n_conds - 1} conditions, excluding Oracle)")

    # C-2
    for i in range(3):
        ratio = pca.explained_variance_ratio_[i] * 100
        cumul = sum(pca.explained_variance_ratio_[:i+1]) * 100
        print(f"C-2: PC{i+1} = {ratio:.1f}\\% (cumulative {cumul:.1f}\\%)")


if __name__ == "__main__":
    main()
