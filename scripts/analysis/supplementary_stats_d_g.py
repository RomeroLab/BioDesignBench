#!/usr/bin/env python3
"""Compute supplementary statistics for BioDesignBench Sections D–G.

Outputs LaTeX-ready numbers for:
  D: Independent Structural Verification
  E: Sequence-level Analysis
  F: Agent Agreement and Clustering
  G: Failure Mode Analysis
"""

from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list, cophenet
from scipy.spatial.distance import squareform, pdist

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.load_results import load_all, load_score_matrix, CONDITION_MAP, EXCLUDED_TASKS
from scripts.analysis.si_common import CONDITION_ORDER, CONDITION_ORDER_NO_ORACLE, CONDITION_SHORT

np.random.seed(42)

# ── Natural UniProt/SwissProt AA frequencies (%) ─────────────────────
STANDARD_AAS = list("ARNDCEQGHILKMFPSTWYV")
NATURAL_AA_FREQ = {
    "A": 8.25, "R": 5.53, "N": 4.06, "D": 5.45, "C": 1.37,
    "E": 6.75, "Q": 3.93, "G": 7.08, "H": 2.27, "I": 5.96,
    "L": 9.66, "K": 5.84, "M": 2.42, "F": 3.86, "P": 4.70,
    "S": 6.56, "T": 5.34, "W": 1.08, "Y": 2.92, "V": 6.87,
}


def _parse_fasta(fasta_path: Path) -> list[str]:
    sequences, current = [], []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            elif line:
                current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def _extract_sequences() -> dict[str, dict[str, list[str]]]:
    """Extract sequences per (condition, task_id) from FASTA files."""
    result = defaultdict(lambda: defaultdict(list))
    for condition, info in CONDITION_MAP.items():
        agent_dir = info["path"]
        if not agent_dir.exists():
            continue
        for task_dir in sorted(agent_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            tid = task_dir.name
            if tid in EXCLUDED_TASKS:
                continue
            fasta = task_dir / "output" / "designed_sequences.fasta"
            if not fasta.exists():
                continue
            seqs = _parse_fasta(fasta)
            for seq in seqs:
                seq_upper = seq.upper()
                aa_frac = sum(1 for c in seq_upper if c in STANDARD_AAS) / max(len(seq_upper), 1)
                if aa_frac > 0.8 and len(seq_upper) >= 10:
                    result[condition][tid].append(seq_upper)
    return dict(result)


def _compute_aa_freq(sequences: list[str]) -> dict[str, float]:
    counts = {aa: 0 for aa in STANDARD_AAS}
    total = 0
    for seq in sequences:
        for ch in seq:
            if ch in counts:
                counts[ch] += 1
                total += 1
    if total == 0:
        return {aa: 0.0 for aa in STANDARD_AAS}
    return {aa: (counts[aa] / total) * 100.0 for aa in STANDARD_AAS}


def _load_raw_results() -> list[dict]:
    """Load all raw result.json files."""
    rows = []
    for condition, info in CONDITION_MAP.items():
        agent_dir = info["path"]
        if not agent_dir.exists():
            continue
        for task_dir in sorted(agent_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            rf = task_dir / "result.json"
            if not rf.exists():
                continue
            with open(rf) as f:
                result = json.load(f)
            tid = result.get("task_id", "")
            if tid in EXCLUDED_TASKS:
                continue
            result["_condition"] = condition
            result["_mode"] = info["mode"]
            result["_llm"] = info["llm"]
            rows.append(result)
    return rows


def _is_binding_task_from_gt(task_id: str) -> bool:
    """Determine if a task is a binding task from ground truth thresholds."""
    gt_file = PROJECT_ROOT / "data" / "tier2" / "ground_truth" / f"{task_id}.json"
    if not gt_file.exists():
        return False
    with open(gt_file) as f:
        gt = json.load(f)
    thresholds = gt.get("evaluation_thresholds", {})
    binding_keys = {"ipTM_good", "kd_nM_good", "predicted_ddG_good", "active_site_rmsd_good"}
    return bool(binding_keys & set(thresholds.keys()))


# =====================================================================
# SECTION D: Independent Structural Verification
# =====================================================================
def section_d(df: pd.DataFrame):
    print("=" * 70)
    print("SECTION D: Independent Structural Verification")
    print("=" * 70)

    # D-1: Per-condition median AF2 metrics
    print("\n--- D-1: Per-condition median AF2 metrics ---")
    af2_cols = ["pLDDT_frac", "pTM_frac", "ipTM_frac", "i_pAE_frac"]

    # Identify binding tasks
    all_tasks = df["task_id"].unique()
    binding_tasks = {t for t in all_tasks if _is_binding_task_from_gt(t)}
    print(f"  Binding tasks: {len(binding_tasks)} / {len(all_tasks)}")

    print(f"\n  {'Condition':30s} {'pLDDT':>8s} {'pTM':>8s} {'ipTM*':>8s} {'i_pAE*':>8s}  (median, *=binding only)")
    print("  " + "-" * 72)
    for cond in CONDITION_ORDER:
        sub = df[df["condition"] == cond]
        plddt = sub["pLDDT_frac"].dropna()
        ptm = sub["pTM_frac"].dropna()
        # Binding tasks only for ipTM / i_pAE
        sub_bind = sub[sub["task_id"].isin(binding_tasks)]
        iptm = sub_bind["ipTM_frac"].dropna()
        ipae = sub_bind["i_pAE_frac"].dropna()

        plddt_str = f"{plddt.median():.3f}" if len(plddt) > 0 else "  --"
        ptm_str = f"{ptm.median():.3f}" if len(ptm) > 0 else "  --"
        iptm_str = f"{iptm.median():.3f}" if len(iptm) > 0 else "  --"
        ipae_str = f"{ipae.median():.3f}" if len(ipae) > 0 else "  --"
        n_plddt = len(plddt)
        print(f"  {cond:30s} {plddt_str:>8s} {ptm_str:>8s} {iptm_str:>8s} {ipae_str:>8s}  (n_plddt={n_plddt})")

    # D-2: pLDDT vs Quality regression
    print("\n--- D-2: pLDDT fraction vs Quality score regression ---")
    sub = df[["quality", "pLDDT_frac", "condition", "task_id"]].dropna(subset=["quality", "pLDDT_frac"])
    x = sub["pLDDT_frac"].values
    y = sub["quality"].values
    n = len(sub)

    r_pearson, p_pearson = stats.pearsonr(x, y)
    rho_spearman, p_spearman = stats.spearmanr(x, y)
    slope, intercept, r_val, p_linreg, se = stats.linregress(x, y)
    r_sq = r_val ** 2

    print(f"  n = {n}")
    print(f"  Pearson r = {r_pearson:.3f}, p = {p_pearson:.2e}")
    print(f"  Spearman ρ = {rho_spearman:.3f}, p = {p_spearman:.2e}")
    print(f"  Linear regression: Quality = {slope:.2f} × pLDDT_frac + {intercept:.2f}")
    print(f"  R² = {r_sq:.3f}")

    # How 728 data points arise
    per_cond_n = sub.groupby("condition").size()
    print(f"\n  Data point breakdown (how n={n} arises):")
    for cond in CONDITION_ORDER:
        if cond in per_cond_n.index:
            print(f"    {cond:30s}: {per_cond_n[cond]:3d}")
    print(f"    {'TOTAL':30s}: {per_cond_n.sum():3d}")

    # D-3: Outlier analysis
    print("\n--- D-3: Outlier analysis (high pLDDT but low Quality) ---")
    outlier_mask = (sub["pLDDT_frac"] > 0.8) & (sub["quality"] < 10)
    outliers = sub[outlier_mask].copy()
    print(f"  n = {len(outliers)} outliers (pLDDT_frac > 0.8 AND Quality < 10)")

    if len(outliers) > 0:
        # Merge with full df for tier breakdown and ipTM
        outlier_full = df[
            (df["task_id"].isin(outliers["task_id"])) &
            (df["condition"].isin(outliers["condition"]))
        ].copy()
        # Match exactly
        outlier_keys = set(zip(outliers["task_id"], outliers["condition"]))
        outlier_full = df[
            df.apply(lambda r: (r["task_id"], r["condition"]) in outlier_keys, axis=1)
        ]

        print(f"\n  {'Task':20s} {'Condition':30s} {'pLDDT':>7s} {'Quality':>7s} {'ipTM':>7s} {'TierA':>6s} {'TierB':>6s}")
        print("  " + "-" * 90)
        for _, row in outlier_full.iterrows():
            iptm_val = row.get("ipTM_frac", None)
            iptm_str = f"{iptm_val:.3f}" if pd.notna(iptm_val) else "  --"
            tier_a = row.get("tier_a", 0)
            tier_b = row.get("tier_b", 0)
            print(f"  {row['task_id']:20s} {row['condition']:30s} "
                  f"{row['pLDDT_frac']:7.3f} {row['quality']:7.1f} "
                  f"{iptm_str:>7s} {tier_a:6.1f} {tier_b:6.1f}")

    # D-4: Condition-level AF2 quality comparison
    print("\n--- D-4: pLDDT median comparison (Mann-Whitney U) ---")
    key_conds = ["Oracle", "Human Expert", "DeepSeek V3 user"]
    for i, c1 in enumerate(key_conds):
        for c2 in key_conds[i + 1:]:
            x1 = df[df["condition"] == c1]["pLDDT_frac"].dropna().values
            x2 = df[df["condition"] == c2]["pLDDT_frac"].dropna().values
            u, p = stats.mannwhitneyu(x1, x2, alternative="two-sided")
            m1, m2 = np.median(x1), np.median(x2)
            p_str = f"p = {p:.2e}" if p < 0.001 else f"p = {p:.4f}"
            print(f"  {c1:20s} (med={m1:.3f}) vs {c2:20s} (med={m2:.3f}): U={u:.0f}, {p_str}")

    # LaTeX summary
    print("\n  LaTeX:")
    print(f"  D-2: $r = {r_pearson:.3f}$, $\\rho = {rho_spearman:.3f}$, $n = {n}$")
    print(f"  D-2: Quality $= {slope:.1f} \\times$ pLDDT$_{{frac}} {intercept:+.1f}$, $R^2 = {r_sq:.3f}$")


# =====================================================================
# SECTION E: Sequence-level Analysis
# =====================================================================
def section_e():
    print(f"\n{'=' * 70}")
    print("SECTION E: Sequence-level Analysis")
    print("=" * 70)

    cond_task_seqs = _extract_sequences()

    # Flatten per condition
    cond_seqs = {}
    for cond, task_dict in cond_task_seqs.items():
        all_seqs = []
        for seqs in task_dict.values():
            all_seqs.extend(seqs)
        cond_seqs[cond] = all_seqs

    # E-1: Sequence length statistics
    print("\n--- E-1: Sequence length statistics ---")
    print(f"  {'Condition':30s} {'n_seq':>6s} {'median':>7s} {'Q1':>6s} {'Q3':>6s} {'IQR':>6s} {'outliers':>8s}")
    print("  " + "-" * 80)

    all_lengths = []
    kw_groups = []
    for cond in CONDITION_ORDER:
        if cond not in cond_seqs or not cond_seqs[cond]:
            print(f"  {cond:30s} {'--':>6s}")
            continue
        lengths = np.array([len(s) for s in cond_seqs[cond]])
        q1, med, q3 = np.percentile(lengths, [25, 50, 75])
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        n_outliers = np.sum(lengths > upper)
        print(f"  {cond:30s} {len(lengths):6d} {med:7.0f} {q1:6.0f} {q3:6.0f} {iqr:6.0f} {n_outliers:8d}")
        all_lengths.append(lengths)
        kw_groups.append(lengths)

    if len(kw_groups) >= 2:
        H, p = stats.kruskal(*kw_groups)
        if p < 1e-10:
            exp = int(np.floor(np.log10(p))) if p > 0 else -99
            print(f"\n  Kruskal-Wallis: H = {H:.1f}, p < 10^{{{exp}}}")
        else:
            print(f"\n  Kruskal-Wallis: H = {H:.1f}, p = {p:.2e}")

    # E-2: AA composition deviation
    print(f"\n--- E-2: AA composition deviation from natural (SwissProt) ---")

    # Compute per-condition frequencies
    cond_freqs = {}
    for cond in CONDITION_ORDER:
        if cond not in cond_seqs or not cond_seqs[cond]:
            continue
        cond_freqs[cond] = _compute_aa_freq(cond_seqs[cond])

    # Print large deviations (|dev| > 1.5%)
    print(f"\n  Significant deviations (|Δ| > 1.5%):")
    print(f"  {'AA':3s} {'Natural%':>8s}  ", end="")
    short_conds = [c for c in CONDITION_ORDER if c in cond_freqs]
    for c in short_conds:
        print(f"{CONDITION_SHORT.get(c, c)[:8]:>8s}", end=" ")
    print()
    print("  " + "-" * (14 + 9 * len(short_conds)))

    for aa in STANDARD_AAS:
        nat = NATURAL_AA_FREQ[aa]
        devs = []
        has_big = False
        for c in short_conds:
            d = cond_freqs[c][aa] - nat
            devs.append(d)
            if abs(d) > 1.5:
                has_big = True
        if has_big:
            print(f"  {aa:3s} {nat:8.2f}  ", end="")
            for d in devs:
                marker = "*" if abs(d) > 1.5 else " "
                print(f"{d:+7.1f}{marker}", end=" ")
            print()

    # Verify specific values from the user
    print(f"\n  Verification of specific values:")
    key_checks = [
        ("E", "Oracle", +2.9),
        ("E", "Human Expert", +5.8),
        ("E", "DeepSeek V3 user", +5.2),
        ("E", "DeepSeek V3 benchmark", +3.1),
        ("L", "Oracle", +21.5),
        ("P", "Oracle", +10.5),
    ]
    for aa, cond, expected in key_checks:
        if cond in cond_freqs:
            actual = cond_freqs[cond][aa] - NATURAL_AA_FREQ[aa]
            match = "OK" if abs(actual - expected) < 0.3 else "MISMATCH"
            print(f"    {aa} {cond:25s}: expected {expected:+.1f}, actual {actual:+.1f}  [{match}]")

    # Chi-squared test / KL divergence per condition
    print(f"\n  KL divergence (designed vs natural), per condition:")
    natural_dist = np.array([NATURAL_AA_FREQ[aa] / 100.0 for aa in STANDARD_AAS])
    for cond in CONDITION_ORDER:
        if cond not in cond_freqs:
            continue
        designed_dist = np.array([cond_freqs[cond][aa] / 100.0 for aa in STANDARD_AAS])
        # Add epsilon to avoid log(0)
        eps = 1e-10
        kl = np.sum(designed_dist * np.log((designed_dist + eps) / (natural_dist + eps)))
        print(f"    {cond:30s}: KL = {kl:.4f}")

    # E-3: Jensen-Shannon divergence matrix
    print(f"\n--- E-3: Jensen-Shannon divergence matrix ---")
    active_conds = [c for c in CONDITION_ORDER if c in cond_freqs]
    n_c = len(active_conds)

    # Include natural as reference
    dists = {}
    for cond in active_conds:
        dists[cond] = np.array([cond_freqs[cond][aa] / 100.0 for aa in STANDARD_AAS])
    dists["Natural"] = natural_dist

    all_labels = active_conds + ["Natural"]
    n_all = len(all_labels)
    js_matrix = np.zeros((n_all, n_all))
    for i in range(n_all):
        for j in range(i + 1, n_all):
            p = dists[all_labels[i]]
            q = dists[all_labels[j]]
            m = (p + q) / 2
            eps = 1e-10
            js = 0.5 * np.sum(p * np.log((p + eps) / (m + eps))) + \
                 0.5 * np.sum(q * np.log((q + eps) / (m + eps)))
            js_matrix[i, j] = js_matrix[j, i] = js

    # Print JSD to Natural for each condition (most natural-like)
    nat_idx = all_labels.index("Natural")
    print(f"  JSD to Natural (lower = more natural-like):")
    jsd_to_nat = [(all_labels[i], js_matrix[i, nat_idx]) for i in range(n_all) if i != nat_idx]
    jsd_to_nat.sort(key=lambda x: x[1])
    for label, jsd in jsd_to_nat:
        print(f"    {label:30s}: JSD = {jsd:.5f}")


# =====================================================================
# SECTION F: Agent Agreement and Clustering
# =====================================================================
def section_f(df: pd.DataFrame):
    print(f"\n{'=' * 70}")
    print("SECTION F: Agent Agreement and Clustering")
    print("=" * 70)

    score_mat = load_score_matrix()

    # F-1: Spearman correlation matrix
    print("\n--- F-1: Spearman correlation matrix ---")
    conds = [c for c in CONDITION_ORDER_NO_ORACLE if c in score_mat.columns]
    n = len(conds)
    rho_mat = np.eye(n)
    p_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            shared = score_mat[[conds[i], conds[j]]].dropna()
            if len(shared) < 3:
                rho_mat[i, j] = rho_mat[j, i] = 0.0
                continue
            rho, p = stats.spearmanr(shared[conds[i]], shared[conds[j]])
            rho_mat[i, j] = rho
            rho_mat[j, i] = rho
            p_mat[i, j] = p_mat[j, i] = p

    # Print full matrix
    print(f"\n  Full Spearman ρ matrix ({n}×{n}):")
    print(f"  {'':12s}", end="")
    for c in conds:
        print(f"{CONDITION_SHORT.get(c, c)[:8]:>9s}", end="")
    print()
    for i, c in enumerate(conds):
        print(f"  {CONDITION_SHORT.get(c, c)[:12]:12s}", end="")
        for j in range(n):
            print(f"{rho_mat[i, j]:9.3f}", end="")
        print()

    # Verify key values
    print(f"\n  Key value verification:")
    key_checks_f = [
        ("Sonnet 4.5 user", "DeepSeek V3 user", 0.67),
        ("GPT-5 user", "DeepSeek V3 user", 0.66),
        ("Gemini 2.5 Pro user", "Gemini 2.5 Pro benchmark", 0.69),
    ]
    for c1, c2, expected in key_checks_f:
        if c1 in conds and c2 in conds:
            i, j = conds.index(c1), conds.index(c2)
            actual = rho_mat[i, j]
            match = "OK" if abs(actual - expected) < 0.02 else "MISMATCH"
            print(f"    {CONDITION_SHORT[c1]:10s} vs {CONDITION_SHORT[c2]:10s}: "
                  f"expected ρ={expected:.2f}, actual ρ={actual:.3f}  [{match}]")

    # Expert vs all others
    if "Human Expert" in conds:
        exp_idx = conds.index("Human Expert")
        exp_rhos = [rho_mat[exp_idx, j] for j in range(n) if j != exp_idx]
        print(f"\n    Expert vs all others: ρ range [{min(exp_rhos):.3f}, {max(exp_rhos):.3f}]")

    # Hardcoded vs all LLMs
    if "Hardcoded Pipeline" in conds:
        hp_idx = conds.index("Hardcoded Pipeline")
        llm_idxs = [j for j, c in enumerate(conds) if c not in ["Human Expert", "Hardcoded Pipeline"]]
        hp_rhos = [rho_mat[hp_idx, j] for j in llm_idxs]
        print(f"    Hardcoded vs all LLMs: ρ range [{min(hp_rhos):.3f}, {max(hp_rhos):.3f}]")

    # F-2: Hierarchical clustering
    print(f"\n--- F-2: Hierarchical clustering ---")
    dist = 1 - rho_mat
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    dist = np.clip(dist, 0, None)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    order = leaves_list(Z)

    # Cophenetic correlation
    coph_corr, coph_dists = cophenet(Z, condensed)
    print(f"  Linkage: Ward, Distance: 1 - ρ")
    print(f"  Cophenetic correlation: {coph_corr:.3f}")

    # Cluster ordering
    print(f"  Leaf order (from dendrogram): {[CONDITION_SHORT.get(conds[i], conds[i]) for i in order]}")

    # Silhouette analysis for k=2,3,4
    from sklearn.metrics import silhouette_score
    from scipy.cluster.hierarchy import fcluster
    print(f"\n  Silhouette scores (optimal cluster count):")
    for k in range(2, min(6, n)):
        labels = fcluster(Z, k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        # Use the distance matrix for silhouette
        dist_sq = squareform(condensed)
        sil = silhouette_score(dist_sq, labels, metric="precomputed")
        print(f"    k={k}: silhouette = {sil:.3f}")

    # F-3: Within-model user vs benchmark
    print(f"\n--- F-3: User vs Benchmark mode agreement (Spearman ρ) ---")
    mode_pairs = [
        ("DeepSeek V3", "DeepSeek V3 user", "DeepSeek V3 benchmark"),
        ("GPT-5", "GPT-5 user", "GPT-5 benchmark"),
        ("Sonnet 4.5", "Sonnet 4.5 user", "Sonnet 4.5 benchmark"),
        ("Gemini 2.5 Pro", "Gemini 2.5 Pro user", "Gemini 2.5 Pro benchmark"),
    ]
    for label, uc, bc in mode_pairs:
        if uc in conds and bc in conds:
            i, j = conds.index(uc), conds.index(bc)
            rho = rho_mat[i, j]
            p_val = p_mat[i, j]
            p_str = f"p = {p_val:.2e}" if p_val < 0.001 else f"p = {p_val:.4f}"
            print(f"  {label:20s}: ρ = {rho:.3f}  ({p_str})")

    print(f"\n  LaTeX:")
    for label, uc, bc in mode_pairs:
        if uc in conds and bc in conds:
            i, j = conds.index(uc), conds.index(bc)
            rho = rho_mat[i, j]
            print(f"    {label}: $\\rho = {rho:.2f}$")


# =====================================================================
# SECTION G: Failure Mode Analysis
# =====================================================================
def section_g():
    print(f"\n{'=' * 70}")
    print("SECTION G: Failure Mode Analysis")
    print("=" * 70)

    raw_results = _load_raw_results()

    # Build per-(condition, task) analysis
    cond_task = defaultdict(dict)
    for r in raw_results:
        cond = r["_condition"]
        tid = r["task_id"]
        cond_task[cond][tid] = r

    # Classify each (condition, task) pair
    categories = ["No output", "Approach=0", "Quality=0", "Tool failure",
                  "Low diversity", "Partial", "Success"]

    print(f"\n--- G-1: Failure mode classification ---")
    print(f"  Categories: {categories}")

    # Count per condition
    cond_counts = {}
    for cond in CONDITION_ORDER:
        if cond == "Oracle":
            continue
        if cond not in cond_task:
            continue

        counts = {cat: 0 for cat in categories}
        task_examples = {cat: [] for cat in categories}

        for tid, r in cond_task[cond].items():
            # Check components
            approach = r.get("approach_metrics", {}).get("score", 0)
            orch = r.get("orchestration_metrics", {}).get("score", 0)
            quality = r.get("quality_metrics", {}).get("score", 0)
            feasibility = r.get("feasibility_metrics", {}).get("score", 0)
            novelty = r.get("novelty_metrics", {}).get("score", 0)
            diversity = r.get("diversity_metrics", {}).get("score", 0)
            total = r.get("partial_score", 0)

            # Designs
            designs = r.get("raw_output", {}).get("designs", [])
            if not designs:
                designs = r.get("designs", [])
            n_designs = len(designs)

            # Tool calls
            tool_log = r.get("raw_output", {}).get("tool_call_log", [])
            n_tool_calls = len(tool_log)
            n_failed = sum(1 for tc in tool_log if not tc.get("success", True))

            success_flag = r.get("success", False)

            # Classification logic (priority order)
            if n_designs == 0 and not success_flag:
                cat = "No output"
            elif n_designs == 0:
                cat = "No output"
            elif approach == 0:
                cat = "Approach=0"
            elif n_failed > 0 and n_failed == n_tool_calls and total < 10:
                cat = "Tool failure"
            elif quality == 0:
                cat = "Quality=0"
            elif diversity == 0:
                cat = "Low diversity"
            elif all(s > 0 for s in [approach, orch, quality, feasibility]):
                cat = "Success"
            else:
                cat = "Partial"

            counts[cat] += 1
            if len(task_examples[cat]) < 3:
                task_examples[cat].append(tid)

        cond_counts[cond] = counts

    # Print table
    active_conds = [c for c in CONDITION_ORDER if c != "Oracle" and c in cond_counts]
    print(f"\n  {'Condition':30s}", end="")
    for cat in categories:
        print(f" {cat[:8]:>8s}", end="")
    print(f" {'Total':>6s}")
    print("  " + "-" * (30 + 9 * len(categories) + 7))

    for cond in active_conds:
        counts = cond_counts[cond]
        print(f"  {cond:30s}", end="")
        for cat in categories:
            print(f" {counts[cat]:8d}", end="")
        print(f" {sum(counts.values()):6d}")

    # Summary row (total across conditions)
    print(f"  {'TOTAL':30s}", end="")
    for cat in categories:
        total = sum(cond_counts[c][cat] for c in active_conds)
        print(f" {total:8d}", end="")
    print(f" {sum(sum(cond_counts[c].values()) for c in active_conds):6d}")

    # Most common failure mode per condition
    print(f"\n  Primary failure mode per condition:")
    for cond in active_conds:
        counts = cond_counts[cond]
        failure_counts = {k: v for k, v in counts.items() if k not in ["Success", "Partial"]}
        if any(v > 0 for v in failure_counts.values()):
            top_fail = max(failure_counts, key=failure_counts.get)
            print(f"    {cond:30s}: {top_fail} ({failure_counts[top_fail]})")
        else:
            print(f"    {cond:30s}: No failures")

    # Gemini-specific failure analysis
    print(f"\n  Gemini 2.5 Pro failure breakdown:")
    for gem_cond in ["Gemini 2.5 Pro benchmark", "Gemini 2.5 Pro user"]:
        if gem_cond in cond_counts:
            counts = cond_counts[gem_cond]
            print(f"    {gem_cond}: {dict(counts)}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    df = load_all()
    print(f"Loaded: {len(df)} rows ({df['task_id'].nunique()} tasks × {df['condition'].nunique()} conditions)\n")

    section_d(df)
    section_e()
    section_f(df)
    section_g()

    # ── Final LaTeX summary ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FINAL LaTeX SUMMARY")
    print("=" * 70)

    # D-2
    sub = df[["quality", "pLDDT_frac"]].dropna()
    x, y = sub["pLDDT_frac"].values, sub["quality"].values
    r_p, p_p = stats.pearsonr(x, y)
    rho_s, p_s = stats.spearmanr(x, y)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    print(f"D-2: Pearson $r = {r_p:.3f}$, Spearman $\\rho = {rho_s:.3f}$, $n = {len(sub)}$")
    print(f"D-2: $R^2 = {r_p**2:.3f}$, slope $= {slope:.1f}$, intercept $= {intercept:.1f}$")


if __name__ == "__main__":
    main()
