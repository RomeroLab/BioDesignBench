#!/usr/bin/env python3
"""PROMPT 14: Score Prediction Model — What determines design quality?

Builds regression models to quantify how much binary coverage, execution depth,
and candidate screening metrics predict final scores. Answers:
  "Is depth predictive beyond binary coverage?"

Key analyses:
  1. Univariate Spearman correlations (ranked by |ρ|)
  2. Multivariate regression: Binary-only vs Depth-only vs Both vs Full
  3. LASSO feature importance
  4. Hierarchical variance partitioning (Binary → +Depth → +Candidate)
  5. Per-component analysis (quality, feasibility, etc.)
  6. Gemini sensitivity check

Outputs:
  results/analysis/score_prediction_v1_univariate.csv
  results/analysis/score_prediction_v1_regression.csv
  results/analysis/score_prediction_v1_importance.csv
  results/analysis/score_prediction_v1_variance_partition.csv
  results/analysis/score_prediction_v1_by_component.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=UserWarning)

from scripts.analysis.load_results import load_all

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

BINARY_FEATURES = [
    "binary_plan_mean",
    "binary_exec_mean",
    "binary_coverage",
]

DEPTH_FEATURES = [
    "total_exec_depth",
    "backbone_gen_depth",
    "seqdesign_depth",
    "structpred_depth",
    "scoring_depth",
    "eval_multiplier",
    "mmvs",
    "plan_specificity_mean",
]

CANDIDATE_FEATURES = [
    "n_candidates",
    "mean_eval_per_candidate",
    "eval_diversity_per_cand",
    "frac_multi_metric",
    "funnel_depth",
    "total_eval_calls",
]

COMPONENTS = ["approach", "orchestration", "quality", "feasibility", "novelty", "diversity"]

ALL_STAGES = ["backbone_generation", "sequence_design", "structure_prediction", "scoring_validation"]


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix() -> pd.DataFrame:
    """Build unified feature matrix from all source data."""
    # Load main scores
    main_df = load_all()

    # Load depth data (stage-level → pivot to task-level)
    depth_raw = pd.read_csv(ROOT / "results" / "analysis" / "plan_exec_depth_v3.csv")

    # Load candidate data
    cand_df = pd.read_csv(ROOT / "results" / "analysis" / "candidate_screening_v1_task_summary.csv")

    # Pivot depth to task-level
    depth_rows = []
    for (tid, cond), grp in depth_raw.groupby(["task_id", "condition"]):
        row = {
            "task_id": tid,
            "condition": cond,
            # Binary metrics
            "binary_plan_mean": grp["binary_plan"].mean(),
            "binary_exec_mean": grp["binary_exec"].mean(),
            "binary_coverage": grp["binary_exec"].mean(),  # fraction of ref steps executed
            # Depth metrics
            "total_exec_depth": grp["exec_depth"].sum(),
            "eval_multiplier": grp["eval_multiplier"].iloc[0],
            "mmvs": grp["mmvs"].iloc[0],
            "plan_specificity_mean": grp["plan_specificity"].mean(),
        }
        # Per-stage depths
        stage_map = {
            "backbone_generation": "backbone_gen_depth",
            "sequence_design": "seqdesign_depth",
            "structure_prediction": "structpred_depth",
            "scoring_validation": "scoring_depth",
        }
        for stage, col_name in stage_map.items():
            s = grp[grp["stage"] == stage]
            row[col_name] = s["exec_depth"].sum() if len(s) > 0 else 0
        depth_rows.append(row)

    depth_df = pd.DataFrame(depth_rows)

    # Merge all together
    merged = main_df[["task_id", "condition", "mode", "llm",
                       "design_approach", "molecular_subject",
                       "approach", "orchestration", "quality",
                       "feasibility", "novelty", "diversity", "total"]].copy()

    merged = merged.merge(depth_df, on=["task_id", "condition"], how="left")

    # Merge candidate metrics
    cand_cols = ["task_id", "condition", "n_candidates", "mean_eval_per_candidate",
                 "mean_eval_diversity_per_candidate", "frac_multi_metric",
                 "funnel_depth", "total_eval_calls"]
    cand_sub = cand_df[cand_cols].copy()
    cand_sub = cand_sub.rename(columns={"mean_eval_diversity_per_candidate": "eval_diversity_per_cand"})
    merged = merged.merge(cand_sub, on=["task_id", "condition"], how="left")

    # Fill NaN with 0 for numeric features
    all_numeric = BINARY_FEATURES + DEPTH_FEATURES + CANDIDATE_FEATURES
    for col in all_numeric:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(float)

    return merged


# ---------------------------------------------------------------------------
# Step 2: Univariate correlations
# ---------------------------------------------------------------------------

def compute_univariate(df: pd.DataFrame, target: str = "total") -> pd.DataFrame:
    """Compute Spearman correlation for each feature vs target."""
    all_features = BINARY_FEATURES + DEPTH_FEATURES + CANDIDATE_FEATURES
    rows = []
    for feat in all_features:
        if feat not in df.columns:
            continue
        valid = df[[feat, target]].dropna()
        if len(valid) < 10:
            continue
        rho, p = spearmanr(valid[feat], valid[target])
        rows.append({
            "feature": feat,
            "spearman_rho": round(rho, 4),
            "abs_rho": round(abs(rho), 4),
            "p_value": p,
            "group": ("binary" if feat in BINARY_FEATURES
                      else "depth" if feat in DEPTH_FEATURES
                      else "candidate"),
        })
    result = pd.DataFrame(rows).sort_values("abs_rho", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result


# ---------------------------------------------------------------------------
# Step 3-4: Regression models
# ---------------------------------------------------------------------------

def _fit_ols(X, y):
    """Fit OLS and return R², adjusted R², AIC, BIC."""
    import statsmodels.api as sm
    X_c = sm.add_constant(X)
    model = sm.OLS(y, X_c).fit()
    return {
        "r2": model.rsquared,
        "r2_adj": model.rsquared_adj,
        "aic": model.aic,
        "bic": model.bic,
        "n": len(y),
        "k": X.shape[1],
        "model": model,
    }


def _cv_r2(X, y, groups, n_splits=5):
    """Cross-validated R² using GroupKFold (grouped by task_id)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold, cross_val_score

    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    scores = cross_val_score(
        LinearRegression(), X, y,
        cv=gkf, groups=groups, scoring="r2",
    )
    return scores.mean(), scores.std()


def compute_regression_models(df: pd.DataFrame, target: str = "total",
                               label_suffix: str = "") -> pd.DataFrame:
    """Compare Binary-only, Depth-only, Both, Full regression models."""
    available = lambda feats: [f for f in feats if f in df.columns]

    b_feats = available(BINARY_FEATURES)
    d_feats = available(DEPTH_FEATURES)
    c_feats = available(CANDIDATE_FEATURES)

    models_spec = [
        ("A. Binary only" + label_suffix, b_feats),
        ("B. Depth only" + label_suffix, d_feats),
        ("C. Candidate only" + label_suffix, c_feats),
        ("D. Binary + Depth" + label_suffix, b_feats + d_feats),
        ("E. Full (B+D+C)" + label_suffix, b_feats + d_feats + c_feats),
    ]

    valid = df.dropna(subset=[target])
    y = valid[target].values
    groups = valid["task_id"].values

    rows = []
    for name, feats in models_spec:
        if not feats:
            continue
        X = valid[feats].fillna(0).values
        ols = _fit_ols(X, y)
        cv_mean, cv_std = _cv_r2(X, y, groups)
        rows.append({
            "model_name": name,
            "features_used": ", ".join(feats),
            "n_features": len(feats),
            "R2_train": round(ols["r2"], 4),
            "R2_adj": round(ols["r2_adj"], 4),
            "R2_cv_mean": round(cv_mean, 4),
            "R2_cv_std": round(cv_std, 4),
            "AIC": round(ols["aic"], 1),
            "BIC": round(ols["bic"], 1),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4: LASSO feature importance
# ---------------------------------------------------------------------------

def compute_lasso_importance(df: pd.DataFrame, target: str = "total") -> pd.DataFrame:
    """LASSO feature selection and coefficient ranking."""
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    all_feats = [f for f in BINARY_FEATURES + DEPTH_FEATURES + CANDIDATE_FEATURES
                 if f in df.columns]
    valid = df.dropna(subset=[target])
    X = valid[all_feats].fillna(0).values
    y = valid[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_scaled, y)

    result = pd.DataFrame({
        "feature": all_feats,
        "coefficient": np.round(lasso.coef_, 4),
        "abs_coefficient": np.round(np.abs(lasso.coef_), 4),
        "selected": lasso.coef_ != 0,
        "group": ["binary" if f in BINARY_FEATURES
                   else "depth" if f in DEPTH_FEATURES
                   else "candidate" for f in all_feats],
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    result.attrs["alpha"] = lasso.alpha_
    result.attrs["r2"] = lasso.score(X_scaled, y)

    return result


# ---------------------------------------------------------------------------
# Step 5: Hierarchical variance partitioning
# ---------------------------------------------------------------------------

def compute_variance_partition(df: pd.DataFrame, target: str = "total",
                                label_suffix: str = "") -> pd.DataFrame:
    """Hierarchical R² partitioning: Binary → +Depth → +Candidate."""
    import statsmodels.api as sm

    available = lambda feats: [f for f in feats if f in df.columns]
    b_feats = available(BINARY_FEATURES)
    d_feats = available(DEPTH_FEATURES)
    c_feats = available(CANDIDATE_FEATURES)

    valid = df.dropna(subset=[target])
    y = valid[target].values
    n = len(y)

    steps = [
        ("1. Binary" + label_suffix, b_feats),
        ("2. + Depth" + label_suffix, b_feats + d_feats),
        ("3. + Candidate" + label_suffix, b_feats + d_feats + c_feats),
    ]

    rows = []
    prev_r2 = 0.0
    prev_k = 0

    for name, feats in steps:
        if not feats:
            continue
        X = sm.add_constant(valid[feats].fillna(0).values)
        model = sm.OLS(y, X).fit()
        r2 = model.rsquared
        k = len(feats)
        r2_inc = r2 - prev_r2
        df_inc = k - prev_k

        # Partial F-test
        if df_inc > 0 and (1 - r2) > 0:
            f_stat = (r2_inc / df_inc) / ((1 - r2) / (n - k - 1))
            p_val = 1 - sp_stats.f.cdf(f_stat, df_inc, n - k - 1)
        else:
            f_stat, p_val = np.nan, np.nan

        rows.append({
            "step": name,
            "n_features": k,
            "R2_cumulative": round(r2, 4),
            "R2_incremental": round(r2_inc, 4),
            "F_stat": round(f_stat, 2) if not np.isnan(f_stat) else np.nan,
            "p_value": p_val,
        })

        prev_r2 = r2
        prev_k = k

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 6: Component-level analysis
# ---------------------------------------------------------------------------

def compute_component_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Run regression for each scoring component."""
    available = lambda feats: [f for f in feats if f in df.columns]
    b_feats = available(BINARY_FEATURES)
    d_feats = available(DEPTH_FEATURES)
    c_feats = available(CANDIDATE_FEATURES)
    all_feats = b_feats + d_feats + c_feats

    valid = df.dropna(subset=COMPONENTS)
    groups = valid["task_id"].values

    rows = []
    for comp in COMPONENTS:
        y = valid[comp].values
        # Binary-only
        X_b = valid[b_feats].fillna(0).values
        cv_b, _ = _cv_r2(X_b, y, groups)
        # Depth-only
        X_d = valid[d_feats].fillna(0).values
        cv_d, _ = _cv_r2(X_d, y, groups)
        # Full
        X_all = valid[all_feats].fillna(0).values
        cv_all, _ = _cv_r2(X_all, y, groups)

        # Top-3 features by Spearman
        top3 = []
        for feat in all_feats:
            rho, _ = spearmanr(valid[feat].fillna(0), y)
            top3.append((feat, abs(rho)))
        top3.sort(key=lambda x: x[1], reverse=True)
        top3_str = ", ".join(f"{f}({r:.2f})" for f, r in top3[:3])

        rows.append({
            "component": comp,
            "R2_binary_cv": round(cv_b, 4),
            "R2_depth_cv": round(cv_d, 4),
            "R2_full_cv": round(cv_all, 4),
            "depth_lift": round(cv_d - cv_b, 4),
            "top_3_features": top3_str,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_tables(
    univar_df: pd.DataFrame,
    reg_df: pd.DataFrame,
    lasso_df: pd.DataFrame,
    vp_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    reg_no_gemini: pd.DataFrame,
    vp_no_gemini: pd.DataFrame,
):
    # ── Table 1: Univariate Correlations ──
    print("\n" + "=" * 85)
    print("Table 1: Univariate Correlations with Total Score (ranked by |ρ|)")
    print("=" * 85)
    print(f"{'Rank':>4s} | {'Feature':<30s} | {'Group':<10s} | {'ρ':>7s} | {'p-value':>10s}")
    print("-" * 85)
    for _, r in univar_df.iterrows():
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        print(f"{r['rank']:>4d} | {r['feature']:<30s} | {r['group']:<10s} | "
              f"{r['spearman_rho']:>+7.3f} | {r['p_value']:>10.2e} {sig}")

    # Summarize: which group dominates top-5?
    top5 = univar_df.head(5)
    group_counts = top5["group"].value_counts()
    print(f"\n  Top-5 dominated by: {dict(group_counts)}")
    # Mean |ρ| by group
    for grp in ["binary", "depth", "candidate"]:
        sub = univar_df[univar_df["group"] == grp]
        if len(sub) > 0:
            print(f"  Mean |ρ| for {grp}: {sub['abs_rho'].mean():.3f}")

    # ── Table 2: Regression Model Comparison ──
    print("\n" + "=" * 100)
    print("Table 2: Regression Model Comparison (all conditions)")
    print("=" * 100)
    print(f"{'Model':<30s} | {'k':>3s} | {'R² train':>8s} | {'R² CV':>12s} | {'AIC':>8s}")
    print("-" * 100)
    for _, r in reg_df.iterrows():
        cv_str = f"{r['R2_cv_mean']:.3f}±{r['R2_cv_std']:.3f}"
        print(f"{r['model_name']:<30s} | {r['n_features']:>3d} | {r['R2_train']:>8.4f} | "
              f"{cv_str:>12s} | {r['AIC']:>8.1f}")

    # Gemini sensitivity
    if len(reg_no_gemini) > 0:
        print("\n  --- Without Gemini ---")
        for _, r in reg_no_gemini.iterrows():
            cv_str = f"{r['R2_cv_mean']:.3f}±{r['R2_cv_std']:.3f}"
            print(f"  {r['model_name']:<30s} | {r['n_features']:>3d} | {r['R2_train']:>8.4f} | "
                  f"{cv_str:>12s} | {r['AIC']:>8.1f}")

    # ── Table 3: Variance Partitioning ──
    print("\n" + "=" * 90)
    print("Table 3: Hierarchical Variance Partitioning")
    print("=" * 90)
    print(f"{'Step':<30s} | {'k':>3s} | {'R² cum':>7s} | {'R² inc':>7s} | {'F':>8s} | {'p':>10s}")
    print("-" * 90)
    for _, r in vp_df.iterrows():
        f_str = f"{r['F_stat']:.2f}" if not np.isnan(r.get("F_stat", np.nan)) else "—"
        p_str = f"{r['p_value']:.2e}" if not np.isnan(r.get("p_value", np.nan)) else "—"
        print(f"{r['step']:<30s} | {r['n_features']:>3d} | {r['R2_cumulative']:>7.4f} | "
              f"{r['R2_incremental']:>7.4f} | {f_str:>8s} | {p_str:>10s}")

    if len(vp_no_gemini) > 0:
        print("\n  --- Without Gemini ---")
        for _, r in vp_no_gemini.iterrows():
            f_str = f"{r['F_stat']:.2f}" if not np.isnan(r.get("F_stat", np.nan)) else "—"
            p_str = f"{r['p_value']:.2e}" if not np.isnan(r.get("p_value", np.nan)) else "—"
            print(f"  {r['step']:<30s} | {r['n_features']:>3d} | {r['R2_cumulative']:>7.4f} | "
                  f"{r['R2_incremental']:>7.4f} | {f_str:>8s} | {p_str:>10s}")

    # ── Table 4: LASSO Top Features ──
    print("\n" + "=" * 85)
    print("Table 4: LASSO Feature Importance (Top-15)")
    print(f"  (α={lasso_df.attrs.get('alpha', '?'):.4f}, R²={lasso_df.attrs.get('r2', '?'):.4f})")
    print("=" * 85)
    print(f"{'Rank':>4s} | {'Feature':<30s} | {'Group':<10s} | {'Coef':>8s} | {'Selected':>8s}")
    print("-" * 85)
    for _, r in lasso_df.head(15).iterrows():
        sel = "✓" if r["selected"] else "—"
        print(f"{r['rank']:>4d} | {r['feature']:<30s} | {r['group']:<10s} | "
              f"{r['coefficient']:>+8.3f} | {sel:>8s}")

    # Count selected by group
    selected = lasso_df[lasso_df["selected"]]
    if len(selected) > 0:
        print(f"\n  Selected features by group: {dict(selected['group'].value_counts())}")

    # ── Table 5: Component Analysis ──
    print("\n" + "=" * 100)
    print("Table 5: Component-Level R² (CV) — Binary vs Depth vs Full")
    print("=" * 100)
    print(f"{'Component':<15s} | {'R² Binary':>10s} | {'R² Depth':>10s} | {'R² Full':>10s} | "
          f"{'Depth Lift':>10s} | Top-3 Features")
    print("-" * 100)
    for _, r in comp_df.iterrows():
        print(f"{r['component']:<15s} | {r['R2_binary_cv']:>10.4f} | {r['R2_depth_cv']:>10.4f} | "
              f"{r['R2_full_cv']:>10.4f} | {r['depth_lift']:>+10.4f} | {r['top_3_features']}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[1/7] Building feature matrix...")
    df = build_feature_matrix()
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    print("[2/7] Univariate correlations...")
    univar = compute_univariate(df)
    print(f"  {len(univar)} features ranked")

    print("[3/7] Regression models (all conditions)...")
    reg_all = compute_regression_models(df, target="total")

    print("[4/7] LASSO feature importance...")
    lasso = compute_lasso_importance(df)

    print("[5/7] Hierarchical variance partitioning...")
    vp_all = compute_variance_partition(df)

    print("[6/7] Component-level analysis...")
    comp = compute_component_analysis(df)

    print("[7/7] Gemini sensitivity check...")
    df_no_gem = df[~df["llm"].str.contains("Gemini", na=False)]
    reg_no_gem = compute_regression_models(df_no_gem, target="total", label_suffix=" (no Gem)")
    vp_no_gem = compute_variance_partition(df_no_gem, target="total", label_suffix=" (no Gem)")

    # Save CSVs
    out_dir = ROOT / "results" / "analysis"
    univar.to_csv(out_dir / "score_prediction_v1_univariate.csv", index=False)
    reg_all.to_csv(out_dir / "score_prediction_v1_regression.csv", index=False)
    lasso.to_csv(out_dir / "score_prediction_v1_importance.csv", index=False)
    vp_all.to_csv(out_dir / "score_prediction_v1_variance_partition.csv", index=False)
    comp.to_csv(out_dir / "score_prediction_v1_by_component.csv", index=False)

    print(f"\n  Saved 5 CSVs to {out_dir}/")

    print_tables(univar, reg_all, lasso, vp_all, comp, reg_no_gem, vp_no_gem)


if __name__ == "__main__":
    main()
