# src/run_alibi_tabular.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from alibi_detect.cd import TabularDrift

# --------------------------
# Config & paths
# --------------------------
DATA_DIR = Path("data")  # adjust if needed
OUT_DIR = Path("reports") / "alibi_tabular"
RUNS_DIR = OUT_DIR / "runs"

PAIRS = [
    ("cur_clean_emb.parquet",      "clean"),
    ("cur_brightness_emb.parquet", "brightness"),
    ("cur_blur_emb.parquet",       "blur"),
    ("cur_rotation_emb.parquet",   "rotation"),
    ("cur_noise_emb.parquet",      "noise"),
]

# --------------------------
# Setup filesystem
# --------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# Load reference
# --------------------------
ref_path = DATA_DIR / "ref_emb.parquet"
assert ref_path.exists(), f"Missing reference file: {ref_path}"

ref_df = pd.read_parquet(ref_path)

# Keep a stable feature order: embeddings first, then 'weight'
cols = [c for c in ref_df.columns if c.startswith("emb_")]
if "weight" in ref_df.columns:
    cols += ["weight"]

X_ref = ref_df[cols].to_numpy(dtype=np.float32)

# --------------------------
# Init detector (version-robust)
# --------------------------
def _init_tabular_drift(X_ref_arr: np.ndarray) -> TabularDrift:
    kw = dict(
        p_val=0.05,
        correction="bonferroni",
        alternative="two-sided",
        n_features=X_ref_arr.shape[1],
        preprocess_at_init=False,
    )
    try:
        return TabularDrift(X_ref=X_ref_arr, **kw)  # newer API
    except TypeError:
        try:
            return TabularDrift(X_ref_arr, **kw)    # older API (positional)
        except TypeError:
            kw.pop("n_features", None)
            return TabularDrift(X_ref_arr, **kw)

cd = _init_tabular_drift(X_ref)

# --------------------------
# Helpers
# --------------------------
def _to_scalar_bool(x) -> bool:
    """Turn scalar/array-like drift signal into a single boolean."""
    if hasattr(x, "__len__") and not np.isscalar(x):
        return bool(np.any(np.asarray(x)))
    return bool(x)

def _safe_array(x, length: int):
    """Return numpy array; if missing/None, return NaNs of required length."""
    if x is None:
        return np.full(length, np.nan, dtype=float)
    x_arr = np.asarray(x)
    if np.isscalar(x_arr) or x_arr.ndim == 0:
        return np.full(length, float(x_arr), dtype=float)
    return x_arr.astype(float, copy=False)

# --------------------------
# One run
# --------------------------
def run_one(cur_file: str, run_name: str):
    cur_path = DATA_DIR / cur_file
    if not cur_path.exists():
        print(f"⚠️  Skipping '{run_name}' (missing file): {cur_path}")
        return None, []

    cur_df = pd.read_parquet(cur_path)

    # Ensure same columns/order as reference
    missing_cols = set(cols) - set(cur_df.columns)
    assert not missing_cols, f"{run_name}: missing columns: {missing_cols}"

    X_cur = cur_df[cols].to_numpy(dtype=np.float32)

    preds = cd.predict(
        X_cur,
        drift_type="feature",      # per-feature tests; change to "batch" for a single test
        return_p_val=True,
        return_distance=True,
    )
    data = preds.get("data", {})

    # Dataset-level flag (robust to array/scalar)
    is_drift_field = data.get("is_drift", False)
    is_drift = _to_scalar_bool(is_drift_field)

    # Alpha (post-correction)
    alpha = float(data.get("threshold", 0.05))

    # Global p-val (may be NA for feature mode)
    p_global = data.get("p_val", np.nan)
    if hasattr(p_global, "__len__") and not np.isscalar(p_global):
        p_global = np.nan
    p_global = float(p_global) if not (isinstance(p_global, float) and np.isnan(p_global)) else np.nan

    # Per-feature p-vals / mask across versions
    p_feat = None
    drift_mask = None
    per_feat = data.get("per_feature")
    if isinstance(per_feat, dict):
        p_feat = per_feat.get("p_val")
        drift_mask = per_feat.get("is_drift")
    if p_feat is None:
        if hasattr(data.get("p_val", None), "__len__") and not np.isscalar(data["p_val"]):
            p_feat = data["p_val"]
    if drift_mask is None:
        if hasattr(is_drift_field, "__len__") and not np.isscalar(is_drift_field):
            drift_mask = is_drift_field
        elif p_feat is not None:
            drift_mask = np.asarray(p_feat) < alpha

    p_feat = _safe_array(p_feat, length=len(cols))
    drift_mask = np.asarray(drift_mask) if drift_mask is not None else (p_feat < alpha)
    drift_mask = drift_mask.astype(bool)
    if drift_mask.size != len(cols):
        drift_mask = np.resize(drift_mask, len(cols))

    drift_ratio = float(np.mean(drift_mask))
    drifted_cols = [cols[i] for i, d in enumerate(drift_mask) if d]

    # Console summary
    print(f"\n=== {run_name.upper()} ===")
    print(
        f"Dataset drift: {is_drift} | global p={p_global if not np.isnan(p_global) else 'NA'} | "
        f"alpha={alpha:.3g} | drifted {drift_mask.sum()}/{len(cols)} ({drift_ratio:.2%})"
    )
    if "weight" in cols:
        w_i = cols.index("weight")
        w_p = p_feat[w_i] if w_i < p_feat.size else np.nan
        print(f"  weight p-value: {w_p:.4g} | drift={bool(drift_mask[w_i])}")

    # Per-run JSON
    run_json = {
        "run": run_name,
        "dataset_drift": bool(is_drift),
        "global_p_value": None if np.isnan(p_global) else float(p_global),
        "alpha": float(alpha),
        "drift_ratio": float(drift_ratio),
        "drifted_features": drifted_cols,
    }
    (RUNS_DIR / f"{run_name}.json").write_text(json.dumps(run_json, indent=2))

    # Per-feature rows
    feat_rows = []
    for i, c in enumerate(cols):
        pval = float(p_feat[i]) if i < p_feat.size and not np.isnan(p_feat[i]) else None
        feat_rows.append({
            "run": run_name,
            "feature": c,
            "p_value": pval,
            "drift": bool(drift_mask[i]),
        })

    return run_json, feat_rows

# --------------------------
# Run all & save rollups
# --------------------------
summary_rows = []
all_feat_rows = []

for f, name in PAIRS:
    run_json, feat_rows = run_one(f, name)
    if run_json is not None:
        summary_rows.append(run_json)
        all_feat_rows.extend(feat_rows)

pd.DataFrame(summary_rows).to_json(OUT_DIR / "summary.json", orient="records", indent=2)
pd.DataFrame(all_feat_rows).to_csv(OUT_DIR / "per_feature.csv", index=False)

print("\n✅ Saved reports to:", OUT_DIR.resolve())
print("   - summary.json (dataset-level per run)")
print("   - per_feature.csv (per-feature p-values & drift flags)")
print("   - runs/*.json (one JSON per run)")
