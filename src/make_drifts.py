# src/make_drifts.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.filters import gaussian
from skimage.transform import rotate
from skimage.util import random_noise

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Load MNIST CSV (expects data/train.csv with column 'label' + 784 pixel columns)
df = pd.read_csv(DATA_DIR / "train.csv")
y = df["label"].astype(int)
X = df.drop(columns=["label"]).astype(np.float32)

# --- Synthetic "weight" feature tied to label (for demo)
rng = np.random.default_rng(42)
digit_base_weight = {d: 50 + 5*d for d in range(10)}  # arbitrary mapping
weight = np.array([digit_base_weight[int(lbl)] + rng.normal(0, 1.0) for lbl in y])
X["weight"] = weight

# --- Build splits
ref_idx = slice(0, 15000)
cur_idx = slice(15000, 20000)

ref_raw = X.iloc[ref_idx].copy()
cur_clean_raw = X.iloc[cur_idx].copy()

# --- Define image transforms (operate on the 784 pixel columns; keep 'weight' intact)
pix_cols = [c for c in X.columns if c != "weight"]

def drift_brightness(block, scale=1.4):
    px = block[pix_cols].to_numpy()
    px = np.clip(px * scale, 0, 255)
    out = block.copy()
    out[pix_cols] = px
    return out

def drift_blur(block, sigma=1.0):
    px = block[pix_cols].to_numpy().reshape(-1, 28, 28)
    px_blur = np.stack([gaussian(img, sigma=sigma, preserve_range=True) for img in px], axis=0)
    out = block.copy()
    out[pix_cols] = px_blur.reshape(len(block), -1)
    return out

def drift_rotation(block, angle=15):
    px = block[pix_cols].to_numpy().reshape(-1, 28, 28)
    px_rot = np.stack([rotate(img, angle=angle, mode="edge", preserve_range=True) for img in px], axis=0)
    out = block.copy()
    out[pix_cols] = px_rot.reshape(len(block), -1)
    return out

def drift_noise(block, var=0.02):
    px = block[pix_cols].to_numpy().reshape(-1, 28, 28)
    px_noisy = np.stack([random_noise(img, mode="gaussian", var=var)*255 for img in px], axis=0)
    out = block.copy()
    out[pix_cols] = px_noisy.reshape(len(block), -1)
    return out

cur_brightness_raw = drift_brightness(cur_clean_raw, scale=1.4)
cur_blur_raw       = drift_blur(cur_clean_raw, sigma=1.0)
cur_rotation_raw   = drift_rotation(cur_clean_raw, angle=15)
cur_noise_raw      = drift_noise(cur_clean_raw, var=0.02)

# --- Optional: simulate weight sensor bias (+10% on all current)
for dfc in [cur_clean_raw, cur_brightness_raw, cur_blur_raw, cur_rotation_raw, cur_noise_raw]:
    dfc["weight"] = dfc["weight"] * 1.10

# --- Fit PCA on reference pixels; transform all to embeddings; append weight
def fit_embeddings(ref_block, n_comp=50):
    scaler = StandardScaler()
    ref_pxs = scaler.fit_transform(ref_block[pix_cols].to_numpy())
    pca = PCA(n_components=n_comp, random_state=0)
    ref_emb = pca.fit_transform(ref_pxs)
    ref_df = pd.DataFrame(ref_emb, columns=[f"emb_{i}" for i in range(n_comp)])
    ref_df["weight"] = ref_block["weight"].to_numpy()
    return ref_df, scaler, pca

def transform_embeddings(block, scaler, pca):
    pxs = scaler.transform(block[pix_cols].to_numpy())
    emb = pca.transform(pxs)
    out = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(pca.n_components_)])
    out["weight"] = block["weight"].to_numpy()
    return out

ref_emb, scaler, pca = fit_embeddings(ref_raw)
cur_clean_emb      = transform_embeddings(cur_clean_raw, scaler, pca)
cur_brightness_emb = transform_embeddings(cur_brightness_raw, scaler, pca)
cur_blur_emb       = transform_embeddings(cur_blur_raw, scaler, pca)
cur_rotation_emb   = transform_embeddings(cur_rotation_raw, scaler, pca)
cur_noise_emb      = transform_embeddings(cur_noise_raw, scaler, pca)

# --- Save parquet files
ref_emb.to_parquet(DATA_DIR / "ref_emb.parquet")
cur_clean_emb.to_parquet(DATA_DIR / "cur_clean_emb.parquet")
cur_brightness_emb.to_parquet(DATA_DIR / "cur_brightness_emb.parquet")
cur_blur_emb.to_parquet(DATA_DIR / "cur_blur_emb.parquet")
cur_rotation_emb.to_parquet(DATA_DIR / "cur_rotation_emb.parquet")
cur_noise_emb.to_parquet(DATA_DIR / "cur_noise_emb.parquet")

print("✅ Embeddings + weight saved under:", DATA_DIR.resolve())
