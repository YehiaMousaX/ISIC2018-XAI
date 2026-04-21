# %% [markdown]
# # Multi-Dimensional XAI Evaluation Pipeline — ISIC 2018 Skin Lesion Classification
# 
# **Thesis:** *A Multi-Dimensional Evaluation of Explainability Methods Across CNN and Vision Transformer Architectures for Skin Lesion Classification*
# 
# ---
# 
# ## Notebook Overview
# 
# This notebook implements the full evaluation pipeline described in Chapter 5 of the thesis. It runs both **locally** and on **Kaggle** (GPU-accelerated).
# 
# ### Architectures
# | Architecture | Family |
# |---|---|
# | EfficientNet-B0 | CNN |
# | DenseNet-121 | CNN |
# | ViT-Base/16 | Transformer |
# | Swin-Tiny | Transformer |
# 
# ### XAI Methods
# Grad-CAM · HiResCAM · Attention Rollout · LIME · KernelSHAP · Integrated Gradients
# 
# ### Evaluation Dimensions
# | Dimension | Metrics |
# |---|---|
# | Faithfulness | AOPC, Insertion/Deletion AUC |
# | Plausibility | IoU, Dice, Pointing Game |
# | Robustness | Max-Sensitivity |
# | Complexity | Saliency Entropy |
# 
# ### Phases
# **A** Data · **B** Models · **C** XAI · **D** Faithfulness · **E** Plausibility · **F** Robustness/Complexity · **G** Analysis · **H** Ablations · **I** Outputs

# %% [markdown]
# ---
# # Phase A — Data Loading, EDA & Preprocessing
# > **Thesis §5.1 — Dataset**
# 
# This phase establishes the full data foundation. The dataset is **pre-organised** — splits, labels, and masks are provided; no custom splitting is required.
# 
# ### Dataset Structure
# ```
# Data/
# ├── csv/
# │   ├── train.csv              # 8,750 rows | cols: image, MEL, NV, BCC, AKIEC, BKL, DF, VASC
# │   ├── val.csv                # 1,458 rows
# │   ├── test.csv               # 1,512 rows
# │   └── lesion_groupings.csv   # image, lesion_id, diagnosis_confirm_type
# ├── images/
# │   ├── train/  (8,750 jpgs)
# │   ├── val/    (1,458 jpgs)
# │   └── test/   (1,512 jpgs)
# └── plausibility/
#     ├── masks/       # 3,694 *_segmentation.png  ← binary lesion masks
#     ├── images/      # corresponding RGB images
#     └── attributes/  # 5 attribute maps per image
#                      # (globules, milia_like_cyst, negative_network,
#                      #  pigment_network, streaks)
# ```
# 
# ### Steps
# A.1 Config · A.2 Install · A.3 Labels · A.4 Plausibility index · A.5 Weights & stats · A.6 Dataset & loaders · A.7 Visual check

# %% [markdown]
# ## A.1 — Configuration
# Single cell controlling all hyperparameters and paths. Toggle `DEBUG = True` for fast local iteration.

# %%
# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit this cell only
# ────────────────────────────────────────────────────────────────────────────────
import os, json, warnings, random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from PIL import Image
from tqdm.auto import tqdm



# ── Execution mode ──────────────────────────────────────────────────────────────────
DEBUG  = False   # True → small subset, 3 epochs; False → full Kaggle run
KAGGLE = "KAGGLE_URL_BASE" in os.environ
SEED   = 42

# ── Kaggle dataset slug ──────────────────────────────────────────────────────────────
KAGGLE_DATASET_SLUG = "isic2018-dataset"
KAGGLE_USER         = "yehiasamir"

# ── Paths ──────────────────────────────────────────────────────────────────────────
if KAGGLE:
    DATA_ROOT = "/kaggle/input/datasets/yehiasamir/isic2018-dataset/Data"
    PREP_ROOT = "/kaggle/working/prepared"
    OUT_ROOT  = "/kaggle/working"

    from huggingface_hub import login
    with open("/kaggle/input/datasets/yehiasamir/hf-credentials/hf_token.txt") as f:
        secret_value = f.read().strip()
    login(secret_value)

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "info"
else:
    DATA_ROOT = "./Data"
    PREP_ROOT = "./prepared"
    OUT_ROOT  = "./outputs"

# On Kaggle, use the separately uploaded re-split CSVs instead of the ones
# bundled with the image dataset (which still reflect the old local split).
CSV_DIR = (
    "/kaggle/input/new-train-val-csv"
    if KAGGLE else
    os.path.join(DATA_ROOT, "csv")
)
TRAIN_IMG = os.path.join(DATA_ROOT, "images", "train")
VAL_IMG   = os.path.join(DATA_ROOT, "images", "val")
TEST_IMG  = os.path.join(DATA_ROOT, "images", "test")

def find_image(img_id: str, primary_dir: str, ext: str = ".jpg") -> str:
    """Return the path to an image.

    On Kaggle the physical folder layout may not match the CSV split after a
    local re-split, so we fall back to sibling train/ and val/ dirs.
    Locally images are always in the correct folder, so no fallback is needed.
    """
    candidate = os.path.join(primary_dir, f"{img_id}{ext}")
    if not KAGGLE:
        return candidate
    if os.path.exists(candidate):
        return candidate
    img_root = os.path.dirname(primary_dir)
    for split in ("train", "val"):
        p = os.path.join(img_root, split, f"{img_id}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Image not found for id={img_id}")
MASK_DIR  = os.path.join(DATA_ROOT, "plausibility", "masks")
ATTR_DIR  = os.path.join(DATA_ROOT, "plausibility", "attributes")

# ── Model / training ───────────────────────────────────────────────────────────────
IMG_SIZE     = 256
BATCH_SIZE   = 32
MAX_EPOCHS   = 300  if not DEBUG else 3
PATIENCE     = 10   if not DEBUG else 2
LR           = 1e-4
WEIGHT_DECAY = 5e-4
NUM_CLASSES  = 7

# Regularisation / ensemble knobs
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA     = 0.2     # 0 disables
MIXUP_PROB      = 0.5     # apply mixup to 50% of batches
CUTMIX_ALPHA    = 1.0     # 0 disables; applied alongside mixup (random choice)
CUTMIX_PROB     = 0.5     # conditional: when aug triggers, this fraction is CutMix
USE_FOCAL_LOSS  = True    # ClassBalancedFocalLoss (γ=2, inv-freq) — minority boost
EARLY_STOP_MIN_DELTA = 0.002   # on macro-F1
_under_papermill = "PAPERMILL_OUTPUT_PATH" in os.environ or "PM_IN_EXECUTION" in os.environ
NUM_WORKERS  = 4 if KAGGLE else 0

# ── XAI ────────────────────────────────────────────────────────────────────────────
LIME_SAMPLES        = 1000 if not DEBUG else 100
SHAP_SAMPLES        = 1000 if not DEBUG else 100
SENSITIVITY_N       = 50   if not DEBUG else 5
SENSITIVITY_STD     = 0.01
BINARIZE_THRESHOLDS = [0.3, 0.5, 0.7]
AOPC_STEPS          = 9

# ── Architectures ────────────────────────────────────────────────────────────────────
ARCHITECTURES = {
    "efficientnet_b2": {"family": "cnn", "timm_name": "efficientnet_b2"},
    "densenet121":     {"family": "cnn", "timm_name": "densenet121"},
    "convnext_tiny":   {"family": "cnn", "timm_name": "convnext_tiny"},
    # "vit_base_16":     {"family": "vit", "timm_name": "vit_base_patch16_224"},
    # "swin_tiny":       {"family": "vit", "timm_name": "swin_tiny_patch4_window7_224"},
}

ATTR_TYPES = ["globules", "milia_like_cyst", "negative_network",
              "pigment_network", "streaks"]

# ── Reproducibility ──────────────────────────────────────────────────────────────────
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything()
os.makedirs(OUT_ROOT,  exist_ok=True)
os.makedirs(PREP_ROOT, exist_ok=True)

def apply_color_constancy(img: np.ndarray, power: int = 6) -> np.ndarray:
    """Shades-of-Gray color constancy on a uint8 HxWx3 image."""
    img_float = img.astype(np.float32) + 1e-6
    norm = (np.mean(img_float ** power, axis=(0, 1)) ** (1.0 / power))
    scale = norm.mean() / norm
    return np.clip(img_float * scale, 0, 255).astype(np.uint8)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device      : {DEVICE}")
print(f"DEBUG       : {DEBUG}")
print(f"KAGGLE      : {KAGGLE}")
print(f"DATA_ROOT   : {DATA_ROOT}")
print(f"Data exists : {os.path.isdir(DATA_ROOT)}")

# %% [markdown]
# ## A.2 — Install Dependencies
# Install packages absent from Kaggle's default image.

# %%
import subprocess, sys

def pip_install(*pkgs):
    """Install packages — only runs on Kaggle. Locally manage your own env."""
    if not KAGGLE:
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

# captum is intentionally excluded: v0.8.0 pins numpy<2.0 which breaks scipy/albumentations.
# Integrated Gradients is implemented manually in Phase C (pure PyTorch, no dependency).
pip_install(
    "timm", "albumentations", "grad-cam",
    "shap", "lime", "seaborn", "statsmodels",
)


# %%
import timm
import albumentations
from pytorch_grad_cam import GradCAM, HiResCAM

print(f"timm           : {timm.__version__}")
print(f"albumentations : {albumentations.__version__}")
print(f"numpy          : {__import__('numpy').__version__}")

# %% [markdown]
# ## A.3 — Load Labels & Lesion Groupings
# Load pre-split CSVs. Convert one-hot to `label_idx`. Attach `lesion_id` for patient-level leakage reporting.

# %%
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

def load_split(csv_path):
    df = pd.read_csv(csv_path).rename(columns={"image": "image_id"})
    df["label_idx"]  = df[CLASS_NAMES].values.argmax(axis=1)
    df["label_name"] = df["label_idx"].map(lambda i: CLASS_NAMES[i])
    return df[["image_id", "label_idx", "label_name"]]

train_df = load_split(os.path.join(CSV_DIR, "train.csv"))
val_df   = load_split(os.path.join(CSV_DIR, "val.csv"))
test_df  = load_split(os.path.join(CSV_DIR, "test.csv"))

# Attach lesion_id and diagnosis_confirm_type from groupings
# lesion_id             -> patient-level leakage tracking
# diagnosis_confirm_type -> label-quality sensitivity (Phase H.3)
groupings = (
    pd.read_csv(os.path.join(CSV_DIR, "lesion_groupings.csv"))
      .rename(columns={"image": "image_id"})
)
train_df = train_df.merge(
    groupings[["image_id", "lesion_id", "diagnosis_confirm_type"]],
    on="image_id", how="left"
)
# test_df gets confirm type for per-image sensitivity in Phase H.3
test_df = test_df.merge(
    groupings[["image_id", "diagnosis_confirm_type"]],
    on="image_id", how="left"
)

# DEBUG: stratified subsample of train
DEBUG_TRAIN_SIZE = 500
if DEBUG:
    n = DEBUG_TRAIN_SIZE
    train_df = pd.concat([
        g.sample(max(1, round(n * len(g) / len(train_df))), random_state=SEED)
        for _, g in train_df.groupby("label_idx", group_keys=False)
    ]).reset_index(drop=True)
    print(f"DEBUG train size: {len(train_df)}  (target {n}, full = 10,015)")

print(f"Train : {len(train_df):>5} | Val : {len(val_df):>3} | Test : {len(test_df):>4}")
print(f"\nTrain label distribution:")
print(train_df["label_name"].value_counts())
print(f"\nDiagnosis confirmation type (train):")
print(train_df["diagnosis_confirm_type"].value_counts())
assert train_df.isna().drop(
    columns=["lesion_id", "diagnosis_confirm_type"], errors="ignore"
).sum().sum() == 0

# %% [markdown]
# ## A.3b — Exploratory Data Analysis
# Brief visualisation of class distribution across splits and one representative
# image per class. Confirms the severe class imbalance (NV ≈67% of train)
# that drives `WeightedRandomSampler` and class-weighted CrossEntropy in later phases.

# %%
palette = sns.color_palette("Set2", NUM_CLASSES)

# ── 1. Class distribution across splits ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (df, title) in zip(axes, [
    (train_df, f"Train  (n={len(train_df):,})"),
    (val_df,   f"Val    (n={len(val_df):,})"),
    (test_df,  f"Test   (n={len(test_df):,})"),
]):
    cnts = df["label_name"].value_counts().reindex(CLASS_NAMES).fillna(0)
    bars = ax.bar(CLASS_NAMES, cnts.values, color=palette, edgecolor="white")
    for bar, v in zip(bars, cnts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                f"{int(v)}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Class"); ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

fig.suptitle("Class Distribution Across Splits", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_ROOT, "eda_class_distribution.png"), dpi=100, bbox_inches="tight")
plt.show()

# ── 2. One representative image per class ─────────────────────────────────────
fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(NUM_CLASSES * 2.5, 3))
for ax, (i, cls_name) in zip(axes, enumerate(CLASS_NAMES)):
    row = train_df[train_df["label_name"] == cls_name].iloc[0]
    img = np.array(
        Image.open(find_image(row['image_id'], TRAIN_IMG))
             .resize((160, 160))
    )
    ax.imshow(img)
    ax.set_title(cls_name, fontsize=10, fontweight="bold")
    ax.axis("off")
fig.suptitle("One Representative Image per Class (Train)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_ROOT, "eda_class_samples.png"), dpi=100, bbox_inches="tight")
plt.show()

# ── 3. Imbalance summary ─────────────────────────────────────────────────────
train_cnts = train_df["label_name"].value_counts()
ratio = train_cnts.max() / train_cnts.min()
print(f"Imbalance ratio : {ratio:.1f}x  "
      f"({train_cnts.idxmax()} = {train_cnts.max()} vs "
      f"{train_cnts.idxmin()} = {train_cnts.min()})")
print(f"\nClass weights (inverse-freq) will be computed in A.5")

# %% [markdown]
# ## A.4 — Plausibility Index
#
# ### Important: two disjoint image populations
#
# ISIC 2018 contains two separate image sets with **non-overlapping IDs**:
#
# | Population | IDs | Location | Labels? | Masks? |
# |---|---|---|---|---|
# | Task 3 classification | ISIC_0024306 – ISIC_0035528 | `images/train\|val\|test/` | yes (CSV) | no |
# | Task 1 segmentation | ISIC_0000000 – ISIC_0003693 | `plausibility/images/` | not in CSVs | yes |
#
# **Consequence for the pipeline:**
# - `has_mask` on `train_df` / `val_df` / `test_df` will always be **False** — by design, not a bug.
# - Phase E (Plausibility) does **not** draw from `eval_subsets`. It builds a separate
#   `plaus_df` from `Data/plausibility/images/` paired with their masks, runs each trained
#   model on that subset, generates heatmaps, and computes IoU/Dice/Pointing Game.
# - `mask_index` and `attr_index` are still built here so Phase E can reference them directly.
# - The `lesion_area_frac` / `lesion_size_group` columns are computed on `plaus_df` in Phase E.

# %%
PLAUS_IMG_DIR = os.path.join(DATA_ROOT, "plausibility", "images")

# mask_index  : image_id → absolute path to *_segmentation.png
mask_index = {
    f.stem.replace("_segmentation", ""): str(f)
    for f in Path(MASK_DIR).glob("*_segmentation.png")
} if Path(MASK_DIR).exists() else {}

# attr_index  : image_id → {attr_type: path}
attr_index = {}
if Path(ATTR_DIR).exists():
    for f in Path(ATTR_DIR).glob("*.png"):
        for attr in ATTR_TYPES:
            if f.stem.endswith(f"_attribute_{attr}"):
                img_id = f.stem.replace(f"_attribute_{attr}", "")
                attr_index.setdefault(img_id, {})[attr] = str(f)

# Build plaus_df: one row per image in plausibility/images/ that has a mask.
# Labels are looked up from train/val/test CSVs where available; otherwise "unknown".
_all_labels = pd.concat([train_df[["image_id","label_idx","label_name"]],
                          val_df  [["image_id","label_idx","label_name"]],
                          test_df [["image_id","label_idx","label_name"]]],
                         ignore_index=True).drop_duplicates("image_id")

plaus_records = []
for img_id, mask_path in mask_index.items():
    img_path = os.path.join(PLAUS_IMG_DIR, f"{img_id}.jpg")
    if not os.path.exists(img_path):
        continue
    row = {"image_id": img_id, "img_path": img_path, "mask_path": mask_path}
    match = _all_labels[_all_labels["image_id"] == img_id]
    row["label_idx"]  = int(match.iloc[0]["label_idx"])  if len(match) else -1
    row["label_name"] = match.iloc[0]["label_name"]      if len(match) else "unknown"
    plaus_records.append(row)

plaus_df = pd.DataFrame(plaus_records)

# Lesion area fraction & size group — computed on plaus_df, used in Phase E.
# ~19% of masked images have lesion area < 5% of the image area.
# A correctly centred heatmap on a tiny lesion scores near-zero IoU regardless of
# method quality — this is a metric artefact, not a model failure. Phase E stratifies
# results by lesion_size_group ("small" < 5%, "standard" ≥ 5%).
def compute_lesion_area_fraction(mask_path, target_size=(IMG_SIZE, IMG_SIZE)):
    if not mask_path or not os.path.exists(mask_path):
        return float("nan")
    mask = np.array(Image.open(mask_path).convert("L").resize(target_size))
    return (mask > 127).sum() / (target_size[0] * target_size[1])

plaus_df["lesion_area_frac"] = plaus_df["mask_path"].map(compute_lesion_area_fraction)
plaus_df["lesion_size_group"] = plaus_df["lesion_area_frac"].map(
    lambda x: "small"    if (x == x and x < 0.05)  else
              "standard" if (x == x and x >= 0.05) else float("nan")
)

small_n = (plaus_df["lesion_size_group"] == "small").sum()
std_n   = (plaus_df["lesion_size_group"] == "standard").sum()
denom   = small_n + std_n if (small_n + std_n) > 0 else 1

print(f"Total masks available        : {len(mask_index)}")
print(f"Plausibility images on disk  : {len(plaus_df)}")
print(f"  small lesions  (<5%)       : {small_n}  ({small_n/denom*100:.1f}%)")
print(f"  standard lesions           : {std_n}")
print(f"Attribute map images         : {len(attr_index)}")
print()
print("NOTE: train/val/test splits share NO image IDs with the plausibility subset.")
print("      has_mask on those splits is always False — this is expected and correct.")
print("      Phase E runs plausibility evaluation exclusively on plaus_df.")


# %% [markdown]
# ## A.5 — Class Weights & Normalisation Stats
# Inverse-frequency class weights and per-channel mean/std from the train split. Saved to `prepared/`.

# %%
# ── Class weights (inverse frequency) ───────────────────────────────────────
counts = Counter(train_df["label_idx"].values)
total  = sum(counts.values())
class_weights = {int(k): total / (NUM_CLASSES * v) for k, v in counts.items()}
with open(os.path.join(PREP_ROOT, "class_weights.json"), "w") as f:
    json.dump(class_weights, f, indent=2)
print("Class weights (higher = rarer):")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:6s} [{i}]: {class_weights.get(i, 0):.4f}")

# ── Per-channel mean / std ───────────────────────────────────────────────────
sample_ids   = train_df["image_id"].sample(min(2000, len(train_df)), random_state=SEED)
pixel_sum    = np.zeros(3, dtype=np.float64)
pixel_sq_sum = np.zeros(3, dtype=np.float64)
n_pixels     = 0

for img_id in tqdm(sample_ids, desc="Computing stats"):
    raw  = np.array(Image.open(find_image(img_id, TRAIN_IMG))
                    .resize((IMG_SIZE, IMG_SIZE)))
    img  = apply_color_constancy(raw) / 255.0
    flat = img.reshape(-1, 3)
    pixel_sum    += flat.sum(0)
    pixel_sq_sum += (flat ** 2).sum(0)
    n_pixels     += flat.shape[0]

mean = pixel_sum    / n_pixels
std  = np.sqrt(pixel_sq_sum / n_pixels - mean ** 2)
data_stats = {"mean": mean.tolist(), "std": std.tolist()}
with open(os.path.join(PREP_ROOT, "data_stats.json"), "w") as f:
    json.dump(data_stats, f, indent=2)
print(f"\nTrain mean : {mean.round(4)}")
print(f"Train std  : {std.round(4)}")

# %% [markdown]
# ## A.6 — Dataset Class & DataLoaders
# `ISICSkinDataset` with per-split `img_dir`. Albumentations augmentations. `WeightedRandomSampler` corrects class imbalance.

# %%
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ISICSkinDataset(Dataset):
    """ISIC 2018 skin lesion dataset with optional segmentation masks."""

    def __init__(self, df, img_dir, transform=None, load_masks=False):
        self.df         = df.reset_index(drop=True)
        self.img_dir    = img_dir
        self.transform  = transform
        self.load_masks = load_masks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        img  = apply_color_constancy(
                   np.array(Image.open(
                       find_image(row['image_id'], self.img_dir)
                   ).convert("RGB"))
               )

        mask = None
        if self.load_masks and row.get("has_mask", False):
            mp = row.get("mask_path")
            if mp and os.path.exists(mp):
                mask = np.array(Image.open(mp).convert("L"))

        if self.transform:
            if mask is not None:
                out  = self.transform(image=img, mask=mask)
                img  = out["image"]
                mask = out["mask"].float() / 255.0
            else:
                img  = self.transform(image=img)["image"]

        label = torch.tensor(row["label_idx"], dtype=torch.long)
        meta  = {"image_id": row["image_id"],
                 "has_mask": bool(row.get("has_mask", False))}
        return img, label, mask if mask is not None else torch.zeros(1), meta


# ── Augmentation pipelines ───────────────────────────────────────────────────
with open(os.path.join(PREP_ROOT, "data_stats.json")) as f:
    stats = json.load(f)

# Augmentation matches the ISIC 2018 winners: full ±180° rotation, flip,
# brightness/contrast/saturation jitter, affine, and random crop.
train_transform = A.Compose([
    A.Resize(IMG_SIZE + 32, IMG_SIZE + 32),   # slight oversample for random crop
    A.RandomCrop(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=180, p=0.7),
    A.Affine(translate_percent=0.05, scale=(0.9, 1.1), p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
    A.Normalize(mean=stats["mean"], std=stats["std"]),
    ToTensorV2(),
])

eval_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=stats["mean"], std=stats["std"]),
    ToTensorV2(),
])

# ── Build datasets & loaders ─────────────────────────────────────────────────
train_ds = ISICSkinDataset(train_df, TRAIN_IMG, train_transform)
val_ds   = ISICSkinDataset(val_df,   VAL_IMG,   eval_transform)
test_ds  = ISICSkinDataset(test_df,  TEST_IMG,  eval_transform, load_masks=True)

with open(os.path.join(PREP_ROOT, "class_weights.json")) as f:
    cw = json.load(f)
_persist = NUM_WORKERS > 0   # keep workers alive across epochs — eliminates
                             # "can only test a child process" GC noise

# ── Combo A: WeightedRandomSampler ──────────────────────────────────────────
# Sample each image with probability proportional to its inverse class frequency.
# Paired with plain CrossEntropyLoss (no class weights) — using both over-corrects
# and collapses precision on MEL/NV.
sample_weights = torch.tensor(
    [cw[str(int(lbl))] for lbl in train_df["label_idx"].values],
    dtype=torch.float32,
)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                replacement=True)
train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=_persist)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=_persist)
test_loader  = DataLoader(test_ds,  1,          shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=_persist)

# ── Sanity check ─────────────────────────────────────────────────────────────
imgs, labels, masks, meta = next(iter(train_loader))
print(f"Batch shape  : {imgs.shape}")
print(f"Label dist   : {labels.bincount(minlength=NUM_CLASSES).tolist()}")
print(f"Pixel range  : [{imgs.min():.3f}, {imgs.max():.3f}]")

# %% [markdown]
# ## A.7 — Visual Sanity Check
# Training batch grid (8 images) and test images with segmentation mask overlays.

# %%
MN  = np.array(stats["mean"])
STD = np.array(stats["std"])

def denorm(t):
    return np.clip(t.permute(1, 2, 0).numpy() * STD + MN, 0, 1)

# ── Training batch grid ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(denorm(imgs[i]))
    ax.set_title(CLASS_NAMES[labels[i].item()], fontsize=11, fontweight="bold")
    ax.axis("off")
fig.suptitle("Training Batch — WeightedRandomSampler", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_ROOT, "sanity_train_batch.png"), dpi=100, bbox_inches="tight")
plt.show()

# ── Plausibility images + green mask overlays ───────────────────────────────────
# Masks are in Data/plausibility/masks/, images in Data/plausibility/images/.
# The 3,694 mask images are a separate annotated subset — not strictly test-split.
PLAUS_IMG_DIR = os.path.join(DATA_ROOT, "plausibility", "images")

sample_ids = random.sample(list(mask_index.keys()), min(6, len(mask_index)))

fig, axes = plt.subplots(2, 3, figsize=(13, 9))
for ax, img_id in zip(axes.flat, sample_ids):
    # Load original image
    img_path = os.path.join(PLAUS_IMG_DIR, f"{img_id}.jpg")
    raw = np.array(
        Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    ) / 255.0

    # Load binary mask
    mask_arr = np.array(
        Image.open(mask_index[img_id]).convert("L").resize((IMG_SIZE, IMG_SIZE))
    )
    binary_mask = mask_arr > 127

    # Build green RGBA overlay (R=0, G=1, B=0, A=0.45 on lesion pixels)
    overlay = np.zeros((IMG_SIZE, IMG_SIZE, 4), dtype=np.float32)
    overlay[binary_mask] = [0.0, 0.9, 0.2, 0.45]

    ax.imshow(raw)
    ax.imshow(overlay, interpolation="nearest")

    # Label with class if available in any split
    for df in [train_df, val_df, test_df]:
        match = df[df["image_id"] == img_id]
        if len(match):
            label_str = match.iloc[0]["label_name"]
            break
    else:
        label_str = "unknown"

    coverage = binary_mask.sum() / binary_mask.size * 100
    ax.set_title(f"{img_id}\n{label_str}  |  coverage {coverage:.1f}%",
                 fontsize=8)
    ax.axis("off")

fig.suptitle("Plausibility Subset — Lesion Segmentation Mask Overlay (green)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_ROOT, "sanity_mask_overlay.png"), dpi=100, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# # Phase B — Model Definition & Training
# > **Thesis §5.2 — Model Architectures**
# 
# Fine-tune four pretrained architectures on ISIC 2018.
# Two CNN baselines (EfficientNet-B0, DenseNet-121) and two Vision Transformers
# (ViT-Base/16, Swin-Tiny).
# 
# ### Steps
# B.1 Model factory · B.2 Weighted CE & training loop · B.3 Train all 4 · B.4 Test evaluation · B.5 Correct-only eval subset · B.6 Weighted ensemble

# %% [markdown]
# ## B.1 — Model Factory
# Creates any of the 4 architectures via `timm` with the correct 7-class head.
# Also returns the target layer name used by Grad-CAM in Phase C.

# %%
import timm

# Grad-CAM target layers (last feature-producing conv layer per CNN arch)
GRADCAM_LAYERS = {
    "efficientnet_b2": "conv_head",
    "densenet121":     "features.denseblock4.denselayer16.conv2",
    "convnext_tiny":   "stages.3.blocks.2.conv_dw",
    "vit_base_16":     None,   # uses Attention Rollout in Phase C
    "swin_tiny":       None,   # uses Attention Rollout in Phase C
}

def create_model(arch_key):
    """Build architecture from ARCHITECTURES config.
    Returns: (model on DEVICE, gradcam_layer_name, family)
    """
    cfg   = ARCHITECTURES[arch_key]
    model = timm.create_model(cfg["timm_name"], pretrained=True,
                              num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    return model, GRADCAM_LAYERS.get(arch_key), cfg["family"]


# Sanity: instantiate all 4, print param counts
for name in ARCHITECTURES:
    m, tl, fam = create_model(name)
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"{name:20s} | {fam:3s} | {n_params:5.1f}M params"
          f" | grad-cam layer: {tl}")
    del m
torch.cuda.empty_cache()


# %% [markdown]
# ## B.2 — Loss Function & Training Loop
# **Why Class-Weighted CrossEntropy?**
# The ISIC 2018 competition winners tested Focal Loss, oversampling, triplet loss,
# and clustering — none matched the ≥10% MCA improvement from class-weighted
# CrossEntropy alone. Inverse-frequency weights (computed in A.5) penalise
# misclassification of DF (115 imgs) and VASC (142 imgs) proportionally.
#
# **Loop:** AdamW + cosine annealing + early stopping on val loss.

# %%
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             recall_score, f1_score, precision_score)
from tqdm.auto import tqdm
import copy


class ClassBalancedFocalLoss(nn.Module):
    """Focal Loss modulated by inverse-frequency class weights.

    Combines Paper 2's finding (Focal Loss → lower variance, more stable minority
    recall) with class weighting (the primary imbalance fix). gamma=2 is standard.
    """
    def __init__(self, weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weights", weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weights, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# Minority class indices (CLASS_NAMES = ["MEL","NV","BCC","AKIEC","BKL","DF","VASC"])
# AKIEC=3, DF=5, VASC=6 — fewest samples; BCC=2 also scarce (514 imgs)
MINORITY_CLASS_INDICES = frozenset([2, 3, 5, 6])


def mixup_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """CutMix: paste a random patch from another image. lam = 1 - patch_area/img_area."""
    lam = float(np.random.beta(alpha, alpha))
    B, _, H, W = x.shape
    idx = torch.randperm(B, device=x.device)
    cut_rat = float(np.sqrt(1.0 - lam))
    cw, ch  = int(W * cut_rat), int(H * cut_rat)
    cx, cy  = np.random.randint(W), np.random.randint(H)
    x1, x2  = max(0, cx - cw // 2), min(W, cx + cw // 2)
    y1, y2  = max(0, cy - ch // 2), min(H, cy + ch // 2)
    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x_mix, y, y[idx], lam


def tta_predict(model: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """5-view TTA: original + H-flip + V-flip + 90° rot + 180° rot.

    Returns averaged softmax probabilities of shape (N, NUM_CLASSES).
    Defined here (before training) so val-snapshot can reuse the same
    inference path as test, keeping calibration/α-tuning aligned.
    """
    views = [
        imgs,
        imgs.flip(-1),
        imgs.flip(-2),
        torch.rot90(imgs, 1, [-2, -1]),
        torch.rot90(imgs, 2, [-2, -1]),
    ]
    probs = torch.stack([torch.softmax(model(v), dim=1) for v in views])
    return probs.mean(0)


def _collect_val_outputs(model, val_loader, criterion):
    """Run one val pass, return (loss, preds, labels, probs, logits)."""
    model.eval()
    loss_sum = 0.0
    all_preds, all_labels, all_probs, all_logits = [], [], [], []
    with torch.no_grad():
        for imgs, labels, _, _ in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits        = model(imgs)
            loss_sum     += criterion(logits, labels).item() * imgs.size(0)
            probs         = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            all_logits.extend(logits.cpu().numpy())
    return (loss_sum / len(val_loader.dataset),
            np.array(all_preds), np.array(all_labels),
            np.array(all_probs), np.array(all_logits))


def train_one_model(arch_key, train_loader, val_loader):
    print(f"\n{chr(61)*60}\n  Training: {arch_key}\n{chr(61)*60}", flush=True)

    model, _, _ = create_model(arch_key)

    # Class-Balanced Focal Loss (γ=2) with inverse-frequency weights by default.
    # Focal term focuses on hard/minority examples; weights further boost rare
    # classes. Falls back to label-smoothed CE if USE_FOCAL_LOSS is False.
    if USE_FOCAL_LOSS:
        w_tensor = torch.tensor(
            [class_weights[i] for i in range(NUM_CLASSES)],
            dtype=torch.float32, device=DEVICE,
        )
        criterion = ClassBalancedFocalLoss(w_tensor, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=PATIENCE * 3, eta_min=1e-6)

    best_val_f1      = -1.0
    best_state       = None
    patience_counter = 0
    history          = {"train_loss": [], "val_loss": [], "val_bacc": [],
                        "val_auc": [], "val_f1_macro": [],
                        "val_per_class_recall": []}

    epoch_bar = tqdm(range(MAX_EPOCHS), desc=arch_key, unit="epoch")
    for epoch in epoch_bar:
        # ─ Train ─
        model.train()
        running_loss = 0.0
        batch_bar = tqdm(train_loader, desc=f"  Ep {epoch+1:02d} train",
                         leave=False, unit="batch")
        for imgs, labels, _, _ in batch_bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            # Mixup or CutMix on a fraction of batches. When aug triggers,
            # CUTMIX_PROB controls CutMix-vs-Mixup ratio.
            if MIXUP_ALPHA > 0 and np.random.rand() < MIXUP_PROB:
                if CUTMIX_ALPHA > 0 and np.random.rand() < CUTMIX_PROB:
                    mixed_x, y_a, y_b, lam = cutmix_batch(imgs, labels, alpha=CUTMIX_ALPHA)
                else:
                    mixed_x, y_a, y_b, lam = mixup_batch(imgs, labels, alpha=MIXUP_ALPHA)
                logits = model(mixed_x)
                loss   = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = running_loss / len(train_loader.dataset)

        # ─ Validate ─
        val_loss, val_preds, val_labels, val_probs, _ = _collect_val_outputs(
            model, val_loader, criterion
        )
        val_bacc         = balanced_accuracy_score(val_labels, val_preds)
        val_auc          = roc_auc_score(val_labels, val_probs,
                                         multi_class="ovr", average="macro")
        val_f1_macro     = f1_score(val_labels, val_preds,
                                    average="macro", zero_division=0)
        per_class_recall = recall_score(val_labels, val_preds,
                                        average=None, zero_division=0).tolist()
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_bacc"].append(val_bacc)
        history["val_auc"].append(val_auc)
        history["val_f1_macro"].append(val_f1_macro)
        history["val_per_class_recall"].append(per_class_recall)

        recall_str = "  ".join(
            f"{n}={r:.2f}" for n, r in zip(CLASS_NAMES, per_class_recall)
        )
        epoch_bar.set_postfix(
            tr_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            bAcc=f"{val_bacc:.4f}",
            F1=f"{val_f1_macro:.4f}",
        )
        print(
            f"  [{arch_key}] Ep {epoch+1:02d}/{MAX_EPOCHS}"
            f" | train={train_loss:.4f} | val={val_loss:.4f}"
            f" | bAcc={val_bacc:.4f} | AUC={val_auc:.4f}"
            f" | F1={val_f1_macro:.4f}",
            flush=True,
        )
        print(f"    per-class recall: {recall_str}", flush=True)

        # Early-stop on macro-F1 with a min-delta: AUC was too flat to trigger
        # early stopping (0.95→0.975 over 40 epochs). F1 moves with the actual
        # decisions the classifier makes, so the patience counter has signal.
        if val_f1_macro > best_val_f1 + EARLY_STOP_MIN_DELTA:
            best_val_f1      = val_f1_macro
            best_state       = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                tqdm.write(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    # One clean val pass on the best checkpoint — used for temperature scaling
    # and per-class threshold tuning in the ensemble step.
    # Non-TTA pass is kept for logits (temperature scaling needs pre-softmax).
    _, _, best_val_labels, _, best_val_logits = _collect_val_outputs(
        model, val_loader, criterion
    )
    # TTA pass: matches test-time inference so calibration/α-tuning are
    # aligned with the ensemble inputs. Fixes train/test distribution skew
    # in val_probs observed in 2026-04-20_21-14.
    model.eval()
    all_probs_tta = []
    with torch.no_grad():
        for imgs, _, _, _ in val_loader:
            imgs = imgs.to(DEVICE)
            all_probs_tta.append(tta_predict(model, imgs).cpu().numpy())
    best_val_probs = np.concatenate(all_probs_tta, axis=0)
    val_snapshot = {
        "labels": best_val_labels,
        "probs":  best_val_probs,
        "logits": best_val_logits,
        "best_f1": best_val_f1,
    }
    return model, history, val_snapshot

# %% [markdown]
# ## B.3 — Train All 4 Models
# Loop over all architectures sequentially. Each model is moved to CPU after
# training to free VRAM for the next one. Checkpoints saved to `outputs/`.

# %%
trained_models  = {}   # arch_key → model (on CPU)
train_histories = {}   # arch_key → history dict
val_snapshots   = {}   # arch_key → {labels, probs, logits, best_f1}

for arch_key in ARCHITECTURES:
    model, history, val_snapshot = train_one_model(arch_key, train_loader, val_loader)
    val_snapshots[arch_key] = val_snapshot

    ckpt_path = os.path.join(OUT_ROOT, f"{arch_key}_best.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved: {ckpt_path}")

    trained_models[arch_key]  = model.cpu()
    train_histories[arch_key] = history
    torch.cuda.empty_cache()

    # Save per-model training history
    with open(os.path.join(OUT_ROOT, f"history_{arch_key}.json"), "w") as _f:
        json.dump(history, _f, indent=2)

print("\n✓ All 4 models trained and saved.")

# %%
# ── Training curves ──────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
colors = {"train_loss": "#e05c5c", "val_loss": "#5c9be0",
          "val_f1":     "#5cbf7a", "val_auc": "#b57edc"}

for ax, (arch_key, hist) in zip(axes.flat, train_histories.items()):
    epochs = range(1, len(hist["train_loss"]) + 1)
    ax2 = ax.twinx()

    ax.plot(epochs, hist["train_loss"], color=colors["train_loss"],
            lw=1.8, label="Train loss")
    ax.plot(epochs, hist["val_loss"],   color=colors["val_loss"],
            lw=1.8, linestyle="--", label="Val loss")
    ax2.plot(epochs, hist["val_f1_macro"], color=colors["val_f1"],
             lw=1.5, linestyle=":", label="Val F1 (macro)")
    ax2.plot(epochs, hist["val_auc"],      color=colors["val_auc"],
             lw=1.2, linestyle=":", alpha=0.7, label="Val AUC (macro)")

    ax.set_title(arch_key, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax2.set_ylabel("Macro F1 / AUC", color=colors["val_f1"])
    ax2.tick_params(axis="y", labelcolor=colors["val_f1"])
    ax2.set_ylim(0, 1)

    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper right")

fig.suptitle("Training Curves — All Architectures", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_ROOT, "training_curves.png"), dpi=100, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## B.4 — Test-Set Classification Performance
# Per-model classification report and confusion matrix. This is **not** the XAI
# evaluation — it establishes baseline performance to contextualise XAI results.
#
# TTA (Test Time Augmentation): 5 forward passes per image (original + H-flip +
# V-flip + 90° rot + 180° rot), softmax averaged. Expected +3–6% bAcc at zero
# retraining cost. Especially beneficial for minority classes with few test samples.
#
# > **Target:** ≥80% balanced accuracy on ≥2 models. Models below ~60% bAcc produce
# > unreliable explanations and should be flagged.

# %%
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


test_results = {}   # arch_key → {preds, labels, probs, report}

for arch_key, model in trained_models.items():
    model = model.to(DEVICE).eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels, _, _ in test_loader:
            imgs   = imgs.to(DEVICE)
            probs  = tta_predict(model, imgs)
            all_preds.extend(probs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    report = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    test_results[arch_key] = {
        "preds":  np.array(all_preds),
        "labels": np.array(all_labels),
        "probs":  np.array(all_probs),
        "report": report,
    }

    bacc = balanced_accuracy_score(all_labels, all_preds)
    flag = "  ⚠️  LOW" if bacc < 0.60 else ""
    print(f"\n{'='*40} {arch_key} {'='*40}")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))
    print(f"Balanced accuracy: {bacc:.4f}{flag}")

    model = model.cpu()
    torch.cuda.empty_cache()

# ── Confusion matrices ───────────────────────────────────────────────────────────────
_n = len(test_results)
_ncols = min(2, _n)
_nrows = (_n + _ncols - 1) // _ncols
fig, axes = plt.subplots(_nrows, _ncols, figsize=(7 * _ncols, 5.5 * _nrows), squeeze=False)
for ax, (arch_key, res) in zip(axes.flat, test_results.items()):
    cm = confusion_matrix(res["labels"], res["preds"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, vmin=0, vmax=1,
                annot_kws={"size": 8}, linewidths=0.4)
    bacc = test_results[arch_key]["report"]["macro avg"]["recall"]
    ax.set_title(f"{arch_key}\nbAcc={bacc:.3f}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

fig.suptitle("Normalised Confusion Matrices — Test Set", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_ROOT, "confusion_matrices.png"), dpi=100, bbox_inches="tight")
plt.show()

# Per-model metrics JSON
summary_lines = []
for arch_key, res in test_results.items():
    bacc = balanced_accuracy_score(res["labels"], res["preds"])
    metrics_out = {
        "balanced_accuracy": round(float(bacc), 4),
        "classification_report": res["report"],
    }
    with open(os.path.join(OUT_ROOT, f"metrics_{arch_key}.json"), "w") as _f:
        json.dump(metrics_out, _f, indent=2)
    flag = "LOW" if bacc < 0.60 else "OK"
    summary_lines.append(f"{arch_key:20s} bAcc={bacc:.4f}  [{flag}]")

# Plain-text summary report (detailed)
import datetime
from sklearn.metrics import (top_k_accuracy_score, matthews_corrcoef,
                              cohen_kappa_score)

def _fmt_cm(cm: np.ndarray, names: list[str]) -> str:
    width = max(6, max(len(n) for n in names) + 1)
    header = " " * width + "".join(f"{n:>{width}}" for n in names) + "   (row=true)"
    lines  = [header]
    for i, n in enumerate(names):
        row = f"{n:<{width}}" + "".join(f"{cm[i, j]:>{width}d}" for j in range(len(names)))
        lines.append(row)
    return "\n".join(lines)


def _per_model_section(arch_key: str, labels: np.ndarray, preds: np.ndarray,
                       probs: np.ndarray) -> str:
    report = classification_report(labels, preds, target_names=CLASS_NAMES,
                                   zero_division=0, digits=4)
    bacc   = balanced_accuracy_score(labels, preds)
    f1m    = f1_score(labels, preds, average="macro", zero_division=0)
    f1w    = f1_score(labels, preds, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    top3   = top_k_accuracy_score(labels, probs, k=3, labels=list(range(NUM_CLASSES)))
    mcc    = matthews_corrcoef(labels, preds)
    kappa  = cohen_kappa_score(labels, preds)
    cm     = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

    out = []
    out.append("=" * 72)
    out.append(f"MODEL: {arch_key}")
    out.append("=" * 72)
    out.append(f"Balanced accuracy : {bacc:.4f}")
    out.append(f"Accuracy (top-1)  : {(preds == labels).mean():.4f}")
    out.append(f"Accuracy (top-3)  : {top3:.4f}")
    out.append(f"Macro F1          : {f1m:.4f}")
    out.append(f"Weighted F1       : {f1w:.4f}")
    out.append(f"Macro AUC-ROC     : {auc:.4f}")
    out.append(f"MCC               : {mcc:.4f}")
    out.append(f"Cohen kappa       : {kappa:.4f}")
    out.append("")
    out.append("Per-class report:")
    out.append(report)
    out.append("Confusion matrix (counts):")
    out.append(_fmt_cm(cm, CLASS_NAMES))
    out.append("")
    return "\n".join(out)



# %% [markdown]
# ## B.5 — Correct-Only Evaluation Subset
# Per thesis §5.5, XAI evaluation is conducted **only on correctly classified images**.
# Build `eval_subsets` here — every downstream phase (C–F) draws from this.

# %%
eval_subsets = {}   # arch_key → DataFrame of correctly-classified test rows

for arch_key in ARCHITECTURES:
    res          = test_results[arch_key]
    correct_mask = res["preds"] == res["labels"]
    correct_df   = test_df[correct_mask].copy().reset_index(drop=True)
    correct_df["pred"]  = res["preds"][correct_mask]
    correct_df["probs"] = list(res["probs"][correct_mask])
    eval_subsets[arch_key] = correct_df

    n_total   = len(test_df)
    n_correct = correct_mask.sum()
    print(f"{arch_key:20s} : {n_correct:4d}/{n_total} correct "
          f"({n_correct/n_total*100:.1f}%)"
          f"  [plausibility evaluated separately on plaus_df in Phase E]")

print("\neval_subsets ready — all XAI phases operate on these.")

# %% [markdown]
# ## B.6 — Weighted Ensemble Evaluation
# Weighted average of softmax probability vectors across all 4 models.
# Weights are derived from each model's best validation balanced accuracy —
# better val performance → higher weight. Follows the ISIC 2018 winners'
# approach of tuning weights on the validation set (∑wᵢ = 1).
#
# FinalScore = Σ wᵢ · sᵢ   where sᵢ is the 7-dim softmax vector for model i.

# %%
# ── Temperature scaling (per model) ──────────────────────────────────────────
# Fit a single scalar T per model by minimising NLL on the val snapshot.
# Better-calibrated probabilities → better weighted averaging in the ensemble.
def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    labels_t = torch.tensor(val_labels, dtype=torch.long)
    log_T    = torch.zeros(1, requires_grad=True)   # parameterise log T for stability
    opt      = torch.optim.LBFGS([log_T], lr=0.1, max_iter=50)
    nll      = nn.CrossEntropyLoss()
    def _closure():
        opt.zero_grad()
        loss = nll(logits_t / log_T.exp(), labels_t)
        loss.backward()
        return loss
    opt.step(_closure)
    T = float(log_T.exp().item())
    return float(np.clip(T, 0.1, 10.0))


def apply_temperature(probs_or_logits: np.ndarray, T: float, from_logits: bool) -> np.ndarray:
    if from_logits:
        x = probs_or_logits / T
    else:
        # convert probs → logits via log, then rescale
        x = np.log(np.clip(probs_or_logits, 1e-12, 1.0)) / T
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


model_temperatures = {}
for k in ARCHITECTURES:
    snap = val_snapshots[k]
    T = fit_temperature(snap["logits"], snap["labels"])
    model_temperatures[k] = T
    print(f"  {k:20s}  temperature T={T:.3f}")

# ── Per-class ensemble weights ───────────────────────────────────────────────
# Instead of one scalar weight per model, use a (K × C) matrix where W[k, c]
# is proportional to model k's per-class F1 on val for class c. Each class
# picks whichever models are most reliable for it. Columns sum to 1.
val_labels_ref = val_snapshots[next(iter(ARCHITECTURES))]["labels"]
_per_model_f1_by_class = {}
for k in ARCHITECTURES:
    preds_k = val_snapshots[k]["probs"].argmax(axis=1)
    _per_model_f1_by_class[k] = f1_score(
        val_labels_ref, preds_k, average=None, zero_division=0,
        labels=list(range(NUM_CLASSES))
    )
_W = np.stack([_per_model_f1_by_class[k] for k in ARCHITECTURES])  # (K, C)
_W = _W + 1e-6
ENSEMBLE_WEIGHTS_PC = _W / _W.sum(axis=0, keepdims=True)  # per-class, sums to 1 per column

# Also compute a scalar summary for logging / backward compat
_val_f1s = {k: val_snapshots[k]["best_f1"] for k in ARCHITECTURES}
_total   = sum(_val_f1s.values())
ENSEMBLE_WEIGHTS = {k: v / _total for k, v in _val_f1s.items()}

print("Per-class ensemble weights (row=model, col=class):")
print("  " + "  ".join(f"{c:>6s}" for c in CLASS_NAMES))
for i, k in enumerate(ARCHITECTURES):
    row = "  ".join(f"{ENSEMBLE_WEIGHTS_PC[i, c]:6.3f}" for c in range(NUM_CLASSES))
    print(f"  {k:20s} {row}  (val_F1={_val_f1s[k]:.4f}, T={model_temperatures[k]:.3f})")

# Temperature-scale each model's *test* probs, then apply per-class weights.
test_probs_cal = {
    k: apply_temperature(test_results[k]["probs"], model_temperatures[k], from_logits=False)
    for k in ARCHITECTURES
}
# Per-class weighted sum: probs[:, c] = Σ_k W[k,c] * probs_k[:, c]
_keys = list(ARCHITECTURES)
_P_test = np.stack([test_probs_cal[k] for k in _keys])  # (K, N, C)
ensemble_probs = (ENSEMBLE_WEIGHTS_PC[:, None, :] * _P_test).sum(axis=0)

# ── Per-class threshold tuning on val ────────────────────────────────────────
# Default argmax often over-predicts the majority class (NV). For each class we
# scale its column by a factor α_c chosen on val. With per-class ensemble
# weights applied first, α picks up residual bias not fixed by weighting.
val_probs_cal = {
    k: apply_temperature(val_snapshots[k]["probs"], model_temperatures[k], from_logits=False)
    for k in ARCHITECTURES
}
_P_val = np.stack([val_probs_cal[k] for k in _keys])
val_ensemble_probs  = (ENSEMBLE_WEIGHTS_PC[:, None, :] * _P_val).sum(axis=0)
val_ensemble_labels = val_labels_ref

def tune_class_scales(probs: np.ndarray, labels: np.ndarray,
                      grid=np.linspace(0.8, 1.5, 15)) -> np.ndarray:
    """Coordinate search on α ∈ R^C (per-class multiplicative prior).

    Objective: 0.5·macro-F1 + 0.5·balanced-accuracy. Pure macro-F1 tuning
    over-suppressed minority classes (AKIEC, VASC) in 2026-04-20_21-14;
    the bAcc term protects per-class recall. Grid clamped to [0.8, 1.5]
    so minority classes cannot be aggressively demoted.
    """
    C      = probs.shape[1]
    alpha  = np.ones(C)
    for _ in range(3):                                 # 3 sweeps is plenty
        for c in range(C):
            best_score, best_a = -1.0, alpha[c]
            for a in grid:
                alpha[c] = a
                preds    = (probs * alpha).argmax(axis=1)
                f1       = f1_score(labels, preds, average="macro", zero_division=0)
                bacc     = balanced_accuracy_score(labels, preds)
                score    = 0.5 * f1 + 0.5 * bacc
                if score > best_score:
                    best_score, best_a = score, a
            alpha[c] = best_a
    return alpha

class_scales = tune_class_scales(val_ensemble_probs, val_ensemble_labels)
print("Per-class scales tuned on val (0.5·macro-F1 + 0.5·bAcc, α∈[0.8,1.5]):")
for c, s in zip(CLASS_NAMES, class_scales):
    print(f"  {c:8s}  α={s:.3f}")

ensemble_probs_tuned = ensemble_probs * class_scales
ensemble_preds       = ensemble_probs_tuned.argmax(axis=1)
ensemble_labels      = test_results[next(iter(ARCHITECTURES))]["labels"]

from sklearn.metrics import classification_report, balanced_accuracy_score
ensemble_bacc = balanced_accuracy_score(ensemble_labels, ensemble_preds)
print(f"\n{'='*60}")
print(f"Weighted Ensemble  bAcc = {ensemble_bacc:.4f}")
print(f"{'='*60}")
print(classification_report(ensemble_labels, ensemble_preds,
                            target_names=CLASS_NAMES, zero_division=0))

# Confusion matrix
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix
fig, ax = plt.subplots(figsize=(7, 6))
cm_norm = confusion_matrix(ensemble_labels, ensemble_preds).astype(float)
cm_norm /= cm_norm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, vmin=0, vmax=1, annot_kws={"size": 8}, linewidths=0.4)
ax.set_title(f"Weighted Ensemble — bAcc={ensemble_bacc:.3f}", fontsize=12, fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.tick_params(axis="x", rotation=45); ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_ROOT, "confusion_ensemble.png"), dpi=100, bbox_inches="tight")
plt.show()

# Save ensemble metrics
with open(os.path.join(OUT_ROOT, "metrics_ensemble.json"), "w") as _f:
    json.dump({
        "weights":       ENSEMBLE_WEIGHTS,
        "weights_per_class": {
            k: {c: float(ENSEMBLE_WEIGHTS_PC[i, j]) for j, c in enumerate(CLASS_NAMES)}
            for i, k in enumerate(ARCHITECTURES)
        },
        "temperatures":  model_temperatures,
        "class_scales":  {c: float(s) for c, s in zip(CLASS_NAMES, class_scales)},
        "balanced_accuracy": round(float(ensemble_bacc), 4),
        "macro_f1":      round(float(f1_score(ensemble_labels, ensemble_preds,
                                              average="macro", zero_division=0)), 4),
        "classification_report": classification_report(
            ensemble_labels, ensemble_preds,
            target_names=CLASS_NAMES, zero_division=0, output_dict=True
        ),
    }, _f, indent=2)
print(f"Ensemble metrics saved.")

# Store ensemble results for potential use in XAI phases
ensemble_results = {
    "preds":  ensemble_preds,
    "labels": ensemble_labels,
    "probs":  ensemble_probs,
}

# Detailed evaluation report (written after all ensemble variables are available)
report_path = os.path.join(OUT_ROOT, "evaluation_report.txt")
with open(report_path, "w", encoding="utf-8") as _f:
    _f.write("ISIC2018 XAI — Evaluation Report\n")
    _f.write(f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    _f.write(f"DEBUG     : {DEBUG}\n")
    _f.write(f"IMG_SIZE  : {IMG_SIZE}   BATCH_SIZE: {BATCH_SIZE}\n")
    _f.write(f"LR        : {LR}   WEIGHT_DECAY: {WEIGHT_DECAY}\n")
    _f.write(f"Label smoothing: {LABEL_SMOOTHING}   Mixup α={MIXUP_ALPHA} p={MIXUP_PROB}\n")
    _f.write("=" * 72 + "\n")

    _f.write("\nSUMMARY\n")
    _f.write("-" * 72 + "\n")
    for line in summary_lines:
        _f.write(line + "\n")
    _f.write("\n")

    # Training summary per model
    _f.write("TRAINING SUMMARY\n")
    _f.write("-" * 72 + "\n")
    for arch_key in ARCHITECTURES:
        h = train_histories[arch_key]
        epochs_run = len(h["train_loss"])
        _f.write(
            f"{arch_key:20s}  epochs={epochs_run:3d}  "
            f"best_val_F1={max(h['val_f1_macro']):.4f}  "
            f"best_val_bAcc={max(h['val_bacc']):.4f}  "
            f"best_val_AUC={max(h['val_auc']):.4f}  "
            f"final_train_loss={h['train_loss'][-1]:.4f}  "
            f"final_val_loss={h['val_loss'][-1]:.4f}\n"
        )
    _f.write("\n")

    # Per-model detailed sections
    for arch_key, res in test_results.items():
        _f.write(_per_model_section(arch_key, res["labels"], res["preds"], res["probs"]))

    # Ensemble section
    _f.write("=" * 72 + "\n")
    _f.write("ENSEMBLE (temperature-scaled, per-class threshold-tuned)\n")
    _f.write("=" * 72 + "\n")
    _f.write("Model temperatures:\n")
    for k, T in model_temperatures.items():
        _f.write(f"  {k:20s}  T={T:.3f}\n")
    _f.write("Scalar weights (by val macro-F1, informational):\n")
    for k, w in ENSEMBLE_WEIGHTS.items():
        _f.write(f"  {k:20s}  w={w:.4f}\n")
    _f.write("Per-class ensemble weights (row=model, col=class; columns sum to 1):\n")
    _f.write("  " + " " * 22 + "  ".join(f"{c:>6s}" for c in CLASS_NAMES) + "\n")
    for i, k in enumerate(ARCHITECTURES):
        row = "  ".join(f"{ENSEMBLE_WEIGHTS_PC[i, j]:6.3f}" for j in range(NUM_CLASSES))
        _f.write(f"  {k:20s}  {row}\n")
    _f.write("Per-class prior scales α (tuned on val for 0.5·macro-F1 + 0.5·bAcc, α∈[0.8,1.5]):\n")
    for c, s in zip(CLASS_NAMES, class_scales):
        _f.write(f"  {c:8s}  α={s:.3f}\n")
    _f.write("\n")
    _f.write(_per_model_section("ensemble", ensemble_labels, ensemble_preds, ensemble_probs))

print("\n" + "=" * 60)
print("Summary:")
for line in summary_lines:
    print(" ", line)
print(f"Report saved to {report_path}")
