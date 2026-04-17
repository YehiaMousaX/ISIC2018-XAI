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
# │   ├── train.csv              # 10,015 rows | cols: image, MEL, NV, BCC, AKIEC, BKL, DF, VASC
# │   ├── val.csv                #    193 rows
# │   ├── test.csv               #  1,512 rows
# │   └── lesion_groupings.csv   # image, lesion_id, diagnosis_confirm_type
# ├── images/
# │   ├── train/  (10,015 jpgs)
# │   ├── val/    (   193 jpgs)
# │   └── test/   ( 1,512 jpgs)
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
    with open("/kaggle/input/hf-credentials/hf_token.txt") as f:
        secret_value = f.read().strip()
    login(secret_value)

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "info"
else:
    DATA_ROOT = "./Data"
    PREP_ROOT = "./prepared"
    OUT_ROOT  = "./outputs"

CSV_DIR   = os.path.join(DATA_ROOT, "csv")
TRAIN_IMG = os.path.join(DATA_ROOT, "images", "train")
VAL_IMG   = os.path.join(DATA_ROOT, "images", "val")
TEST_IMG  = os.path.join(DATA_ROOT, "images", "test")
MASK_DIR  = os.path.join(DATA_ROOT, "plausibility", "masks")
ATTR_DIR  = os.path.join(DATA_ROOT, "plausibility", "attributes")

# ── Model / training ───────────────────────────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 32
MAX_EPOCHS   = 50  if not DEBUG else 3
PATIENCE     = 7   if not DEBUG else 2
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES  = 7
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
    "efficientnet_b0": {"family": "cnn", "timm_name": "efficientnet_b0"},
    "densenet121":     {"family": "cnn", "timm_name": "densenet121"},
    "vit_base_16":     {"family": "vit", "timm_name": "vit_base_patch16_224"},
    "swin_tiny":       {"family": "vit", "timm_name": "swin_tiny_patch4_window7_224"},
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
        Image.open(os.path.join(TRAIN_IMG, f"{row['image_id']}.jpg"))
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
    raw  = np.array(Image.open(os.path.join(TRAIN_IMG, f"{img_id}.jpg"))
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
                       os.path.join(self.img_dir, f"{row['image_id']}.jpg")
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
sample_weights = [cw[str(int(l))] for l in train_df["label_idx"]]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

_persist = NUM_WORKERS > 0   # keep workers alive across epochs — eliminates
                             # "can only test a child process" GC noise
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
    "efficientnet_b0": "conv_head",
    "densenet121":     "features.denseblock4.denselayer16.conv2",
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import balanced_accuracy_score
from tqdm.auto import tqdm
import copy


def train_one_model(arch_key, train_loader, val_loader):
    print(f"\n{chr(61)*60}\n  Training: {arch_key}\n{chr(61)*60}", flush=True)

    model, _, _ = create_model(arch_key)

    with open(os.path.join(PREP_ROOT, "class_weights.json")) as f:
        cw = json.load(f)
    weights   = torch.tensor([cw[str(i)] for i in range(NUM_CLASSES)],
                             dtype=torch.float32).to(DEVICE)
    # Class-weighted CrossEntropy: the single technique that delivered ≥10% MCA
    # improvement for the ISIC 2018 competition winners (focal loss was second).
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_val_loss    = float("inf")
    best_state       = None
    patience_counter = 0
    history          = {"train_loss": [], "val_loss": [], "val_bacc": []}

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
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = running_loss / len(train_loader.dataset)

        # ─ Validate ─
        model.eval()
        val_loss_sum          = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels, _, _ in tqdm(val_loader, desc="  val",
                                           leave=False, unit="batch"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits        = model(imgs)
                val_loss_sum += criterion(logits, labels).item() * imgs.size(0)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss = val_loss_sum / len(val_loader.dataset)
        val_bacc = balanced_accuracy_score(all_labels, all_preds)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_bacc"].append(val_bacc)

        epoch_bar.set_postfix(
            tr_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            bAcc=f"{val_bacc:.4f}"
        )
        print(
            f"  [{arch_key}] Ep {epoch+1:02d}/{MAX_EPOCHS}" f" | train={train_loss:.4f}" f" | val={val_loss:.4f}" f" | bAcc={val_bacc:.4f}",
            flush=True
        )

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                tqdm.write(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, history

# %% [markdown]
# ## B.3 — Train All 4 Models
# Loop over all architectures sequentially. Each model is moved to CPU after
# training to free VRAM for the next one. Checkpoints saved to `outputs/`.

# %%
trained_models  = {}   # arch_key → model (on CPU)
train_histories = {}   # arch_key → {train_loss, val_loss, val_bacc}

for arch_key in ARCHITECTURES:
    model, history = train_one_model(arch_key, train_loader, val_loader)

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
colors = {"train_loss": "#e05c5c", "val_loss": "#5c9be0", "val_bacc": "#5cbf7a"}

for ax, (arch_key, hist) in zip(axes.flat, train_histories.items()):
    epochs = range(1, len(hist["train_loss"]) + 1)
    ax2 = ax.twinx()

    ax.plot(epochs, hist["train_loss"], color=colors["train_loss"],
            lw=1.8, label="Train loss")
    ax.plot(epochs, hist["val_loss"],   color=colors["val_loss"],
            lw=1.8, linestyle="--", label="Val loss")
    ax2.plot(epochs, hist["val_bacc"],  color=colors["val_bacc"],
             lw=1.5, linestyle=":", label="Val bAcc")

    ax.set_title(arch_key, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax2.set_ylabel("Balanced Accuracy", color=colors["val_bacc"])
    ax2.tick_params(axis="y", labelcolor=colors["val_bacc"])
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
            logits = model(imgs)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(torch.softmax(logits, 1).cpu().numpy())

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
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
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

# Plain-text summary report
import datetime
report_path = os.path.join(OUT_ROOT, "evaluation_report.txt")
with open(report_path, "w") as _f:
    _f.write("ISIC2018 XAI Evaluation Report")
    _f.write(f"Generated : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}")
    _f.write(f"DEBUG      : {DEBUG}")
    _f.write("=" * 60 + "")
    for line in summary_lines:
        _f.write(line + "")
    _f.write("" + "=" * 60 + "")
    for arch_key, res in test_results.items():
        _f.write(f"{arch_key}")
        _f.write(classification_report(res["labels"], res["preds"],
                                       target_names=CLASS_NAMES, zero_division=0))
print("" + "=" * 60)
print("Summary:")
for line in summary_lines:
    print(" ", line)
print(f"Report saved to {report_path}")

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
# Weights are set equal by default; tune manually on the validation set
# following the ISIC 2018 winners' approach (∑wᵢ = 1).
#
# FinalScore = Σ wᵢ · sᵢ   where sᵢ is the 7-dim softmax vector for model i.

# %%
ENSEMBLE_WEIGHTS = {
    "efficientnet_b0": 0.25,
    "densenet121":     0.25,
    "vit_base_16":     0.25,
    "swin_tiny":       0.25,
}
assert abs(sum(ENSEMBLE_WEIGHTS.values()) - 1.0) < 1e-6, "Ensemble weights must sum to 1"

# Stack per-model probability arrays (N_test × NUM_CLASSES) weighted sum
ensemble_probs = sum(
    ENSEMBLE_WEIGHTS[k] * test_results[k]["probs"]
    for k in ARCHITECTURES
)
ensemble_preds  = ensemble_probs.argmax(axis=1)
ensemble_labels = test_results["efficientnet_b0"]["labels"]   # same for all models

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
        "weights": ENSEMBLE_WEIGHTS,
        "balanced_accuracy": round(float(ensemble_bacc), 4),
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
