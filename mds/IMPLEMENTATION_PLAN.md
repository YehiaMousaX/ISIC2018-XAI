# Implementation Plan — Multi-Dimensional XAI Evaluation Pipeline

> **Thesis:** A Multi-Dimensional Evaluation of Explainability Methods Across CNN and Vision Transformer Architectures for Skin Lesion Classification
>
> **Target:** Single Jupyter notebook (`XAI_Evaluation_Pipeline_Kaggle.ipynb`)
>
> **Workflow:** Local development → Kaggle execution (mirrors GlaS pipeline pattern)

---

## 1. High-Level Roadmap

| Phase | Thesis Section | Implementation | Est. Cells | Depends On |
|-------|---------------|----------------|------------|------------|
| **A** | §5.1 Dataset | Data loading, EDA, preprocessing | 7–8 | — |
| **B** | §5.2 Models | Define & fine-tune 4 architectures | 10–12 | A |
| **C** | §5.3 XAI Methods | Implement 6 explanation methods | 8–10 | B |
| **D** | §5.4.1 Faithfulness | AOPC, Insertion/Deletion AUC | 6–8 | C |
| **E** | §5.4.2 Plausibility | IoU, Dice, Pointing Game | 4–6 | C |
| **F** | §5.4.3–4 Robustness & Complexity | Max-Sensitivity, Entropy | 4–6 | C |
| **G** | §7 Analysis | Comparison matrix, radar charts, statistics | 6–8 | D+E+F |
| **H** | §7.3 Ablations | Threshold sweep, baseline sensitivity, per-class | 4–6 | G |
| **I** | — | Save outputs, Kaggle adaptation, final report | 3–4 | H |

**Total: ~52–68 cells** (markdown + code interleaved)

---

## 2. Dataset Structure

The dataset is pre-organised — splits, labels, and masks are already provided. No splitting logic needed.

```
Data/
├── csv/
│   ├── train.csv              # 10,015 rows | cols: image, MEL, NV, BCC, AKIEC, BKL, DF, VASC (one-hot)
│   ├── val.csv                #    193 rows | same cols (official validation set)
│   ├── test.csv               #  1,512 rows | same cols (official test set)
│   └── lesion_groupings.csv   # cols: image, lesion_id, diagnosis_confirm_type
│                              # → lesion_id enables patient-level leakage analysis
│
├── images/
│   ├── train/                 # 10,015 ISIC_xxxxxxx.jpg
│   ├── val/                   #    193 ISIC_xxxxxxx.jpg
│   └── test/                  #  1,512 ISIC_xxxxxxx.jpg
│
└── plausibility/
    ├── masks/                 # 3,694 ISIC_xxxxxxx_segmentation.png  (binary lesion masks)
    ├── images/                # corresponding RGB images for the mask subset
    └── attributes/            # 5 attribute maps per image:
                               #   *_attribute_globules.png
                               #   *_attribute_milia_like_cyst.png
                               #   *_attribute_negative_network.png
                               #   *_attribute_pigment_network.png
                               #   *_attribute_streaks.png
```

### Two disjoint image populations — critical architectural fact

ISIC 2018 contains **two completely separate image sets** with non-overlapping IDs:

| Population | ID range | Location | Labels? | Masks? |
|---|---|---|---|---|
| Task 3 classification | ISIC_0024306 – ISIC_0035528 | `images/train\|val\|test/` | yes (CSV) | **no** |
| Task 1 segmentation | ISIC_0000000 – ISIC_0003693 | `plausibility/images/` | not in CSVs | **yes** |

**Consequence:** `has_mask` on `train_df` / `val_df` / `test_df` is always `False` — by design, not a bug.

Phase E (Plausibility) does **not** draw from `eval_subsets`. It uses a separate `plaus_df` built in Step A.4 from `Data/plausibility/images/` paired with their masks. Each trained model is run on `plaus_df` at inference time to obtain predictions and heatmaps, which are then compared against the ground-truth masks.

### Attribute masks note
The 5 attribute maps (globules, milia-like cysts, negative network, pigment network, streaks) are dermoscopic structural features. These can be used in Phase E for fine-grained plausibility analysis beyond binary lesion boundary IoU.

## 3. Repo Layout

```
ISIC2018/
├── XAI_Evaluation_Pipeline_Kaggle.ipynb   # THE notebook
├── Data/                                  # Dataset (see above)
├── prepared/                              # Pipeline-computed artefacts
│   ├── class_weights.json                 # inverse-frequency weights from train split
│   └── data_stats.json                    # per-channel mean/std from train images
├── outputs/                               # Model checkpoints, eval results, figures
│   └── YYYY-MM-DD_HH-MM/
├── scripts/
│   ├── run.sh
│   ├── push_to_kaggle.sh
│   └── wait_for_kernel.sh
└── IMPLEMENTATION_PLAN.md
```

---

## 3. Detailed Step-by-Step Plan

---

### PHASE A — Data Loading, EDA & Preprocessing

> Maps to: **Thesis §5.1 (Dataset)**

---

#### Step A.1 — Environment & Configuration

**Goal:** Single config cell controlling all hyperparameters and paths. Auto-switches between local and Kaggle.

```python
import os, json, warnings, random
import numpy as np
import torch

warnings.filterwarnings("ignore")

DEBUG  = True   # True → 200-image DEBUG subset, 3 epochs
KAGGLE = "KAGGLE_URL_BASE" in os.environ
SEED   = 42

if KAGGLE:
    DATA_ROOT = "/kaggle/input/isic2018"
    PREP_ROOT = "/kaggle/working/prepared"
    OUT_ROOT  = "/kaggle/working"
else:
    DATA_ROOT = "./Data"
    PREP_ROOT = "./prepared"
    OUT_ROOT  = "./outputs"

CSV_DIR      = os.path.join(DATA_ROOT, "csv")
TRAIN_IMG    = os.path.join(DATA_ROOT, "images", "train")
VAL_IMG      = os.path.join(DATA_ROOT, "images", "val")
TEST_IMG     = os.path.join(DATA_ROOT, "images", "test")
MASK_DIR     = os.path.join(DATA_ROOT, "plausibility", "masks")
ATTR_DIR     = os.path.join(DATA_ROOT, "plausibility", "attributes")

IMG_SIZE     = 224
BATCH_SIZE   = 32
MAX_EPOCHS   = 50  if not DEBUG else 3
PATIENCE     = 7   if not DEBUG else 2
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES  = 7
NUM_WORKERS  = 4   if KAGGLE else 0

LIME_SAMPLES        = 1000 if not DEBUG else 100
SHAP_SAMPLES        = 1000 if not DEBUG else 100
SENSITIVITY_N       = 50   if not DEBUG else 5
SENSITIVITY_STD     = 0.01
BINARIZE_THRESHOLDS = [0.3, 0.5, 0.7]
AOPC_STEPS          = 9

ARCHITECTURES = {
    "efficientnet_b0": {"family": "cnn", "timm_name": "efficientnet_b0"},
    "densenet121":     {"family": "cnn", "timm_name": "densenet121"},
    "vit_base_16":     {"family": "vit", "timm_name": "vit_base_patch16_224"},
    "swin_tiny":       {"family": "vit", "timm_name": "swin_tiny_patch4_window7_224"},
}

ATTR_TYPES = ["globules", "milia_like_cyst", "negative_network", "pigment_network", "streaks"]

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything()
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(PREP_ROOT, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE} | DEBUG: {DEBUG} | KAGGLE: {KAGGLE}")
```

**Validation:** Prints device, DEBUG flag, KAGGLE flag. No errors.

---

#### Step A.2 — Install Dependencies

**Goal:** Install packages not in Kaggle's default image.

```python
%%capture
import subprocess, sys
def pip_install(*pkgs):
    if not KAGGLE: return   # locally, manage your own env
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

# captum excluded: v0.8.0 pins numpy<2.0, breaking scipy/albumentations.
# Integrated Gradients implemented manually in Phase C (pure PyTorch).
pip_install("timm", "albumentations", "pytorch-grad-cam",
            "shap", "lime", "seaborn", "statsmodels")

import timm, albumentations
from pytorch_grad_cam import GradCAM, HiResCAM
print(f"timm={timm.__version__}, albumentations={albumentations.__version__}")
```

**Validation:** No import errors. Versions printed.

---

#### Step A.3 — Load Labels & Lesion Groupings

**Goal:** Load pre-split CSVs from `Data/csv/`. Convert one-hot to `label_idx` / `label_name`. Attach `lesion_id` from `lesion_groupings.csv` for patient-level leakage awareness.

```python
import pandas as pd
from pathlib import Path

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

def load_split(csv_path, img_dir):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"image": "image_id"})
    df["label_idx"]  = df[CLASS_NAMES].values.argmax(axis=1)
    df["label_name"] = df["label_idx"].map(lambda i: CLASS_NAMES[i])
    df["img_path"]   = df["image_id"].map(lambda x: os.path.join(img_dir, f"{x}.jpg"))
    return df[["image_id", "label_idx", "label_name", "img_path"]]

train_df = load_split(os.path.join(CSV_DIR, "train.csv"), TRAIN_IMG)
val_df   = load_split(os.path.join(CSV_DIR, "val.csv"),   VAL_IMG)
test_df  = load_split(os.path.join(CSV_DIR, "test.csv"),  TEST_IMG)

# Attach lesion_id and diagnosis_confirm_type — used to track leakage risk and
# label-quality sensitivity analysis (see EDA §C and §J).
# ~26% of lesions have multiple images; ~47% of train labels are non-histopathology.
groupings = pd.read_csv(os.path.join(CSV_DIR, "lesion_groupings.csv"))
groupings  = groupings.rename(columns={"image": "image_id"})
# Keep both columns: lesion_id for leakage tracking, diagnosis_confirm_type for Phase H.3
train_df   = train_df.merge(
    groupings[["image_id", "lesion_id", "diagnosis_confirm_type"]],
    on="image_id", how="left"
)
# Also attach confirm type to test_df for per-image sensitivity analysis in Phase H.3
test_df    = test_df.merge(
    groupings[["image_id", "diagnosis_confirm_type"]],
    on="image_id", how="left"
)

print(f"Train: {len(train_df):>5} | Val: {len(val_df):>3} | Test: {len(test_df):>4}")
print(f"\nTrain label dist:\n{train_df['label_name'].value_counts()}")
print(f"\nConfirmation type dist (train):\n{train_df['diagnosis_confirm_type'].value_counts()}")
```

**Validation:** Train 10,015 / Val 193 / Test 1,512. 7 classes, NV dominates. `diagnosis_confirm_type` shows ~53% histopathology, ~37% serial imaging, ~9% consensus.

---

#### Step A.4 — Build Plausibility Index

**Goal:** Index masks and attribute maps from `Data/plausibility/`. Build `plaus_df` — the dedicated DataFrame used by Phase E. Do **not** attach `has_mask` to train/val/test (those splits have no masks).

```python
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

# Build plaus_df: one row per plausibility image that has a mask on disk.
# Labels are looked up from train/val/test CSVs where available; "unknown" otherwise.
_all_labels = pd.concat([
    train_df[["image_id","label_idx","label_name"]],
    val_df  [["image_id","label_idx","label_name"]],
    test_df [["image_id","label_idx","label_name"]],
], ignore_index=True).drop_duplicates("image_id")

plaus_records = []
for img_id, mask_path in mask_index.items():
    img_path = os.path.join(PLAUS_IMG_DIR, f"{img_id}.jpg")
    if not os.path.exists(img_path):
        continue
    row = {"image_id": img_id, "img_path": img_path, "mask_path": mask_path}
    match = _all_labels[_all_labels["image_id"] == img_id]
    row["label_idx"]  = int(match.iloc[0]["label_idx"]) if len(match) else -1
    row["label_name"] = match.iloc[0]["label_name"]     if len(match) else "unknown"
    plaus_records.append(row)
plaus_df = pd.DataFrame(plaus_records)

# Lesion area fraction & size group — computed on plaus_df only.
# EDA finding: 19.1% of masked images have lesion area < 5% of image area.
# A correctly centred heatmap on a tiny lesion scores near-zero IoU regardless of
# method quality — metric artefact, not XAI failure. Phase E stratifies by size group.
def compute_lesion_area_fraction(mask_path, target_size=(IMG_SIZE, IMG_SIZE)):
    if not mask_path or not os.path.exists(mask_path):
        return float("nan")
    mask = np.array(Image.open(mask_path).convert("L").resize(target_size))
    return (mask > 127).sum() / (target_size[0] * target_size[1])

plaus_df["lesion_area_frac"]  = plaus_df["mask_path"].map(compute_lesion_area_fraction)
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
print("NOTE: train/val/test splits share NO image IDs with plaus_df.")
print("      Phase E runs exclusively on plaus_df.")
```

**Validation:** `plaus_df` has ~3,694 rows. `train_df`/`val_df`/`test_df` have no `has_mask` column (intentional). Small-lesion fraction ~19%.

---

#### Step A.5 — Class Weights & Normalisation Stats

**Goal:** Inverse-frequency class weights from train split. Per-channel mean/std sampled from train images. Both saved to `prepared/`.

```python
from collections import Counter
from PIL import Image
from tqdm.auto import tqdm

counts = Counter(train_df["label_idx"].values)
total  = sum(counts.values())
class_weights = {int(k): total / (NUM_CLASSES * v) for k, v in counts.items()}
with open(os.path.join(PREP_ROOT, "class_weights.json"), "w") as f:
    json.dump(class_weights, f, indent=2)

sample_ids   = train_df["image_id"].sample(min(2000, len(train_df)), random_state=SEED)
pixel_sum    = np.zeros(3, dtype=np.float64)
pixel_sq_sum = np.zeros(3, dtype=np.float64)
n_pixels     = 0

for img_id in tqdm(sample_ids, desc="Computing normalisation stats"):
    img = np.array(Image.open(os.path.join(TRAIN_IMG, f"{img_id}.jpg"))
                   .resize((IMG_SIZE, IMG_SIZE))) / 255.0
    flat = img.reshape(-1, 3)
    pixel_sum    += flat.sum(0); pixel_sq_sum += (flat**2).sum(0)
    n_pixels     += flat.shape[0]

mean = pixel_sum / n_pixels
std  = np.sqrt(pixel_sq_sum / n_pixels - mean**2)
data_stats = {"mean": mean.tolist(), "std": std.tolist()}
with open(os.path.join(PREP_ROOT, "data_stats.json"), "w") as f:
    json.dump(data_stats, f, indent=2)
print(f"mean={mean.round(4)}, std={std.round(4)}")
```

**Validation:** Mean ~[0.76, 0.55, 0.57], std ~[0.14, 0.16, 0.18]. NV weight < 1 (majority), DF/VASC > 1 (minority).

---

#### Step A.6 — Dataset Class & DataLoaders

**Goal:** `ISICSkinDataset` loading from per-split `img_dir`. Albumentations augmentations. `WeightedRandomSampler` corrects class imbalance.

```python
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ISICSkinDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, load_masks=False):
        self.df = df.reset_index(drop=True)
        self.img_dir    = img_dir
        self.transform  = transform
        self.load_masks = load_masks

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        img  = np.array(Image.open(os.path.join(self.img_dir,
                         f"{row['image_id']}.jpg")).convert("RGB"))
        mask = None
        if self.load_masks and row.get("has_mask", False):
            mp = row.get("mask_path")
            if mp and os.path.exists(mp):
                mask = np.array(Image.open(mp).convert("L"))

        if self.transform:
            if mask is not None:
                out  = self.transform(image=img, mask=mask)
                img  = out["image"]; mask = out["mask"].float() / 255.0
            else:
                img = self.transform(image=img)["image"]

        label = torch.tensor(row["label_idx"], dtype=torch.long)
        meta  = {"image_id": row["image_id"], "has_mask": bool(row.get("has_mask", False))}
        return img, label, mask if mask is not None else torch.zeros(1), meta

with open(os.path.join(PREP_ROOT, "data_stats.json")) as f: stats = json.load(f)

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Normalize(mean=stats["mean"], std=stats["std"]), ToTensorV2(),
])
eval_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(mean=stats["mean"], std=stats["std"]), ToTensorV2(),
])

train_ds = ISICSkinDataset(train_df, TRAIN_IMG, train_transform)
val_ds   = ISICSkinDataset(val_df,   VAL_IMG,   eval_transform)
test_ds  = ISICSkinDataset(test_df,  TEST_IMG,  eval_transform, load_masks=True)

with open(os.path.join(PREP_ROOT, "class_weights.json")) as f: cw = json.load(f)
sample_weights = [cw[str(int(l))] for l in train_df["label_idx"]]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,   num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  1,          shuffle=False,   num_workers=NUM_WORKERS, pin_memory=True)

imgs, labels, _, _ = next(iter(train_loader))
print(f"Batch: {imgs.shape} | dist: {labels.bincount(minlength=NUM_CLASSES).tolist()}")
```

**Validation:** Batch `[32, 3, 224, 224]`. Label distribution spread across classes (sampler effect).

---

#### Step A.7 — Visual Sanity Check

**Goal:** 8-image training grid + 4 test images with mask overlays.

```python
import matplotlib.pyplot as plt

MN, STD = np.array(stats["mean"]), np.array(stats["std"])
def denorm(t): return np.clip(t.permute(1,2,0).numpy() * STD + MN, 0, 1)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(denorm(imgs[i])); ax.set_title(CLASS_NAMES[labels[i].item()]); ax.axis("off")
plt.suptitle("Training Batch — WeightedRandomSampler"); plt.tight_layout()
plt.savefig(os.path.join(OUT_ROOT, "sanity_train_batch.png"), dpi=100); plt.show()

# Mask overlay on test subset
masked = test_df[test_df["has_mask"]].head(4)
if len(masked):
    fig, axes = plt.subplots(1, len(masked), figsize=(4*len(masked), 4))
    for ax, (_, row) in zip(axes if len(masked) > 1 else [axes], masked.iterrows()):
        raw = np.array(Image.open(os.path.join(TEST_IMG, f"{row['image_id']}.jpg"))
                       .resize((IMG_SIZE, IMG_SIZE))) / 255.0
        msk = np.array(Image.open(row["mask_path"]).resize((IMG_SIZE, IMG_SIZE)))
        ax.imshow(raw); ax.imshow(msk, alpha=0.35, cmap="Reds")
        ax.set_title(f"{row['label_name']}"); ax.axis("off")
    plt.suptitle("Test — Segmentation Mask Overlay"); plt.tight_layout(); plt.show()
```

**Validation:** Dermoscopic images with correct labels. Mask overlays are red regions on lesions.

---

### PHASE B — Model Training

> Maps to: **Thesis §5.2 (Model Architectures)**

---

#### Step B.1 — Model Factory

**Goal:** A function that creates any of the 4 architectures from `timm` with the correct classification head. Returns the model AND the target layer name for Grad-CAM.

```python
import timm

def create_model(arch_key):
    """Create a model from ARCHITECTURES config.
    Returns: (model, target_layer_name_for_gradcam, family)
    """
    cfg = ARCHITECTURES[arch_key]
    model = timm.create_model(cfg["timm_name"], pretrained=True,
                              num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # Identify the target layer for Grad-CAM (last feature-producing layer)
    target_layers = {
        "efficientnet_b0": "conv_head",       # last conv before pooling
        "densenet121":     "features.denseblock4.denselayer16.conv2",
        "vit_base_16":     None,               # not used for Grad-CAM
        "swin_tiny":       None,               # not used for Grad-CAM
    }

    return model, target_layers.get(arch_key), cfg["family"]


# Sanity: instantiate all 4
for name in ARCHITECTURES:
    m, tl, fam = create_model(name)
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"{name:20s} | {fam} | {n_params:.1f}M params | grad-cam layer: {tl}")
    del m
torch.cuda.empty_cache()
```

**Validation:** Four lines printed. EfficientNet ~5.3M, DenseNet ~8.0M, ViT ~86M, Swin ~29M. No CUDA OOM.

---

#### Step B.2 — Training Loop (Single Architecture)

**Goal:** A reusable `train_one_model(arch_key)` function with: **Focal Loss** (replaces weighted CE — see EDA §B/§J), AdamW + cosine annealing, early stopping on val loss, checkpoint saving. Returns the trained model + training history.

> **EDA rationale:** The 58.3× class imbalance (DF vs NV) makes plain weighted CrossEntropyLoss insufficient — it still allows easy NV predictions to dominate the gradient. Focal Loss adds a `(1 − p)^γ` modulating factor that down-weights high-confidence majority predictions, focusing capacity on DF (115 images) and VASC (142 images). This is critical for producing meaningful XAI explanations on minority classes.

```python
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import balanced_accuracy_score
import copy

FOCAL_GAMMA = 2.0  # standard starting value; sweep [1, 2, 3] in ablations if needed

class FocalLoss(nn.Module):
    """Multiclass Focal Loss with per-class weights.
    gamma=0 reduces to weighted CrossEntropyLoss.
    """
    def __init__(self, weight=None, gamma=FOCAL_GAMMA, reduction="mean"):
        super().__init__()
        self.weight    = weight   # class-frequency weights (tensor on DEVICE)
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Standard CE (with class weights)
        ce = nn.functional.cross_entropy(logits, targets,
                                         weight=self.weight,
                                         reduction="none")
        # Probability of the true class
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean() if self.reduction == "mean" else focal


def train_one_model(arch_key, train_loader, val_loader):
    print(f"\n{'='*60}\n  Training: {arch_key}\n{'='*60}")

    model, _, family = create_model(arch_key)

    # ── Focal Loss with class weights ──
    with open(os.path.join(PREP_ROOT, "class_weights.json")) as f:
        cw = json.load(f)
    weights = torch.tensor([cw[str(i)] for i in range(NUM_CLASSES)],
                           dtype=torch.float32).to(DEVICE)
    criterion = FocalLoss(weight=weights, gamma=FOCAL_GAMMA)

    # ── Optimizer & scheduler ──
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # ── Training loop ──
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_bacc": []}

    for epoch in range(MAX_EPOCHS):
        # — Train —
        model.train()
        running_loss = 0.0
        for imgs, labels, _, _ in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # — Validate —
        model.eval()
        val_loss_sum = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels, _, _ in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                val_loss_sum += criterion(logits, labels).item() * imgs.size(0)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss = val_loss_sum / len(val_loader.dataset)
        val_bacc = balanced_accuracy_score(all_labels, all_preds)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_bacc"].append(val_bacc)

        print(f"  Epoch {epoch+1:02d}/{MAX_EPOCHS} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_bAcc={val_bacc:.4f}")

        # — Early stopping on val loss —
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    return model, history
```

**Validation:** Run with `DEBUG=True` (3 epochs). Loss decreases. No NaN. No CUDA OOM.

---

#### Step B.3 — Train All 4 Models Sequentially

**Goal:** Loop over all architectures, train each, save checkpoint, store histories.

```python
trained_models = {}   # arch_key → model (on CPU to save VRAM)
train_histories = {}  # arch_key → history dict

for arch_key in ARCHITECTURES:
    model, history = train_one_model(arch_key, train_loader, val_loader)

    # Save checkpoint
    ckpt_path = os.path.join(OUT_ROOT, f"{arch_key}_best.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved: {ckpt_path}")

    # Move to CPU to free GPU for next model
    trained_models[arch_key] = model.cpu()
    train_histories[arch_key] = history
    torch.cuda.empty_cache()

print("\n✓ All 4 models trained and saved.")
```

**Validation:** 4 checkpoint files created. No OOM (models moved to CPU between runs).

---

#### Step B.4 — Evaluate Classification Performance (Test Set)

**Goal:** For each model, compute accuracy, balanced accuracy, F1 (macro), per-class precision/recall. This is NOT the XAI evaluation — this is the baseline classification performance needed to contextualise the XAI results.

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

test_results = {}

for arch_key, model in trained_models.items():
    model = model.to(DEVICE).eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels, masks, meta in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    report = classification_report(all_labels, all_preds,
                                   target_names=CLASS_NAMES, output_dict=True)
    test_results[arch_key] = {
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
        "report": report,
    }

    print(f"\n{'='*40} {arch_key} {'='*40}")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    model = model.cpu()
    torch.cuda.empty_cache()
```

**Validation:** All 4 models print a classification report. Target: ≥80% balanced accuracy on ≥2 models (to have meaningful explanations to evaluate). If a model performs poorly (<60%), flag it — its explanations may not be meaningful.

---

#### Step B.5 — Identify Correctly Classified Test Images

**Goal:** Per thesis §5.5 — XAI evaluation is conducted on **correctly classified images only**. Build the evaluation subset.

```python
eval_subsets = {}  # arch_key → DataFrame of correctly-classified test images

for arch_key in ARCHITECTURES:
    preds = test_results[arch_key]["preds"]
    labels = test_results[arch_key]["labels"]
    correct_mask = preds == labels
    correct_df = test_df[correct_mask].copy()
    correct_df["pred"] = preds[correct_mask]
    eval_subsets[arch_key] = correct_df

    n_correct = correct_mask.sum()
    print(f"{arch_key}: {n_correct}/{len(test_df)} correct "
          f"({n_correct/len(test_df)*100:.1f}%)"
          f"  [plausibility evaluated separately on plaus_df in Phase E]")
```

**Validation:** Each model's correct subset size printed.
- **Faithfulness, Robustness, Complexity** → `eval_subsets[arch_key]` (correctly classified test images)
- **Plausibility** → `plaus_df` (separate population in `plausibility/images/`; has masks; no overlap with test split)

---

### PHASE C — XAI Method Implementation

> Maps to: **Thesis §5.3 (XAI Methods)**

---

#### Step C.1 — Unified Heatmap Generation Interface

**Goal:** A `generate_heatmap(model, img_tensor, method, arch_key)` function that returns a normalised [0,1] attribution map of shape `(H, W)`, regardless of the method. This is the core abstraction.

```python
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import Lime, KernelShap
from captum.attr._core.lime import get_exp_kernel_similarity_function

def get_gradcam_target_layer(model, arch_key):
    """Return the nn.Module for the last conv/feature layer."""
    if arch_key == "efficientnet_b0":
        return [model.conv_head]
    elif arch_key == "densenet121":
        return [model.features.denseblock4.denselayer16.conv2]
    else:
        raise ValueError(f"No Grad-CAM target for {arch_key}")


def generate_heatmap(model, img_tensor, method, arch_key, target_class=None):
    """
    Generate a normalised [0,1] attribution heatmap.

    Args:
        model: trained model (on DEVICE, eval mode)
        img_tensor: (1, 3, H, W) tensor on DEVICE
        method: one of "gradcam", "hirescam", "lime", "kernelshap",
                "attention_rollout", "attnlrp"
        arch_key: key from ARCHITECTURES
        target_class: int or None (uses predicted class if None)

    Returns:
        heatmap: np.ndarray of shape (H, W), values in [0, 1]
    """
    model.eval()
    family = ARCHITECTURES[arch_key]["family"]

    if target_class is None:
        with torch.no_grad():
            target_class = model(img_tensor).argmax(1).item()

    H, W = img_tensor.shape[2], img_tensor.shape[3]

    # ── CNN-specific methods ──
    if method == "gradcam":
        assert family == "cnn", "Grad-CAM only for CNNs"
        targets = [ClassifierOutputTarget(target_class)]
        cam = GradCAM(model=model,
                      target_layers=get_gradcam_target_layer(model, arch_key))
        heatmap = cam(input_tensor=img_tensor, targets=targets)[0]

    elif method == "hirescam":
        assert family == "cnn", "HiResCAM only for CNNs"
        targets = [ClassifierOutputTarget(target_class)]
        cam = HiResCAM(model=model,
                       target_layers=get_gradcam_target_layer(model, arch_key))
        heatmap = cam(input_tensor=img_tensor, targets=targets)[0]

    # ── Model-agnostic methods ──
    elif method == "lime":
        lime_attr = Lime(model)
        # Captum LIME expects (1, C, H, W)
        attr = lime_attr.attribute(
            img_tensor, target=target_class,
            n_samples=LIME_SAMPLES,
            perturbations_per_eval=4,
            show_progress=False
        )
        heatmap = attr.squeeze().abs().mean(dim=0).cpu().numpy()  # avg channels

    elif method == "kernelshap":
        ks = KernelShap(model)
        attr = ks.attribute(
            img_tensor, target=target_class,
            n_samples=SHAP_SAMPLES,
            show_progress=False
        )
        heatmap = attr.squeeze().abs().mean(dim=0).cpu().numpy()

    # ── Transformer-specific methods ──
    elif method == "attention_rollout":
        assert family == "vit", "Attention Rollout only for ViTs"
        heatmap = _attention_rollout(model, img_tensor, arch_key)

    elif method == "attnlrp":
        assert family == "vit", "AttnLRP only for ViTs"
        heatmap = _transformer_attribution(model, img_tensor,
                                            target_class, arch_key)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalise to [0, 1]
    heatmap = heatmap.astype(np.float32)
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap = np.zeros_like(heatmap)

    # Resize to input dimensions if needed
    if heatmap.shape != (H, W):
        from skimage.transform import resize
        heatmap = resize(heatmap, (H, W), order=1, anti_aliasing=True)

    return heatmap
```

**Validation:** Deferred to Step C.4 (visual check on one image).

---

#### Step C.2 — Attention Rollout Implementation

**Goal:** Implement `_attention_rollout()` for ViT-Base and Swin-T.

```python
def _attention_rollout(model, img_tensor, arch_key):
    """
    Compute Attention Rollout for a ViT.
    Returns heatmap of shape (H, W).
    """
    model.eval()

    # Hook to capture attention weights
    attention_maps = []
    hooks = []

    def hook_fn(module, input, output):
        # timm ViTs return attention from the Attention module
        # For vit_base: model.blocks[i].attn → output is (B, num_heads, N, N)
        if hasattr(module, "attn_drop"):
            # Recompute attention from qkv
            B, N, C = input[0].shape
            qkv = module.qkv(input[0]).reshape(B, N, 3, module.num_heads,
                                                 C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            attention_maps.append(attn.detach().cpu().numpy())

    # Register hooks on all attention modules
    if arch_key == "vit_base_16":
        for block in model.blocks:
            hooks.append(block.attn.register_forward_hook(hook_fn))
    elif arch_key == "swin_tiny":
        # Swin has a different structure — use a simpler approach
        # Fall back to gradient-based attention for Swin
        return _swin_attention_rollout(model, img_tensor)

    with torch.no_grad():
        _ = model(img_tensor)

    for h in hooks:
        h.remove()

    if not attention_maps:
        return np.zeros((IMG_SIZE, IMG_SIZE))

    # Rollout: multiply attention matrices, add identity (residual)
    result = None
    for attn in attention_maps:
        # Average over heads: (B, num_heads, N, N) → (B, N, N)
        attn_avg = attn.mean(axis=1)[0]  # (N, N)
        attn_avg = attn_avg + np.eye(attn_avg.shape[0])  # residual
        attn_avg = attn_avg / attn_avg.sum(axis=-1, keepdims=True)
        if result is None:
            result = attn_avg
        else:
            result = result @ attn_avg

    # CLS token row → discard CLS position → reshape to grid
    cls_attention = result[0, 1:]  # exclude CLS token itself
    num_patches = int(np.sqrt(cls_attention.shape[0]))
    heatmap = cls_attention.reshape(num_patches, num_patches)
    return heatmap


def _swin_attention_rollout(model, img_tensor):
    """Simplified attention extraction for Swin Transformer via gradient."""
    # Swin's shifted-window attention is harder to roll up.
    # Use Captum's Layer GradCAM on the last norm layer as a pragmatic fallback.
    from captum.attr import LayerGradCam
    lgc = LayerGradCam(model, model.layers[-1].blocks[-1].norm1)
    with torch.no_grad():
        target = model(img_tensor).argmax(1).item()
    attr = lgc.attribute(img_tensor, target=target)
    heatmap = attr.squeeze().abs().mean(dim=0).cpu().numpy()
    return heatmap
```

**Validation:** Deferred to C.4.

---

#### Step C.3 — Transformer Attribution (AttnLRP) Implementation

**Goal:** Implement class-specific transformer attribution using the Chefer et al. (2021) method.

```python
def _transformer_attribution(model, img_tensor, target_class, arch_key):
    """
    Chefer et al. (2021) Transformer Attribution.
    Combines attention with gradient-based relevance.
    Returns heatmap of shape (H, W).
    """
    model.eval()
    model.zero_grad()

    # Enable gradient computation
    img_tensor = img_tensor.clone().requires_grad_(True)

    # Forward pass
    output = model(img_tensor)
    target_score = output[0, target_class]

    # Collect attention maps with gradients
    attention_maps = []
    gradients = []

    # Hook into attention modules
    hooks = []
    if arch_key == "vit_base_16":
        for block in model.blocks:
            # Store attention weights AND register gradient hook
            attn_weights = []

            def make_hook(storage):
                def hook_fn(module, input, output):
                    B, N, C = input[0].shape
                    qkv = module.qkv(input[0]).reshape(
                        B, N, 3, module.num_heads, C // module.num_heads
                    ).permute(2, 0, 3, 1, 4)
                    q, k, _ = qkv.unbind(0)
                    attn = (q @ k.transpose(-2, -1)) * module.scale
                    attn = attn.softmax(dim=-1)
                    attn.retain_grad()
                    storage.append(attn)
                return hook_fn

            storage = []
            hooks.append(block.attn.register_forward_hook(make_hook(storage)))
            attention_maps.append(storage)

    elif arch_key == "swin_tiny":
        # Use same gradient-based fallback as rollout
        for h in hooks:
            h.remove()
        return _swin_attention_rollout(model, img_tensor)

    # Forward again with hooks
    model.zero_grad()
    img_tensor_grad = img_tensor.clone().detach().requires_grad_(True)
    output = model(img_tensor_grad)
    target_score = output[0, target_class]
    target_score.backward(retain_graph=True)

    for h in hooks:
        h.remove()

    # Combine attention × gradient (Chefer method)
    result = None
    for storage in attention_maps:
        if not storage:
            continue
        attn = storage[0]  # (B, heads, N, N)
        grad = attn.grad
        if grad is None:
            continue
        # Element-wise: attention * gradient, clamp negatives, avg heads
        cam = (attn * grad.clamp(min=0)).mean(dim=1)[0].detach().cpu().numpy()
        cam = cam + np.eye(cam.shape[0])
        cam = cam / cam.sum(axis=-1, keepdims=True)
        result = cam if result is None else result @ cam

    if result is None:
        return np.zeros((IMG_SIZE, IMG_SIZE))

    cls_attention = result[0, 1:]
    num_patches = int(np.sqrt(cls_attention.shape[0]))
    heatmap = cls_attention.reshape(num_patches, num_patches)
    return heatmap
```

**Validation:** Deferred to C.4.

---

#### Step C.4 — Visual Validation of All XAI Methods

**Goal:** Pick one test image per architecture, generate heatmaps from all applicable methods, display as a grid. This is the critical sanity check before metric computation.

```python
def get_applicable_methods(arch_key):
    family = ARCHITECTURES[arch_key]["family"]
    if family == "cnn":
        return ["gradcam", "hirescam", "lime", "kernelshap"]
    else:
        return ["lime", "kernelshap", "attention_rollout", "attnlrp"]


# Pick one image from test set
sample_idx = 0
sample = test_ds[sample_idx]
img_tensor = sample[0].unsqueeze(0).to(DEVICE)
label = sample[1].item()

fig, axes = plt.subplots(len(ARCHITECTURES), 5, figsize=(20, 16))

for row, arch_key in enumerate(ARCHITECTURES):
    model = trained_models[arch_key].to(DEVICE).eval()
    methods = get_applicable_methods(arch_key)

    # Show original image in first column
    img_display = img_tensor[0].cpu().permute(1, 2, 0).numpy()
    img_display = img_display * np.array(stats["std"]) + np.array(stats["mean"])
    img_display = np.clip(img_display, 0, 1)
    axes[row, 0].imshow(img_display)
    axes[row, 0].set_title(f"{arch_key}\nTrue: {CLASS_NAMES[label]}", fontsize=9)
    axes[row, 0].axis("off")

    for col, method in enumerate(methods):
        heatmap = generate_heatmap(model, img_tensor, method, arch_key)
        axes[row, col+1].imshow(img_display)
        axes[row, col+1].imshow(heatmap, cmap="jet", alpha=0.4)
        axes[row, col+1].set_title(method, fontsize=9)
        axes[row, col+1].axis("off")

    model = model.cpu()
    torch.cuda.empty_cache()

plt.suptitle("XAI Method Visual Validation", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_ROOT, "xai_visual_validation.png"), dpi=150,
            bbox_inches="tight")
plt.show()
```

**Validation (CRITICAL — must pass before proceeding):**
- All cells render without error
- Heatmaps are not all-zero or all-one
- Grad-CAM / HiResCAM highlight the lesion area (not background artefacts)
- Attention Rollout shows a coherent spatial pattern (not random noise)
- LIME produces superpixel-granularity highlights

---

### PHASE D — Faithfulness Metrics

> Maps to: **Thesis §5.4.1 (Dimension 1)**

---

#### Step D.1 — AOPC (Area Over the Perturbation Curve)

**Goal:** Implement AOPC. Progressively mask top-k% of pixels (by attribution rank), measure drop in predicted class probability.

```python
def compute_aopc(model, img_tensor, heatmap, target_class, steps=AOPC_STEPS):
    """
    Compute AOPC: average probability drop when masking top-k% pixels.

    Returns:
        aopc: float (higher = more faithful)
        curve: list of (fraction_masked, prob) tuples
    """
    model.eval()
    H, W = heatmap.shape
    total_pixels = H * W

    # Baseline: mean pixel value of training set
    baseline = torch.tensor(stats["mean"], dtype=torch.float32).to(DEVICE)
    baseline = (baseline - torch.tensor(stats["mean"]).to(DEVICE)) / \
               torch.tensor(stats["std"]).to(DEVICE)  # normalised baseline

    # Original prediction probability
    with torch.no_grad():
        orig_prob = torch.softmax(model(img_tensor), dim=1)[0, target_class].item()

    # Rank pixels by attribution (descending)
    flat_heatmap = heatmap.flatten()
    sorted_indices = np.argsort(flat_heatmap)[::-1]  # highest first

    curve = [(0.0, orig_prob)]
    drops = []

    for step in range(1, steps + 1):
        frac = step / (steps + 1)
        n_mask = int(frac * total_pixels)
        mask_indices = sorted_indices[:n_mask]

        # Create masked image
        masked_img = img_tensor.clone()
        rows = mask_indices // W
        cols = mask_indices % W
        for c in range(3):
            masked_img[0, c, rows, cols] = baseline[c]

        with torch.no_grad():
            new_prob = torch.softmax(model(masked_img), dim=1)[0, target_class].item()

        curve.append((frac, new_prob))
        drops.append(orig_prob - new_prob)

    aopc = np.mean(drops)
    return aopc, curve
```

**Validation:** Run on one image. AOPC > 0 (probability should drop when masking important pixels). Plot the curve — it should decrease monotonically.

---

#### Step D.2 — Insertion & Deletion AUC

**Goal:** Implement Insertion (reveal pixels by importance onto blurred canvas) and Deletion (remove pixels by importance from original).

```python
from scipy.ndimage import gaussian_filter

def compute_insertion_deletion(model, img_tensor, heatmap, target_class,
                                n_steps=100):
    """
    Compute Insertion and Deletion AUC.

    Returns:
        insertion_auc: float (higher = better)
        deletion_auc: float (lower = better)
    """
    model.eval()
    H, W = heatmap.shape
    total_pixels = H * W

    # Blurred baseline for insertion
    blurred = img_tensor.clone()
    for c in range(3):
        blurred[0, c] = torch.tensor(
            gaussian_filter(img_tensor[0, c].cpu().numpy(), sigma=5),
            dtype=torch.float32
        ).to(DEVICE)

    # Rank pixels
    sorted_indices = np.argsort(heatmap.flatten())[::-1]

    insertion_probs = []
    deletion_probs = []
    step_size = max(total_pixels // n_steps, 1)

    ins_img = blurred.clone()
    del_img = img_tensor.clone()

    for i in range(0, total_pixels, step_size):
        idx = sorted_indices[max(0, i - step_size):i]
        rows = idx // W
        cols = idx % W

        # Insertion: reveal original pixels
        for c in range(3):
            ins_img[0, c, rows, cols] = img_tensor[0, c, rows, cols]
        # Deletion: replace with blurred
        for c in range(3):
            del_img[0, c, rows, cols] = blurred[0, c, rows, cols]

        with torch.no_grad():
            ins_prob = torch.softmax(model(ins_img), dim=1)[0, target_class].item()
            del_prob = torch.softmax(model(del_img), dim=1)[0, target_class].item()

        insertion_probs.append(ins_prob)
        deletion_probs.append(del_prob)

    # AUC via trapezoidal rule
    x = np.linspace(0, 1, len(insertion_probs))
    insertion_auc = np.trapz(insertion_probs, x)
    deletion_auc = np.trapz(deletion_probs, x)

    return insertion_auc, deletion_auc
```

**Validation:** Run on one image. Insertion AUC should be ∈ [0.3, 0.9]. Deletion AUC should be lower than insertion AUC.

---

### PHASE E — Plausibility Metrics

> Maps to: **Thesis §5.4.2 (Dimension 2)**

> **Data source:** Phase E runs exclusively on `plaus_df` — the 3,694 images in `Data/plausibility/images/` paired with their ground-truth segmentation masks. These images have different IDs from the train/val/test classification splits and are never seen by the model during training.
>
> **No classification label is needed.** Plausibility evaluation asks: *"Does the XAI heatmap highlight the lesion region?"* — not *"Did the model predict the right class?"* The workflow is:
> 1. Run model on plausibility image → get **predicted class** (whatever the model commits to)
> 2. Generate XAI heatmap **for that predicted class**
> 3. Measure overlap of heatmap with **ground-truth mask** via IoU / Dice / Pointing Game
>
> The mask defines *where* the lesion is. A good XAI method should focus on that region regardless of which disease the model predicted. This is standard practice in the XAI literature (GradCAM, RISE, etc.) and does not require a correct or even known classification label.
>
> **EDA note (§G):** 19.1% of masked images have a lesion covering < 5% of the image. For tiny lesions any spatial offset in the heatmap produces near-zero IoU regardless of method quality — this is a metric property, not an XAI failure. All plausibility results **must** be stratified by `lesion_size_group` ("small" < 5%, "standard" ≥ 5%) and reported separately before pooling. The `lesion_size_group` column is computed on `plaus_df` in Step A.4.
>
> **L2 attribute coverage note (§G):** Pigment network (n = 1,950, 52.8% non-empty) is the only attribute with sufficient N for reliable per-class breakdowns. Streaks (n = 152) and negative network (n = 244) should be treated as exploratory with confidence intervals, or aggregated into an "other" group.

---

#### Step E.1 — IoU, Dice, and Pointing Game

**Goal:** Compare binarised heatmap against ground-truth segmentation mask. Results are stored per-image (with `lesion_size_group`) so that the analysis in G.2 can stratify automatically.

```python
def compute_plausibility(heatmap, gt_mask, thresholds=BINARIZE_THRESHOLDS):
    """
    Compute IoU, Dice, Pointing Game at multiple thresholds.

    Args:
        heatmap: np.ndarray (H, W), values in [0, 1]
        gt_mask: np.ndarray (H, W), binary 0/1

    Returns:
        dict with keys like 'iou_0.3', 'dice_0.5', 'pointing_game'
    """
    results = {}

    # Resize gt_mask to heatmap size if needed
    if gt_mask.shape != heatmap.shape:
        from skimage.transform import resize
        gt_mask = resize(gt_mask, heatmap.shape, order=0,
                         anti_aliasing=False) > 0.5
    gt_binary = gt_mask.astype(bool)

    for t in thresholds:
        pred_binary = heatmap >= t

        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        pred_sum = pred_binary.sum()
        gt_sum = gt_binary.sum()

        iou = intersection / union if union > 0 else 0.0
        dice = (2 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0

        results[f"iou_{t}"] = iou
        results[f"dice_{t}"] = dice

    # Pointing Game: does the max-attribution pixel fall inside the mask?
    max_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    results["pointing_game"] = float(gt_binary[max_idx[0], max_idx[1]])

    return results
```

**Validation:** Run on one image that has a mask. IoU should be > 0 (not zero unless heatmap misses the lesion entirely). Pointing game = 0 or 1.

---

### PHASE F — Robustness & Complexity Metrics

> Maps to: **Thesis §5.4.3–4 (Dimensions 3 & 4)**

---

#### Step F.1 — Max-Sensitivity

**Goal:** Perturb input N times with small Gaussian noise, measure maximum L2 change in the heatmap.

```python
def compute_max_sensitivity(model, img_tensor, method, arch_key,
                             target_class, original_heatmap,
                             n_perturbations=SENSITIVITY_N,
                             sigma=SENSITIVITY_STD):
    """
    Max-Sensitivity: max L2 distance between original and perturbed heatmaps.
    Lower = more robust.
    """
    max_l2 = 0.0

    for _ in range(n_perturbations):
        noise = torch.randn_like(img_tensor) * sigma
        perturbed = img_tensor + noise

        perturbed_heatmap = generate_heatmap(model, perturbed, method,
                                              arch_key, target_class)
        l2 = np.linalg.norm(original_heatmap - perturbed_heatmap)
        max_l2 = max(max_l2, l2)

    return max_l2
```

---

#### Step F.2 — Explanation Entropy

**Goal:** Shannon entropy of the normalised attribution map.

```python
def compute_entropy(heatmap):
    """Shannon entropy of attribution map. Lower = more focused."""
    # Normalise to probability distribution
    flat = heatmap.flatten()
    flat = flat / (flat.sum() + 1e-10)
    flat = flat[flat > 0]  # avoid log(0)
    entropy = -np.sum(flat * np.log2(flat))
    return entropy
```

**Validation:** Entropy of a uniform heatmap ≈ log2(224*224) ≈ 15.6. A focused heatmap should have entropy << 15.

---

### PHASE G — Full Evaluation & Analysis

> Maps to: **Thesis §7 (Experimental Design and Analysis Plan)**

---

#### Step G.1 — Run Full Evaluation Loop

**Goal:** Two separate loops over the same (model × method) grid:

1. **Faithfulness + Robustness + Complexity** — run on `eval_subsets[arch_key]` (correctly classified test images).
2. **Plausibility** — run on `plaus_df` (the 3,694 images in `plausibility/images/` with ground-truth masks).

Results are merged by `(arch, method, image_id)` into a single `results_df`.

```python
from tqdm.auto import tqdm

faith_results = []   # faithfulness / robustness / complexity rows
plaus_results = []   # plausibility rows

for arch_key in ARCHITECTURES:
    model = trained_models[arch_key].to(DEVICE).eval()
    methods = get_applicable_methods(arch_key)

    # ── Loop 1: Faithfulness / Robustness / Complexity ─────────────────────
    eval_df = eval_subsets[arch_key]
    if DEBUG:
        eval_df = eval_df.head(20)

    for method in methods:
        print(f"\n▶ [Faith/Rob/Cplx] {arch_key} × {method} ({len(eval_df)} images)")
        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
            img = apply_color_constancy(np.array(
                Image.open(os.path.join(TEST_IMG, f"{row['image_id']}.jpg")).convert("RGB")
            ))
            img_tensor = eval_transform(image=img)["image"].unsqueeze(0).to(DEVICE)
            target_class = int(row["pred"])

            heatmap = generate_heatmap(model, img_tensor, method, arch_key, target_class)

            aopc, _ = compute_aopc(model, img_tensor, heatmap, target_class)
            ins_auc, del_auc = compute_insertion_deletion(
                model, img_tensor, heatmap, target_class
            )
            max_sens = compute_max_sensitivity(
                model, img_tensor, method, arch_key, target_class, heatmap
            )
            faith_results.append({
                "arch": arch_key, "method": method,
                "image_id": row["image_id"],
                "label": row["label_idx"], "label_name": row["label_name"],
                "confirm_type": row.get("diagnosis_confirm_type", np.nan),
                "aopc": aopc, "insertion_auc": ins_auc, "deletion_auc": del_auc,
                "max_sensitivity": max_sens,
                "entropy": compute_entropy(heatmap),
            })

    # ── Loop 2: Plausibility ────────────────────────────────────────────────
    plaus_eval = plaus_df.copy()
    if DEBUG:
        plaus_eval = plaus_eval.head(20)

    for method in methods:
        print(f"\n▶ [Plausibility]    {arch_key} × {method} ({len(plaus_eval)} images)")
        for _, row in tqdm(plaus_eval.iterrows(), total=len(plaus_eval)):
            img = apply_color_constancy(np.array(
                Image.open(row["img_path"]).convert("RGB")
            ))
            img_tensor = eval_transform(image=img)["image"].unsqueeze(0).to(DEVICE)
            # Use model's predicted class (not ground-truth label) for the heatmap
            with torch.no_grad():
                target_class = model(img_tensor).argmax(1).item()

            heatmap = generate_heatmap(model, img_tensor, method, arch_key, target_class)

            gt_mask = (np.array(Image.open(row["mask_path"]).convert("L")) > 127).astype(np.float32)
            plaus = compute_plausibility(heatmap, gt_mask)
            plaus_results.append({
                "arch": arch_key, "method": method,
                "image_id": row["image_id"],
                "label": row["label_idx"], "label_name": row["label_name"],
                "lesion_size_group": row["lesion_size_group"],
                **plaus,
            })

    model = model.cpu()
    torch.cuda.empty_cache()

faith_df = pd.DataFrame(faith_results)
plaus_df_results = pd.DataFrame(plaus_results)

# Merge on (arch, method, image_id) — NaN where an image appears in only one loop
results_df = faith_df.merge(
    plaus_df_results, on=["arch", "method", "image_id", "label", "label_name"],
    how="outer"
)
results_df.to_csv(os.path.join(OUT_ROOT, "full_results.csv"), index=False)
print(f"\n✓ Full results: {results_df.shape}")
print(results_df.head())
```

**Validation:** `faith_df` has no NaN in `aopc`/`insertion_auc`/`deletion_auc`. `plaus_df_results` has no NaN in `iou_*`/`dice_*`/`pointing_game`. After merge, plausibility columns are NaN for test-set rows (expected).

**⚠️ Runtime note:** This is the longest step. Full run (~1,500 eval images + ~3,700 plaus images, × 16 method-cells) takes ~8–12 hours on a T4. Plan accordingly.

---

#### Step G.2 — Summary Statistics Table

**Goal:** Aggregate results into the comparison matrix (mean ± std per cell per metric). Plausibility metrics are also reported **stratified by lesion size** (EDA §G — 19.1% small lesions would otherwise suppress IoU/Dice scores across all methods equally).

```python
# Group by (architecture, method) → compute mean ± std for each metric
metric_cols = ["aopc", "insertion_auc", "deletion_auc",
               "iou_0.5", "dice_0.5", "pointing_game",
               "max_sensitivity", "entropy"]

summary = results_df.groupby(["arch", "method"])[metric_cols].agg(
    ["mean", "std"]
).round(4)

# Flatten multi-level columns
summary.columns = [f"{m}_{s}" for m, s in summary.columns]
summary = summary.reset_index()

summary.to_csv(os.path.join(OUT_ROOT, "summary_table.csv"), index=False)
print(summary.to_string())

# ── Plausibility stratified by lesion size (EDA §G) ──
plaus_cols = ["iou_0.5", "dice_0.5", "pointing_game"]
plaus_mask_df = results_df[results_df["lesion_size_group"].notna()]

size_summary = plaus_mask_df.groupby(
    ["arch", "method", "lesion_size_group"]
)[plaus_cols].agg(["mean", "std", "count"]).round(4)
size_summary.columns = [f"{m}_{s}" for m, s in size_summary.columns]
size_summary = size_summary.reset_index()

size_summary.to_csv(os.path.join(OUT_ROOT, "plausibility_by_size.csv"), index=False)
print("\n── Plausibility by lesion size group ──")
print(size_summary.to_string())
```

**Validation:** Main faithfulness table has 16 rows (4 models × 4 methods each); plausibility columns from `plaus_df_results` will be NaN for any row that has no plausibility counterpart. Size-stratified plausibility table has up to 32 rows (16 × 2 size groups). Small-lesion IoU/Dice values should be noticeably lower than standard-lesion values — if not, re-check the `lesion_size_group` computation in A.4.

---

#### Step G.3 — Radar (Spider) Charts

**Goal:** For each model-method pair, create a radar chart with 4 axes: Faithfulness (AOPC), Plausibility (IoU@0.5), Robustness (1 - normalised max-sensitivity), Complexity (1 - normalised entropy). All normalised to [0,1] where higher = better.

```python
from matplotlib.patches import FancyBboxPatch

def make_radar_chart(summary_df, save_path):
    """Create radar charts: one per architecture, methods overlaid."""
    categories = ["Faithfulness\n(AOPC)", "Plausibility\n(IoU@0.5)",
                  "Robustness\n(1-MaxSens)", "Complexity\n(1-Entropy)"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, axes = plt.subplots(2, 2, figsize=(14, 14),
                              subplot_kw=dict(projection="polar"))

    for ax, arch_key in zip(axes.flat, ARCHITECTURES):
        arch_data = summary_df[summary_df["arch"] == arch_key]

        for _, row in arch_data.iterrows():
            # Normalise metrics to [0,1] where higher=better
            vals = [
                row["aopc_mean"],                      # already higher=better
                row.get("iou_0.5_mean", 0),            # higher=better
                1 - row["max_sensitivity_mean"] / summary_df["max_sensitivity_mean"].max(),
                1 - row["entropy_mean"] / summary_df["entropy_mean"].max(),
            ]
            vals += vals[:1]
            ax.plot(angles, vals, "o-", label=row["method"], linewidth=2)
            ax.fill(angles, vals, alpha=0.1)

        ax.set_thetagrids(np.degrees(angles[:-1]), categories, size=8)
        ax.set_title(arch_key, fontsize=12, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax.set_ylim(0, 1)

    plt.suptitle("Multi-Dimensional XAI Evaluation Profiles", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

make_radar_chart(summary, os.path.join(OUT_ROOT, "radar_charts.png"))
```

---

#### Step G.4 — Faithfulness vs. Plausibility Scatter Plot

**Goal:** Test H3 (trade-off hypothesis). Each point = one model-method pair.

```python
def plot_faithfulness_vs_plausibility(summary_df, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"cnn": "#2196F3", "vit": "#FF5722"}
    markers = {"gradcam": "o", "hirescam": "s", "lime": "^",
               "kernelshap": "D", "attention_rollout": "P", "attnlrp": "*"}

    for _, row in summary_df.iterrows():
        family = ARCHITECTURES[row["arch"]]["family"]
        ax.scatter(
            row["aopc_mean"], row.get("iou_0.5_mean", 0),
            c=colors[family], marker=markers.get(row["method"], "o"),
            s=150, edgecolors="black", linewidth=0.5, zorder=3
        )
        ax.annotate(f"{row['arch'][:4]}-{row['method'][:4]}",
                    (row["aopc_mean"], row.get("iou_0.5_mean", 0)),
                    fontsize=7, ha="left", va="bottom")

    ax.set_xlabel("Faithfulness (AOPC) →", fontsize=12)
    ax.set_ylabel("Plausibility (IoU@0.5) →", fontsize=12)
    ax.set_title("Faithfulness vs Plausibility Trade-off (H3)", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Compute and show correlation
    from scipy.stats import pearsonr
    mask = ~summary_df["iou_0.5_mean"].isna()
    if mask.sum() > 2:
        r, p = pearsonr(summary_df.loc[mask, "aopc_mean"],
                        summary_df.loc[mask, "iou_0.5_mean"])
        ax.text(0.05, 0.95, f"Pearson r = {r:.3f} (p = {p:.3f})",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

plot_faithfulness_vs_plausibility(summary, os.path.join(OUT_ROOT, "faith_vs_plaus.png"))
```

---

#### Step G.5 — Statistical Significance Testing

**Goal:** Wilcoxon signed-rank test (paired per-image) between each pair of methods on each architecture. Bonferroni correction.

```python
from scipy.stats import wilcoxon
from itertools import combinations

def pairwise_significance(results_df, metric, arch_key, alpha=0.05):
    """Run pairwise Wilcoxon tests for all method pairs within one architecture."""
    arch_data = results_df[results_df["arch"] == arch_key]
    methods = arch_data["method"].unique()
    n_comparisons = len(list(combinations(methods, 2)))
    corrected_alpha = alpha / max(n_comparisons, 1)  # Bonferroni

    results = []
    for m1, m2 in combinations(methods, 2):
        d1 = arch_data[arch_data["method"] == m1].set_index("image_id")[metric]
        d2 = arch_data[arch_data["method"] == m2].set_index("image_id")[metric]
        common = d1.index.intersection(d2.index)
        if len(common) < 10:
            continue

        stat, p = wilcoxon(d1.loc[common], d2.loc[common])
        sig = "***" if p < corrected_alpha else "n.s."
        results.append({"arch": arch_key, "m1": m1, "m2": m2,
                        "metric": metric, "statistic": stat,
                        "p_value": p, "significant": sig})
    return pd.DataFrame(results)

# Run for key metrics
sig_results = []
for arch_key in ARCHITECTURES:
    for metric in ["aopc", "insertion_auc", "max_sensitivity"]:
        sig_results.append(pairwise_significance(results_df, metric, arch_key))

sig_df = pd.concat(sig_results, ignore_index=True)
sig_df.to_csv(os.path.join(OUT_ROOT, "significance_tests.csv"), index=False)
print(sig_df.to_string())
```

---

### PHASE H — Ablation Studies

> Maps to: **Thesis §7.3**

---

#### Step H.1 — Threshold Sensitivity Sweep

**Goal:** Compute IoU at thresholds 0.1–0.9 for each model-method pair to show sensitivity to threshold choice.

```python
sweep_thresholds = np.arange(0.1, 1.0, 0.1)

# Threshold sweep runs on plaus_df — the only population with ground-truth masks.
sweep_results = []
for arch_key in ARCHITECTURES:
    model = trained_models[arch_key].to(DEVICE).eval()
    methods = get_applicable_methods(arch_key)
    sweep_sample = plaus_df.head(50 if not DEBUG else 10)

    for method in methods:
        for _, row in sweep_sample.iterrows():
            img = apply_color_constancy(np.array(
                Image.open(row["img_path"]).convert("RGB")
            ))
            img_tensor = eval_transform(image=img)["image"].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                target_class = model(img_tensor).argmax(1).item()
            heatmap = generate_heatmap(model, img_tensor, method, arch_key, target_class)
            gt_mask = (np.array(Image.open(row["mask_path"]).convert("L")) > 127).astype(np.float32)

            for t in sweep_thresholds:
                plaus = compute_plausibility(heatmap, gt_mask, [round(t, 1)])
                sweep_results.append({
                    "arch": arch_key, "method": method,
                    "threshold": round(t, 1),
                    "iou": plaus[f"iou_{round(t, 1)}"]
                })

    model = model.cpu()
    torch.cuda.empty_cache()

sweep_df = pd.DataFrame(sweep_results)
# Plot: IoU vs threshold for each method, faceted by architecture
# (seaborn FacetGrid with hue=method)
```

---

#### Step H.2 — Per-Class Analysis

**Goal:** Break down all metrics by diagnostic class to test whether some lesion types are harder to explain.

> **EDA note (§F):** MEL has the highest hair + dark artifact burden — if Grad-CAM highlights hair/edges for MEL, note this is a legitimate training signal, not necessarily a failure. DF images are unusually sharp (blur_score 769), so DF heatmaps may appear more localised regardless of method quality.

```python
per_class = results_df.groupby(["arch", "method", "label_name"])[
    ["aopc", "insertion_auc", "iou_0.5", "entropy"]
].mean().round(4)

per_class.to_csv(os.path.join(OUT_ROOT, "per_class_analysis.csv"))
print(per_class.to_string())
```

---

#### Step H.3 — Confirmation-Type Sensitivity Analysis

**Goal:** Test whether XAI metric distributions differ between histopathology-confirmed and non-confirmed images (EDA §C). 46.7% of training labels are not histopathology-confirmed — if label noise is confounding results, plausibility and faithfulness scores will differ substantially between confirmation groups.

```python
from scipy.stats import mannwhitneyu

# Map confirm types to binary: histopathology vs non-histopathology
histo_label = "histopathology"  # exact value from groupings.csv
results_df["is_histo"] = results_df["confirm_type"] == histo_label

confirm_sensitivity = []
for arch_key in ARCHITECTURES:
    for method in get_applicable_methods(arch_key):
        subset = results_df[
            (results_df["arch"] == arch_key) &
            (results_df["method"] == method) &
            results_df["confirm_type"].notna()
        ]
        histo    = subset[subset["is_histo"]]
        non_hist = subset[~subset["is_histo"]]

        if len(histo) < 10 or len(non_hist) < 10:
            continue

        for metric in ["aopc", "iou_0.5", "pointing_game", "max_sensitivity"]:
            h_vals  = histo[metric].dropna()
            n_vals  = non_hist[metric].dropna()
            if len(h_vals) < 5 or len(n_vals) < 5:
                continue
            stat, p = mannwhitneyu(h_vals, n_vals, alternative="two-sided")
            confirm_sensitivity.append({
                "arch": arch_key, "method": method, "metric": metric,
                "histo_mean": h_vals.mean().round(4),
                "non_histo_mean": n_vals.mean().round(4),
                "delta": (h_vals.mean() - n_vals.mean()).round(4),
                "p_value": round(p, 4),
                "significant": "yes" if p < 0.05 else "no",
            })

conf_df = pd.DataFrame(confirm_sensitivity)
conf_df.to_csv(os.path.join(OUT_ROOT, "confirm_type_sensitivity.csv"), index=False)
print(conf_df.to_string())
```

**Validation:** If `significant == "yes"` for plausibility metrics, report in thesis §5.1 that label quality (confirmation method) confounds XAI plausibility results. If no significant differences, non-histopathology labels can be pooled without qualification.

---

### PHASE I — Outputs & Kaggle Adaptation

---

#### Step I.1 — Save All Outputs

```python
import datetime

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
final_dir = os.path.join(OUT_ROOT, timestamp)
os.makedirs(final_dir, exist_ok=True)

# Copy all CSV results
import shutil
for f in ["full_results.csv", "summary_table.csv", "significance_tests.csv",
          "per_class_analysis.csv", "plausibility_by_size.csv",
          "confirm_type_sensitivity.csv"]:
    src = os.path.join(OUT_ROOT, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(final_dir, f))

# Copy all figures
for f in ["radar_charts.png", "faith_vs_plaus.png", "xai_visual_validation.png",
          "sanity_train_batch.png"]:
    src = os.path.join(OUT_ROOT, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(final_dir, f))

# Save model checkpoints
for arch_key in ARCHITECTURES:
    src = os.path.join(OUT_ROOT, f"{arch_key}_best.pt")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(final_dir, f"{arch_key}_best.pt"))

print(f"✓ All outputs saved to {final_dir}")
```

---

## 4. Notebook Cell Layout

| Cell # | Type | Content | Phase |
|--------|------|---------|-------|
| 1 | Code | Configuration (A.1) | A |
| 2 | Code | Install dependencies (A.2) | A |
| 3 | Markdown | "## Data Loading & Preprocessing" | A |
| 4 | Code | Load & parse labels (A.3) | A |
| 5 | Code | Mask availability (A.4) | A |
| 6 | Code | Stratified split (A.5) — **skip on Kaggle if prepared/ exists** | A |
| 7 | Code | Class weights & normalisation stats (A.6) | A |
| 8 | Code | Dataset class & DataLoaders (A.7) | A |
| 9 | Code | Visual sanity check (A.8) | A |
| 10 | Markdown | "## Model Training" | B |
| 11 | Code | Model factory (B.1) | B |
| 12 | Code | Training loop function (B.2) | B |
| 13 | Code | Train all 4 models (B.3) | B |
| 14 | Code | Classification performance (B.4) | B |
| 15 | Code | Correct-classification subsets (B.5) | B |
| 16 | Markdown | "## XAI Methods" | C |
| 17 | Code | `generate_heatmap()` interface (C.1) | C |
| 18 | Code | Attention Rollout (C.2) | C |
| 19 | Code | Transformer Attribution (C.3) | C |
| 20 | Code | Visual validation grid (C.4) | C |
| 21 | Markdown | "## Evaluation Metrics" | D–F |
| 22 | Code | AOPC (D.1) | D |
| 23 | Code | Insertion/Deletion AUC (D.2) | D |
| 24 | Code | IoU, Dice, Pointing Game (E.1) | E |
| 25 | Code | Max-Sensitivity (F.1) | F |
| 26 | Code | Entropy (F.2) | F |
| 27 | Markdown | "## Full Evaluation" | G |
| 28 | Code | Main evaluation loop (G.1) | G |
| 29 | Code | Summary table (G.2) | G |
| 30 | Code | Radar charts (G.3) | G |
| 31 | Code | Faithfulness vs Plausibility plot (G.4) | G |
| 32 | Code | Statistical tests (G.5) | G |
| 33 | Markdown | "## Ablation Studies" | H |
| 34 | Code | Threshold sweep (H.1) | H |
| 35 | Code | Per-class analysis (H.2) | H |
| 36 | Code | Confirmation-type sensitivity analysis (H.3) | H |
| 37 | Markdown | "## Save Outputs" | I |
| 38 | Code | Save all outputs (I.1) | I |

---

## 5. Final Kaggle Adaptation Notes

### Paths
| Variable | Local | Kaggle |
|----------|-------|--------|
| `DATA_ROOT` | `./data/isic2018` | `/kaggle/input/isic2018-skin-lesion` |
| `PREP_ROOT` | `./prepared` | `/kaggle/input/isic2018-prepared` |
| `OUT_ROOT` | `./outputs` | `/kaggle/working` |
| `IMG_DIR` | relative to DATA_ROOT | relative to DATA_ROOT |

**Auto-switching is handled by cell 1** (`KAGGLE = "KAGGLE_URL_BASE" in os.environ`).

### Datasets to attach in Kaggle kernel settings
1. **ISIC 2018 images + masks** — search Kaggle for `isic 2018` dataset or upload your own
2. **prepared/** — upload once via `kaggle datasets create -p prepared/`

### GPU
Set **GPU T4 x2** (or P100) in Kaggle kernel settings → "Accelerator".

### Skipping splits on Kaggle
Cell 6 (stratified split) should detect that `prepared/train.csv` already exists and skip:
```python
if os.path.exists(os.path.join(PREP_ROOT, "train.csv")):
    print("Loading pre-computed splits from prepared/")
    train_df = pd.read_csv(os.path.join(PREP_ROOT, "train.csv"))
    val_df   = pd.read_csv(os.path.join(PREP_ROOT, "val.csv"))
    test_df  = pd.read_csv(os.path.join(PREP_ROOT, "test.csv"))
else:
    # ... split logic from A.5 ...
```

### Runtime budget
| Component | Est. time (T4) |
|-----------|----------------|
| Train 4 models (50 epochs each) | ~4–6 hours |
| XAI heatmap generation (all cells) | ~3–5 hours |
| Metric computation | ~1–2 hours |
| **Total** | **~8–13 hours** |

Kaggle's GPU limit is 12 hours/session. Strategy: **split into 2 notebooks** if needed:
- **Notebook 1:** Train + save checkpoints + heatmap cache
- **Notebook 2:** Load checkpoints + load cached heatmaps + metric computation + analysis

Alternatively, set `DEBUG=True` for a faster end-to-end test (~30 min).

### Scripts (same pattern as GlaS)
Adapt `scripts/push_to_kaggle.sh` to use the new kernel slug and dataset sources:
```json
{
  "id": "yehiasamir/xai-evaluation-pipeline",
  "title": "XAI Evaluation Pipeline",
  "code_file": "XAI_Evaluation_Pipeline_Kaggle.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "dataset_sources": [
    "YOUR_USERNAME/isic2018-skin-lesion",
    "YOUR_USERNAME/isic2018-prepared"
  ],
  "enable_gpu": true,
  "competition_sources": [],
  "enable_internet": true
}
```

---

## 6. Dependency Summary

| Package | Purpose | Install |
|---------|---------|---------|
| `timm` | 4 pretrained architectures | `pip install timm` |
| `albumentations` | Augmentation pipeline | `pip install albumentations` |
| `pytorch-grad-cam` | Grad-CAM, HiResCAM | `pip install pytorch-grad-cam` |
| `captum` | LIME, KernelSHAP, LRP | `pip install captum` |
| Attention Rollout | Abnar & Zuidema (2020) | Implemented manually in Phase C — no package needed |
| `lime` | Backup LIME implementation | `pip install lime` |
| `shap` | Backup SHAP implementation | `pip install shap` |
| `seaborn` | Plotting | `pip install seaborn` |
| `wandb` | Experiment tracking (optional) | `pip install wandb` |
| `statsmodels` | Statistical tests | `pip install statsmodels` |
| `scikit-image` | Heatmap resizing | Usually pre-installed |

---

## 7. Checkpoints & Resume Strategy

After each major phase, save intermediate state so you can resume without re-running everything:

| After Phase | What to save | Filename |
|-------------|-------------|----------|
| B (training) | Model weights | `{arch}_best.pt` |
| C (heatmaps) | All heatmaps as .npy | `heatmaps/{arch}_{method}_{image_id}.npy` |
| G (evaluation) | Results DataFrame | `full_results.csv` |

Add a `RESUME_FROM` config flag in cell 1:
```python
RESUME_FROM = None  # or "training" or "heatmaps" or "evaluation"
```

This lets you restart from any phase after a Kaggle timeout.

---

## 8. EDA Findings → Plan Mapping

Quick-reference table connecting every actionable EDA finding to its implementation location.

| EDA Finding | Severity | Plan Location | What Changed |
|-------------|----------|---------------|--------------|
| 58.3× class imbalance (DF vs NV) | **High** | Step B.2 | `CrossEntropyLoss` replaced with `FocalLoss(gamma=2)` |
| 19.1% of masked images have lesion area < 5% | **High** | Steps A.4, E.1, G.2 | `lesion_area_frac` + `lesion_size_group` columns added to `plaus_df`; Phase E stratifies plausibility by size group |
| 46.7% of training labels are non-histopathology | Medium | Steps A.3, H.3 | `diagnosis_confirm_type` retained in all DataFrames; Step H.3 added for sensitivity analysis |
| Patient-level leakage (~26% multi-image lesions) | Medium | Step A.3, thesis §5.1 | `lesion_id` attached to `train_df`; documented as known limitation — no re-split |
| Test set is sharper than train (Cohen's d = 0.21) | Low | Phase E validation note | Caveat added: heatmaps on test may appear sharper, inflating plausibility scores marginally |
| Test set has more hair (Cohen's d = 0.21) | Low | Phase E/G notes | MEL heatmap caveats documented; no hair-simulation augmentation needed |
| 5 near-duplicate train↔test pairs | Low | Thesis §5.1 | Exclude from qualitative case-study examples; no metric impact (0.33% of test) |
| Pigment network dominant L2 attribute (n=1,950) | Low | Phase E note | Identified as primary L2 attribute; streaks/negative_network treated as exploratory |
| Val → test contrast gap (d = 0.337) | Low | Phase G note | Val loss still valid for early stopping; shift documented in §5.1 |
| All images > 224px (0% upsampling) | — | Step A.1 / A.5 | Direct resize confirmed safe; no change needed |
| 17/18 integrity checks passed | — | Pipeline start | No filtering or repair step needed |
