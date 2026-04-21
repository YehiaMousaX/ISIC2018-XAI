# %% [markdown]
# # ISIC 2018 Skin Lesion Classifier — Paper-Faithful SENet-154
#
# Single-model 7-class classification on ISIC 2018, reproducing the recipe from
# Zhuang et al. (2018) — the winner-tier configuration documented in
# `mds/Classification.md`.
#
# **Recipe (paper-faithful):**
# - Backbone: SENet-154 (ImageNet-pretrained), `num_classes=7`
# - Input: Resize 300×300 → RandomCrop 224 (train) / CenterCrop 224 (eval)
# - Color constancy: Shades-of-Gray, power=6
# - Loss: **Class-weighted CrossEntropy** (inverse-frequency weights)
# - Sampler: **Standard random shuffle** (paper rejects resampling)
# - Optimizer: AdamW + CosineAnnealingLR, AMP
# - Early stopping on **val balanced accuracy**
#
# **Targets:** bAcc ≥ 0.80, top-1 ≥ 0.90, macro-F1 ≥ 0.75, macro-AUC ≥ 0.95.
#
# Runs locally (`DEBUG=True` → 500-image subset × 3 epochs) and on Kaggle T4.
# XAI is intentionally out of scope — a separate notebook loads `senet154_best.pt`
# and runs explainability analysis.

# %% [markdown]
# ## 1 — Configuration

# %%
import os, json, random, time, warnings
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

DEBUG  = False   # True → 500-image subset, 3 epochs — local smoke test
KAGGLE = "KAGGLE_URL_BASE" in os.environ
SEED   = 42

KAGGLE_DATASET_SLUG = "isic2018-dataset"
KAGGLE_USER         = "yehiasamir"

if KAGGLE:
    DATA_ROOT = "/kaggle/input/datasets/yehiasamir/isic2018-dataset/Data"
    PREP_ROOT = "/kaggle/working/prepared"
    OUT_ROOT  = "/kaggle/working"
    CSV_DIR   = "/kaggle/input/datasets/yehiasamir/new-train-val-csv"

    from huggingface_hub import login
    with open("/kaggle/input/datasets/yehiasamir/hf-credentials/hf_token.txt") as f:
        login(f.read().strip())
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
else:
    DATA_ROOT = "./Data"
    PREP_ROOT = "./prepared"
    OUT_ROOT  = "./outputs"
    CSV_DIR   = os.path.join(DATA_ROOT, "csv")

TRAIN_IMG = os.path.join(DATA_ROOT, "images", "train")
VAL_IMG   = os.path.join(DATA_ROOT, "images", "val")
TEST_IMG  = os.path.join(DATA_ROOT, "images", "test")

# ── Model / training hyperparameters ────────────────────────────────────────
BACKBONE      = "legacy_senet154"    # try first; fallback to "senet154" below
IMG_RESIZE    = 300                  # paper: resize 300, crop 224
IMG_CROP      = 224
BATCH_SIZE    = 16                   # SENet-154 is 115M params; 32 risks OOM on T4
ACCUM_STEPS   = 1                    # bump to 2 if OOM at batch=16
MAX_EPOCHS    = 60  if not DEBUG else 3
PATIENCE      = 10  if not DEBUG else 2
MIN_DELTA     = 0.002
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
NUM_CLASSES   = 7
NUM_WORKERS   = 4 if KAGGLE else 0
USE_AMP       = True

CLASS_NAMES   = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# ── Output directory for this run ───────────────────────────────────────────
RUN_TAG = datetime.now().strftime("%Y-%m-%d_%H-%M")
if KAGGLE:
    RUN_DIR = OUT_ROOT           # Kaggle flattens working dir on download
else:
    RUN_DIR = os.path.join(OUT_ROOT, RUN_TAG)

os.makedirs(RUN_DIR,   exist_ok=True)
os.makedirs(PREP_ROOT, exist_ok=True)


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device      : {DEVICE}")
print(f"DEBUG       : {DEBUG}")
print(f"KAGGLE      : {KAGGLE}")
print(f"DATA_ROOT   : {DATA_ROOT}  (exists={os.path.isdir(DATA_ROOT)})")
print(f"RUN_DIR     : {RUN_DIR}")
print(f"Backbone    : {BACKBONE} | img: resize {IMG_RESIZE} → crop {IMG_CROP}")

# %% [markdown]
# ## 2 — Install dependencies (Kaggle only)

# %%
import subprocess, sys

def pip_install(*pkgs):
    if not KAGGLE:
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip_install("timm", "albumentations", "seaborn")

import timm, albumentations
print(f"timm={timm.__version__} | albumentations={albumentations.__version__}")

# %% [markdown]
# ## 3 — Load CSVs
# Reads pre-split CSVs, converts one-hot class columns to a single `label_idx`.

# %%
import pandas as pd


def find_image(img_id: str, primary_dir: str, ext: str = ".jpg") -> str:
    """Resolve image path; on Kaggle fall back across train/val/test subdirs."""
    candidate = os.path.join(primary_dir, f"{img_id}{ext}")
    if not KAGGLE or os.path.exists(candidate):
        return candidate
    img_root = os.path.dirname(primary_dir)
    for split in ("train", "val", "test"):
        p = os.path.join(img_root, split, f"{img_id}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Image not found: {img_id}")


def load_split(csv_path):
    df = pd.read_csv(csv_path).rename(columns={"image": "image_id"})
    df["label_idx"]  = df[CLASS_NAMES].values.argmax(axis=1)
    df["label_name"] = df["label_idx"].map(lambda i: CLASS_NAMES[i])
    return df[["image_id", "label_idx", "label_name"]]


train_df = load_split(os.path.join(CSV_DIR, "train.csv"))
val_df   = load_split(os.path.join(CSV_DIR, "val.csv"))
test_df  = load_split(os.path.join(CSV_DIR, "test.csv"))

if DEBUG:
    # Stratified subsample for smoke tests
    N_DEBUG = 500
    train_df = pd.concat([
        g.sample(max(1, round(N_DEBUG * len(g) / len(train_df))), random_state=SEED)
        for _, g in train_df.groupby("label_idx", group_keys=False)
    ]).reset_index(drop=True)
    val_df  = val_df.sample(min(200, len(val_df)),  random_state=SEED).reset_index(drop=True)
    test_df = test_df.sample(min(200, len(test_df)), random_state=SEED).reset_index(drop=True)
    print(f"DEBUG subsample → train={len(train_df)} val={len(val_df)} test={len(test_df)}")

print(f"Train : {len(train_df):>5} | Val : {len(val_df):>4} | Test : {len(test_df):>4}")
print("\nTrain label distribution:")
print(train_df["label_name"].value_counts())

# %% [markdown]
# ## 4 — Class weights & normalization stats
# Inverse-frequency class weights for the loss; per-channel mean/std on the
# color-constancy-transformed train distribution for input normalization.

# %%
def apply_color_constancy(img: np.ndarray, power: int = 6) -> np.ndarray:
    """Shades-of-Gray color constancy on a uint8 HxWx3 image."""
    img_float = img.astype(np.float32) + 1e-6
    norm = (np.mean(img_float ** power, axis=(0, 1)) ** (1.0 / power))
    scale = norm.mean() / norm
    return np.clip(img_float * scale, 0, 255).astype(np.uint8)


# ── Class weights ───────────────────────────────────────────────────────────
counts = Counter(train_df["label_idx"].values)
total  = sum(counts.values())
class_weights = {int(k): total / (NUM_CLASSES * v) for k, v in counts.items()}
with open(os.path.join(PREP_ROOT, "class_weights.json"), "w") as f:
    json.dump(class_weights, f, indent=2)

print("Class weights (higher = rarer):")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:6s} [{i}]: {class_weights.get(i, 0):.4f}")

# ── Per-channel mean / std on color-constancy-adjusted images ──────────────
STATS_SAMPLE = 200 if DEBUG else 2000
sample_ids   = train_df["image_id"].sample(min(STATS_SAMPLE, len(train_df)),
                                           random_state=SEED)
pixel_sum    = np.zeros(3, dtype=np.float64)
pixel_sq_sum = np.zeros(3, dtype=np.float64)
n_pixels     = 0

for img_id in tqdm(sample_ids, desc="Computing stats"):
    raw = np.array(Image.open(find_image(img_id, TRAIN_IMG))
                   .convert("RGB").resize((IMG_CROP, IMG_CROP)))
    img  = apply_color_constancy(raw) / 255.0
    flat = img.reshape(-1, 3)
    pixel_sum    += flat.sum(0)
    pixel_sq_sum += (flat ** 2).sum(0)
    n_pixels     += flat.shape[0]

mean = pixel_sum / n_pixels
std  = np.sqrt(pixel_sq_sum / n_pixels - mean ** 2)
data_stats = {"mean": mean.tolist(), "std": std.tolist()}
with open(os.path.join(PREP_ROOT, "data_stats.json"), "w") as f:
    json.dump(data_stats, f, indent=2)

print(f"\nTrain mean : {mean.round(4)}")
print(f"Train std  : {std.round(4)}")

# %% [markdown]
# ## 5 — Dataset class, augmentations, DataLoaders
# Paper augmentation: resize 300 → random crop 224, H/V flip, ±180° rotation,
# ColorJitter, affine. Eval: resize 300 → center crop 224. No WeightedRandomSampler.

# %%
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ISICClassifierDataset(Dataset):
    """ISIC 2018 classification dataset. Returns (image_tensor, label_idx)."""

    def __init__(self, df, img_dir, transform):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = apply_color_constancy(
            np.array(Image.open(find_image(row["image_id"], self.img_dir))
                     .convert("RGB"))
        )
        img = self.transform(image=img)["image"]
        label = torch.tensor(int(row["label_idx"]), dtype=torch.long)
        return img, label


train_transform = A.Compose([
    A.Resize(IMG_RESIZE, IMG_RESIZE),
    A.RandomCrop(IMG_CROP, IMG_CROP),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=180, p=0.7),
    A.Affine(translate_percent=0.05, scale=(0.9, 1.1), p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
    A.Normalize(mean=data_stats["mean"], std=data_stats["std"]),
    ToTensorV2(),
])

eval_transform = A.Compose([
    A.Resize(IMG_RESIZE, IMG_RESIZE),
    A.CenterCrop(IMG_CROP, IMG_CROP),
    A.Normalize(mean=data_stats["mean"], std=data_stats["std"]),
    ToTensorV2(),
])

train_ds = ISICClassifierDataset(train_df, TRAIN_IMG, train_transform)
val_ds   = ISICClassifierDataset(val_df,   VAL_IMG,   eval_transform)
test_ds  = ISICClassifierDataset(test_df,  TEST_IMG,  eval_transform)

_persist = NUM_WORKERS > 0
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=_persist, drop_last=True)
val_loader   = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=_persist)
test_loader  = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=_persist)

# Sanity batch
imgs, labels = next(iter(train_loader))
print(f"Batch shape : {imgs.shape}")
print(f"Label dist  : {labels.bincount(minlength=NUM_CLASSES).tolist()}")
print(f"Pixel range : [{imgs.min():.3f}, {imgs.max():.3f}]")

# %% [markdown]
# ## 6 — Model factory (SENet-154 via timm)

# %%
def create_senet154():
    """Create SENet-154 with 7-class head. Falls back across timm name variants."""
    for name in (BACKBONE, "senet154"):
        try:
            model = timm.create_model(name, pretrained=True,
                                      num_classes=NUM_CLASSES)
            resolved = name
            break
        except Exception as e:
            last_err = e
            print(f"  timm name '{name}' failed: {e}")
    else:
        raise RuntimeError(f"Could not load SENet-154 from timm: {last_err}")
    return model.to(DEVICE), resolved


model, resolved_name = create_senet154()
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model   : {resolved_name}")
print(f"Params  : {n_params:.1f}M")
print(f"Device  : {next(model.parameters()).device}")

# %% [markdown]
# ## 7 — Training loop (class-weighted CE, AdamW + Cosine, AMP, early-stop on val bAcc)

# %%
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             recall_score, f1_score, accuracy_score)
import copy


def evaluate(model, loader, criterion):
    """Single-pass eval. Returns (loss, preds, labels, probs)."""
    model.eval()
    loss_sum, n = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            with autocast(enabled=USE_AMP):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            loss_sum += loss.item() * imgs.size(0)
            n        += imgs.size(0)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    return (loss_sum / max(n, 1),
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


def train_model():
    # Paper-faithful: class-weighted CE, no focal, no label smoothing.
    w_tensor = torch.tensor([class_weights[i] for i in range(NUM_CLASSES)],
                            dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w_tensor)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    scaler    = GradScaler(enabled=USE_AMP)

    best_val_bacc    = -1.0
    best_state       = None
    best_epoch       = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_bacc": [], "val_top1": [],
               "val_auc": [], "val_f1_macro": [], "val_per_class_recall": [],
               "lr": []}

    t_start = time.time()
    for epoch in range(MAX_EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        running_loss, n = 0.0, 0
        optimizer.zero_grad()
        bar = tqdm(train_loader, desc=f"Ep {epoch+1:02d}/{MAX_EPOCHS} train",
                   leave=False, unit="batch")
        for step, (imgs, labels) in enumerate(bar):
            imgs   = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast(enabled=USE_AMP):
                logits = model(imgs)
                loss   = criterion(logits, labels) / ACCUM_STEPS

            scaler.scale(loss).backward()
            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * imgs.size(0) * ACCUM_STEPS
            n            += imgs.size(0)
            bar.set_postfix(loss=f"{loss.item() * ACCUM_STEPS:.4f}")
        train_loss = running_loss / max(n, 1)

        # ── Validate ───────────────────────────────────────────────────────
        val_loss, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion)
        val_bacc = balanced_accuracy_score(val_labels, val_preds)
        val_top1 = accuracy_score(val_labels, val_preds)
        try:
            val_auc = roc_auc_score(val_labels, val_probs, multi_class="ovr", average="macro")
        except ValueError:
            val_auc = float("nan")   # can fail early if some class absent in val preds
        val_f1   = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        per_rec  = recall_score(val_labels, val_preds, average=None,
                                zero_division=0, labels=list(range(NUM_CLASSES))).tolist()

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_bacc"].append(val_bacc)
        history["val_top1"].append(val_top1)
        history["val_auc"].append(val_auc)
        history["val_f1_macro"].append(val_f1)
        history["val_per_class_recall"].append(per_rec)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        rec_str = "  ".join(f"{n}={r:.2f}" for n, r in zip(CLASS_NAMES, per_rec))
        print(
            f"[{resolved_name}] Ep {epoch+1:02d}/{MAX_EPOCHS}"
            f" | train={train_loss:.4f} | val={val_loss:.4f}"
            f" | top1={val_top1:.4f} | bAcc={val_bacc:.4f}"
            f" | F1={val_f1:.4f} | AUC={val_auc:.4f}",
            flush=True,
        )
        print(f"    per-class recall: {rec_str}", flush=True)

        # Early-stop on val bAcc (the target metric)
        if val_bacc > best_val_bacc + MIN_DELTA:
            best_val_bacc    = val_bacc
            best_epoch       = epoch + 1
            best_state       = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"    ✓ new best val bAcc={best_val_bacc:.4f} @ epoch {best_epoch}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1} "
                      f"(no val-bAcc improvement for {PATIENCE} epochs)", flush=True)
                break

    elapsed = time.time() - t_start
    model.load_state_dict(best_state)
    print(f"\n✓ Training done in {elapsed/60:.1f} min | "
          f"best val bAcc={best_val_bacc:.4f} @ epoch {best_epoch}")
    return history, best_val_bacc, best_epoch, elapsed


history, best_val_bacc, best_epoch, train_elapsed = train_model()

# Save checkpoint + training history immediately
ckpt_path = os.path.join(RUN_DIR, "senet154_best.pt")
torch.save(model.state_dict(), ckpt_path)
print(f"  Saved: {ckpt_path}")

with open(os.path.join(RUN_DIR, "history_senet154.json"), "w") as f:
    json.dump(history, f, indent=2)

# %% [markdown]
# ## 8 — Test-set evaluation
# Single forward pass per image (no TTA). All the metrics the thesis cares about.

# %%
from sklearn.metrics import (classification_report, confusion_matrix,
                             matthews_corrcoef, cohen_kappa_score,
                             top_k_accuracy_score, precision_score)
import matplotlib.pyplot as plt
import seaborn as sns


# Rebuild criterion for test-loss (same class weights as training)
w_tensor   = torch.tensor([class_weights[i] for i in range(NUM_CLASSES)],
                          dtype=torch.float32, device=DEVICE)
_criterion = nn.CrossEntropyLoss(weight=w_tensor)

test_loss, test_preds, test_labels, test_probs = evaluate(model, test_loader, _criterion)

top1      = accuracy_score(test_labels, test_preds)
top3      = top_k_accuracy_score(test_labels, test_probs, k=3,
                                 labels=list(range(NUM_CLASSES)))
bacc      = balanced_accuracy_score(test_labels, test_preds)
macro_f1  = f1_score(test_labels, test_preds, average="macro", zero_division=0)
weight_f1 = f1_score(test_labels, test_preds, average="weighted", zero_division=0)
macro_auc = roc_auc_score(test_labels, test_probs, multi_class="ovr", average="macro")
mcc       = matthews_corrcoef(test_labels, test_preds)
kappa     = cohen_kappa_score(test_labels, test_preds)
per_prec  = precision_score(test_labels, test_preds, average=None,
                            zero_division=0, labels=list(range(NUM_CLASSES))).tolist()
per_rec   = recall_score(test_labels, test_preds, average=None,
                         zero_division=0, labels=list(range(NUM_CLASSES))).tolist()
per_f1    = f1_score(test_labels, test_preds, average=None,
                     zero_division=0, labels=list(range(NUM_CLASSES))).tolist()

cls_report_txt = classification_report(test_labels, test_preds,
                                       target_names=CLASS_NAMES, digits=4,
                                       zero_division=0)
cm = confusion_matrix(test_labels, test_preds, labels=list(range(NUM_CLASSES)))

metrics = {
    "test_loss": test_loss,
    "top_1": top1, "top_3": top3,
    "balanced_accuracy": bacc,
    "macro_f1": macro_f1, "weighted_f1": weight_f1,
    "macro_auc": macro_auc,
    "mcc": mcc, "cohen_kappa": kappa,
    "per_class_precision": dict(zip(CLASS_NAMES, per_prec)),
    "per_class_recall":    dict(zip(CLASS_NAMES, per_rec)),
    "per_class_f1":        dict(zip(CLASS_NAMES, per_f1)),
    "confusion_matrix": cm.tolist(),
}
with open(os.path.join(RUN_DIR, "metrics_senet154.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("=" * 60)
print(f"SENet-154 — Test Metrics")
print("=" * 60)
print(f"Top-1              : {top1:.4f}")
print(f"Top-3              : {top3:.4f}")
print(f"Balanced accuracy  : {bacc:.4f}   (target ≥ 0.80)")
print(f"Macro F1           : {macro_f1:.4f}   (target ≥ 0.75)")
print(f"Weighted F1        : {weight_f1:.4f}")
print(f"Macro AUC (OvR)    : {macro_auc:.4f}   (target ≥ 0.95)")
print(f"MCC                : {mcc:.4f}")
print(f"Cohen κ            : {kappa:.4f}")
print()
print("Per-class recall:")
for n, r in zip(CLASS_NAMES, per_rec):
    flag = "" if r >= 0.50 else "  ← below 0.50"
    print(f"  {n:6s}: {r:.4f}{flag}")
print()
print(cls_report_txt)

# %% [markdown]
# ## 9 — Confusion matrix + training curves

# %%
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar=False, square=True, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"SENet-154 — Confusion Matrix (Test)\n"
             f"bAcc={bacc:.4f} | top-1={top1:.4f}", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "confusion_matrix.png"), dpi=120, bbox_inches="tight")
plt.show()

# ── Training curves ─────────────────────────────────────────────────────────
epochs = range(1, len(history["train_loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].plot(epochs, history["train_loss"], label="train", color="#e05c5c", lw=1.8)
axes[0].plot(epochs, history["val_loss"],   label="val",   color="#5c9be0", lw=1.8, ls="--")
axes[0].axvline(best_epoch, color="gray", ls=":", alpha=0.6, label=f"best ep {best_epoch}")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Loss"); axes[0].legend(fontsize=8)

axes[1].plot(epochs, history["val_bacc"],     label="val bAcc",   color="#5cbf7a", lw=1.8)
axes[1].plot(epochs, history["val_top1"],     label="val top-1",  color="#b57edc", lw=1.4)
axes[1].plot(epochs, history["val_f1_macro"], label="val macro-F1", color="#e0a85c", lw=1.4, ls="--")
axes[1].axhline(0.80, color="red", ls=":", alpha=0.5, label="target bAcc=0.80")
axes[1].axvline(best_epoch, color="gray", ls=":", alpha=0.6)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score"); axes[1].set_ylim(0, 1)
axes[1].set_title("Validation metrics"); axes[1].legend(fontsize=8)

fig.suptitle(f"SENet-154 — Training Curves ({len(epochs)} epochs)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "training_curves.png"), dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 10 — Save metadata + human-readable report

# %%
metadata = {
    "backbone": "SENet-154",
    "timm_name": resolved_name,
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "img_resize": IMG_RESIZE,
    "img_crop": IMG_CROP,
    "normalization_mean": data_stats["mean"],
    "normalization_std":  data_stats["std"],
    "recipe": "paper-faithful-weighted-ce",
    "loss": "CrossEntropyLoss(inverse-freq class weights)",
    "sampler": "random-shuffle",
    "optimizer": f"AdamW(lr={LR}, wd={WEIGHT_DECAY})",
    "scheduler": f"CosineAnnealingLR(T_max={MAX_EPOCHS})",
    "batch_size": BATCH_SIZE,
    "accumulation_steps": ACCUM_STEPS,
    "use_amp": USE_AMP,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "early_stop_metric": "val_balanced_accuracy",
    "best_epoch": best_epoch,
    "best_val_bacc": best_val_bacc,
    "train_elapsed_seconds": train_elapsed,
    "run_tag": RUN_TAG,
    "seed": SEED,
    "debug": DEBUG,
    "kaggle": KAGGLE,
}
with open(os.path.join(RUN_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# ── Human-readable evaluation report ────────────────────────────────────────
TARGETS = {"bAcc": 0.80, "top-1": 0.90, "macro-F1": 0.75, "macro-AUC": 0.95}
achieved = {"bAcc": bacc, "top-1": top1, "macro-F1": macro_f1, "macro-AUC": macro_auc}

report = [
    "=" * 70,
    f"ISIC 2018 Classification — SENet-154 (paper-faithful recipe)",
    f"Run: {RUN_TAG}",
    "=" * 70,
    "",
    f"Backbone        : {resolved_name} ({n_params:.1f}M params)",
    f"Input           : resize {IMG_RESIZE} → crop {IMG_CROP}",
    f"Loss            : class-weighted CrossEntropy (no focal, no sampler)",
    f"Optimizer       : AdamW (lr={LR}, wd={WEIGHT_DECAY}) + CosineAnnealingLR",
    f"Batch           : {BATCH_SIZE} (accum={ACCUM_STEPS}) | AMP={USE_AMP}",
    f"Epochs          : ran {len(history['train_loss'])} (max {MAX_EPOCHS}) | "
    f"best bAcc @ epoch {best_epoch}",
    f"Train elapsed   : {train_elapsed/60:.1f} min",
    "",
    "─" * 70,
    "Test metrics vs. targets",
    "─" * 70,
]
for k, tgt in TARGETS.items():
    got  = achieved[k]
    mark = "PASS" if got >= tgt else "MISS"
    report.append(f"  {k:10s} : {got:.4f}   target ≥ {tgt:.2f}   [{mark}]")

report += [
    "",
    f"  Top-3 accuracy      : {top3:.4f}",
    f"  Weighted F1         : {weight_f1:.4f}",
    f"  MCC                 : {mcc:.4f}",
    f"  Cohen kappa         : {kappa:.4f}",
    "",
    "─" * 70,
    "Per-class performance",
    "─" * 70,
    f"  {'Class':6s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}",
]
for i, name in enumerate(CLASS_NAMES):
    report.append(f"  {name:6s}  {per_prec[i]:10.4f}  {per_rec[i]:10.4f}  {per_f1[i]:10.4f}")

report += [
    "",
    "─" * 70,
    "Classification report (sklearn)",
    "─" * 70,
    cls_report_txt,
    "─" * 70,
    f"Checkpoint      : {ckpt_path}",
    f"Metadata        : metadata.json",
    f"Training curves : training_curves.png",
    f"Confusion       : confusion_matrix.png",
    "=" * 70,
]
report_txt = "\n".join(report)
with open(os.path.join(RUN_DIR, "evaluation_report.txt"), "w", encoding="utf-8") as f:
    f.write(report_txt)

print(report_txt)

# %% [markdown]
# ## 11 — Summary

# %%
print("━" * 60)
print(f"Run  : {RUN_TAG}")
print(f"bAcc : {bacc:.4f}   (target 0.80) — "
      f"{'PASS' if bacc >= 0.80 else 'MISS'}")
print(f"top-1: {top1:.4f}   (target 0.90) — "
      f"{'PASS' if top1 >= 0.90 else 'MISS'}")
print(f"Model: {ckpt_path}")
print(f"Dir  : {RUN_DIR}")
print("━" * 60)
