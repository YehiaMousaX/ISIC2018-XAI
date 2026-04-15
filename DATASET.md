# ISIC 2018 Dataset — Complete Reference

> **Challenge:** Skin Lesion Analysis Towards Melanoma Detection (ISIC 2018)
> **Source:** International Skin Imaging Collaboration (ISIC)
> **Underlying dataset:** ISIC 2018
> **License:** Task 1 & 2 — CC-0; Task 3 — CC-BY-NC

---

## Overview

The ISIC 2018 challenge is divided into three tasks, each building on a shared image pool. All images are dermoscopic photographs (close-up skin surface microscopy). The challenge uses **two distinct image pools**:

| Pool | Tasks | Images |
|------|-------|--------|
| Pool A | Task 1, Task 2 | 2,594 training + 100 val + 1,000 test |
| Pool B | Task 3 | 10,015 training + 193 val + 1,512 test |

---

## Directory Structure (Local)

```
Data/
├── images/
│   ├── train/                               # 10,015 dermoscopy images (.jpg) — classification training
│   ├── val/                                 # 193 images (.jpg)
│   └── test/                                # 1,512 images (.jpg)
│
├── plausibility/
│   ├── images/                              # 3,694 dermoscopy images (.jpg) — XAI evaluation only
│   ├── masks/                               # 3,694 binary lesion masks (.png) — L1 plausibility
│   └── attributes/                          # 16,104 attribute masks (.png) — L2 plausibility
│       # naming: ISIC_XXXXXXX_attribute_<name>.png
│       # attributes: globules, milia_like_cyst, negative_network, pigment_network, streaks
│
└── csv/
    ├── train.csv               # 10,015 rows — one-hot labels
    ├── val.csv                 # 193 rows
    ├── test.csv                # 1,512 rows
    └── lesion_groupings.csv    # lesion_id ↔ image_id ↔ confirm type
```

---

## Plausibility Data — Lesion Masks (L1) & Attribute Masks (L2)

### L1 — Lesion Segmentation Masks (`plausibility/masks/`)

Binary masks showing the exact boundary of the lesion. Used to answer: *"Is the XAI heatmap focusing on the lesion or the background?"*

- **Format:** Single-channel PNG. Pixel values: `255` = lesion, `0` = background.
- **Naming:** `ISIC_XXXXXXX_segmentation.png`
- **Count:** 3,694 masks (one per image in `plausibility/images/`)

### L2 — Attribute Masks (`plausibility/attributes/`)

Five binary masks per image marking specific clinical dermoscopy features. Used to answer: *"Within the lesion, is the XAI heatmap focusing on clinically meaningful structures?"*

- **Format:** Single-channel PNG. `255` = attribute present, `0` = absent.
- **Naming:** `ISIC_XXXXXXX_attribute_<name>.png`
- **Count:** 16,104 masks across 2,594 images (some test images have fewer than 5)

| Attribute | Clinical meaning |
|-----------|-----------------|
| `globules` | Round/oval dark structures |
| `milia_like_cyst` | White/yellowish cyst-like structures |
| `negative_network` | Inverse pigment network pattern |
| `pigment_network` | Reticular pattern of intersecting lines |
| `streaks` | Radial projections at lesion periphery |

**Note:** Many attribute masks will be all-zeros for a given image (the attribute is simply absent). Filter to non-empty masks before computing metrics.

---

## Classification Data — Disease Classification (Primary Task for Thesis)

**Goal:** Classify dermoscopy images into one of 7 diagnostic categories.

This is the primary task for the XAI evaluation thesis.

### Classes

| Code | Full Name | Training Count | Training % |
|------|-----------|---------------|-----------|
| NV | Melanocytic Nevi | 6,705 | 66.95% |
| MEL | Melanoma | 1,113 | 11.11% |
| BKL | Benign Keratosis-like Lesions | 1,099 | 10.97% |
| BCC | Basal Cell Carcinoma | 514 | 5.13% |
| AKIEC | Actinic Keratoses & Intraepithelial Carcinoma | 327 | 3.27% |
| VASC | Vascular Lesions | 142 | 1.42% |
| DF | Dermatofibroma | 115 | 1.15% |
| **Total** | | **10,015** | 100% |

**Class imbalance:** NV dominates at ~67%. The rarest class (DF) is 58× less frequent than NV. This severe imbalance requires weighted loss or oversampling strategies.

### Class Descriptions

- **NV (Melanocytic Nevi):** Benign moles. Most common skin lesion. The dominant class in this dataset.
- **MEL (Melanoma):** Malignant skin cancer. The most clinically important class to detect correctly. High false-negative cost.
- **BKL (Benign Keratosis):** Includes seborrheic keratoses, solar lentigines, and lichen-planus-like keratoses. Benign.
- **BCC (Basal Cell Carcinoma):** Most common malignant skin cancer. Rarely metastasizes but locally destructive.
- **AKIEC (Actinic Keratoses):** Pre-malignant lesions and squamous cell carcinoma in situ (Bowen's disease).
- **VASC (Vascular Lesions):** Includes angiomas, angiokeratomas, pyogenic granulomas, and hemorrhage. Benign.
- **DF (Dermatofibroma):** Benign fibrous skin nodule. Rarest class in this dataset.

### Ground Truth CSV Format

All three splits use one-hot encoded CSVs:

```
image,MEL,NV,BCC,AKIEC,BKL,DF,VASC
ISIC_0024306,0.0,1.0,0.0,0.0,0.0,0.0,0.0
ISIC_0024307,0.0,1.0,0.0,0.0,0.0,0.0,0.0
```

Each row has exactly one `1.0` — single-label classification (no multi-label cases).

### Split Sizes and Class Distribution

| Class | Training | Validation | Test |
|-------|----------|-----------|------|
| NV | 6,705 | 123 | 909 |
| MEL | 1,113 | 21 | 171 |
| BKL | 1,099 | 22 | 217 |
| BCC | 514 | 15 | 93 |
| AKIEC | 327 | 8 | 43 |
| VASC | 142 | 3 | 35 |
| DF | 115 | 1 | 44 |
| **Total** | **10,015** | **193** | **1,512** |

**Note:** The validation set is very small (193 images) and the class proportions roughly mirror training. The test set (1,512 images) was the held-out challenge evaluation set — ground truth is now publicly available.

---

## Supplemental Data Files

### `Data/csv/lesion_groupings.csv`

Maps each of the 10,015 training images to its lesion group and diagnosis confirmation method.

**Columns:** `image`, `lesion_id`, `diagnosis_confirm_type`

**Diagnosis confirmation types:**

| Type | Count | Description |
|------|-------|-------------|
| histopathology | 5,340 | Gold standard — tissue biopsy with microscopic analysis |
| serial imaging showing no change | 3,704 | Lesion photographed over time; stability confirms benign diagnosis |
| single image expert consensus | 902 | Board-certified dermatologist agreement from single image |
| confocal microscopy with consensus dermoscopy | 69 | Reflectance confocal microscopy + dermoscopy expert agreement |

**Key insight for thesis:** Confidence in ground truth labels varies by confirm type. Histopathology-confirmed cases (~53%) are the most reliable. This stratification is useful for ablation studies (e.g., does XAI faithfulness differ by label quality?).

**Lesion groupings:** Multiple images can share a single `lesion_id` (same lesion photographed from different angles or at different times). This is critical for **data leakage prevention** — images of the same lesion must be kept in the same split.

### `Data/csv/train.csv`, `val.csv`, `test.csv`

One-hot encoded label files for each split.

**Path convention for code:**
```python
CSV_DIR = "Data/csv"
train_df  = pd.read_csv(f"{CSV_DIR}/train.csv")
val_df    = pd.read_csv(f"{CSV_DIR}/val.csv")
test_df   = pd.read_csv(f"{CSV_DIR}/test.csv")
groups_df = pd.read_csv(f"{CSV_DIR}/lesion_groupings.csv")
```



## Important Dataset Properties

### Image Characteristics
- **Format:** JPEG (input), PNG (masks)
- **Resolution:** Variable — dermoscopy images range from ~450×600 to ~4000×3000 pixels
- **Color space:** RGB (3-channel)
- **Content:** Close-up dermoscopic images with optional hair, skin artifacts, and dark frame corners from the dermoscope

### Known Challenges for Modeling
1. **Severe class imbalance:** NV at 67%, DF at 1.15% → use weighted cross-entropy or focal loss
2. **Lesion-level leakage:** Same lesion appears in multiple images → must split by lesion_id, not image_id
3. **Variable image size:** Requires resizing/padding to a fixed input size (typically 224×224 or 299×299)
4. **Dermoscopy artifacts:** Hair, ruler markings, ink spots, dark vignetting corners
5. **Label uncertainty:** ~37% of labels are from follow-up/consensus (not biopsy-confirmed)

### Relationship Between Tasks
```
Task 1-2 image pool (2,594 images)
    ├── Task 1: Is this pixel part of the lesion? (binary segmentation)
    └── Task 2: Does this pixel show globules/milia/etc.? (attribute segmentation)

Task 3 image pool (10,015 images)
    └── Task 3: What is the diagnosis? (7-class classification)

```

---

## What We Use for the Thesis

| Data | Location | Used for | Phase |
|------|----------|----------|-------|
| Classification images | `images/train`, `val`, `test` | Model training & evaluation | B |
| Diagnosis labels | `csv/` | Supervised learning targets | A, B |
| LesionGroupings CSV | `csv/` | Stratified split by lesion_id (no leakage) | A |
| Lesion masks | `plausibility/masks/` | L1 plausibility — does XAI focus on lesion? | E |
| Attribute masks | `plausibility/attributes/` | L2 plausibility — does XAI focus on clinical features? | E |

---

## EDA Checklist

Before training, the EDA notebook should verify:

- [ ] Image count per split matches expected counts (10015 / 193 / 1512)
- [ ] CSV row count matches image count (no orphaned labels)
- [ ] One-hot sanity check: each row sums to exactly 1.0
- [ ] Class distribution bar chart (highlight imbalance)
- [ ] Sample images per class (at least 3 per class)
- [ ] Image resolution distribution (histogram of H × W)
- [ ] Aspect ratio distribution
- [ ] Pixel intensity statistics (mean, std per channel — compute from training split only)
- [ ] Lesion groupings: how many unique lesion_ids? Average images per lesion?
- [ ] Confirmation type breakdown per class (which classes rely most on follow_up?)
- [ ] Check for corrupt/unreadable images
- [ ] Verify mask count matches plausibility image count (3,694 each)
- [ ] Attribute mask coverage — per image, how many of the 5 attributes are non-empty?
- [ ] Confirm all 4 CSVs load correctly from `Data/csv/`
