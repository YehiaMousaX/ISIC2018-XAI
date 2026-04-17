Here is a **more specific, implementation-level one-page cheat sheet**, tightly grounded in what the paper actually used (no generalization, no guessing).

---

# ISIC 2018 — Exact Method Cheat Sheet (Paper-Specific)

---

## Task

* 7-class skin lesion classification
* Metric: Multi-Class Accuracy (MCA)

---

## Dataset Imbalance (Exact)

| Class | Samples |
| ----- | ------- |
| NV    | 6705    |
| MEL   | 1113    |
| BKL   | 1099    |
| BCC   | 514     |
| AKIEC | 327     |
| VASC  | 143     |
| DF    | 115     |

* Max/min ratio ≈ **58:1**

---

## Final Models Used

### 1. SENet-154

* Input:

  * Resize: 300×300
  * Crop: 224×224
* Pretrained on ImageNet

---

### 2. PNASNet-5-Large

* Input:

  * Resize: 441×441
  * Crop: 331×331
* Pretrained on ImageNet

---

## Training Strategy

* Train **each model separately**
* Stop training **before overfitting**
* Use model outputs on validation/test
* Combine predictions via ensemble

---

## Ensemble (Final)

### Weighted Average

[
FinalScore = \sum w_i s_i
]

* ( s_i ): 7-dim probability vector per model
* ( w_i ): manually tuned weights
* Constraint: ( \sum w_i = 1 )

---

## Preprocessing (Exact)

### 1. Color Constancy

* Normalize illumination and color variations

---

### 2. Data Augmentation

Applied randomly:

* Horizontal flip
* Vertical flip
* Rotation: -180° to +180°
* Brightness adjustment
* Contrast adjustment
* Saturation adjustment
* Affine transformations
* Random crop

---

## Core Fix for Imbalance

### Class-Weighted Cross Entropy (USED)

* Assign higher weights to minority classes
* Penalize misclassification of rare classes more

**Effect observed:**

* ≥10% improvement in MCA
* Better confusion matrix (minority classes predicted more)

---

## Alternative Tried

### Focal Loss

* More stable (lower variance)
* Not selected (lower MCA than weighted CE)

---

## Evaluation Method

* **5-fold cross-validation**
* Compared:

  * Standard loss
  * Weighted loss
  * Focal loss

---

## What They Explicitly Tried and Rejected

### 1. Oversampling / Undersampling

* No consistent improvement
* Sometimes worse

---

### 2. Triplet Loss / Contrastive Loss

* Failed due to lack of meaningful sample relationships

---

### 3. Clustering Majority Class

* Split large class into smaller groups
* No improvement

---

### 4. Teacher–Student (Distillation)

* No significant gain

---

### 5. CNN Features + SVM

* ~70% MCA
* Much worse than end-to-end CNN

---

## Final Performance

| Model                      | MCA   |
| -------------------------- | ----- |
| PNASNet                    | 88.7% |
| SENet                      | 89.8% |
| Ensemble SENet             | 91.7% |
| Ensemble (SENet + PNASNet) | 92.3% |

Final reported: **93.1%**

---

## Minimal Reproduction Recipe (Paper-Accurate)

1. Use:

   * SENet-154
   * PNASNet-5-Large
     (ImageNet pretrained)

2. Preprocess:

   * Apply color constancy
   * Apply full augmentation set

3. Train:

   * Use **class-weighted cross entropy**
   * Apply early stopping

4. Validate:

   * 5-fold cross-validation

5. Inference:

   * Get probability vectors from each model

6. Combine:

   * Weighted average of outputs

---

## Key Takeaway (Strictly from Paper)

* **Only method that significantly solved imbalance:**
  → Class-weighted cross entropy

* Everything else:
  → marginal, unstable, or ineffective

---

## Drop-in Prompt (Strict Version)

"Implement ISIC-style imbalanced classification using:

* SENet-154 and/or PNASNet-5-Large (ImageNet pretrained)
* Input resizing: 300→224 (SENet), 441→331 (PNASNet)
* Heavy augmentation (flip, rotate, color jitter, affine, crop)
* Class-weighted cross entropy (primary imbalance solution)
* 5-fold cross-validation
* Weighted ensemble of model outputs

Avoid:

* Oversampling/undersampling
* Metric learning losses (triplet/contrastive)
* Clustering-based class splitting
* Distillation unless justified"

---

## Applied to XAI Pipeline (`XAI_Evaluation_Pipeline_Kaggle.py`)

| Technique | Applied | Notes |
|---|---|---|
| Color constancy (Shades of Gray p=6) | ✅ A.2 + A.5 + A.6 | Applied before all transforms and stat computation |
| Class-weighted CrossEntropy | ✅ B.2 | Replaced Focal Loss; paper: ≥10% MCA gain |
| Full augmentation (flip, ±180° rot, jitter, affine, crop) | ✅ A.6 | Expanded from ±15° to ±180° rotation |
| Weighted ensemble | ✅ B.6 | Equal weights by default; tune on val set |
| SENet-154 / PNASNet-5-Large | ❌ | Kept EfficientNet/DenseNet/ViT/Swin for XAI comparability |
| 5-fold cross-validation | ❌ | Pre-split dataset used; single fixed train/val/test |
