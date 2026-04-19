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
| Class-Balanced Focal Loss (γ=2) | ✅ B.2 | Replaces plain CE; lower variance per Paper 2 analysis |
| Test Time Augmentation (5×) | ✅ B.4 | H-flip, V-flip, 90°, 180° rotations averaged at inference |
| Val-tuned ensemble weights | ✅ B.6 | Weights ∝ each model's best val bAcc (not equal 0.25) |
| EfficientNet-B0 → B2 | ✅ B.1 | Stronger backbone (+9.1M params); same conv_head GradCAM |
| Mixup for minority classes | ✅ B.2 | Applied when batch contains AKIEC/DF/VASC samples |

---

## Comparative Analysis: Two ISIC 2018 Approaches

### Paper 1: Li & Li — Segmentation-Crop Strategy

**Core Philosophy:** Leverage Task 1 (Segmentation) to boost Task 3 (Classification) via lesion cropping.

**Cross-Task Transfer Learning Chain:**
1. Train ResNet50 on Task 3 (Classification) → skin-lesion-aware weights
2. Use those weights to initialize Mask R-CNN for Task 1 (Segmentation) → Jaccard improved 0.783 → **0.818**
3. Use improved Mask R-CNN to **crop backgrounds** from all Task 3 images
4. Re-train Classifier on cropped images only

**Result:** +2% improvement in normalized multi-class accuracy

**Key Risk:** Performance *"highly depends on the performance of the lesion boundary segmentation model."* If segmentation fails, classifier sees a cut-off lesion.

**Models Used:** ResNet50 (segmentation init), ResNet152, DenseNet201, Inception_V4
**Loss:** Class-Weighted CrossEntropy
**Ensemble:** None — compared single models via confusion matrix
**Best Reported:** **81.5%** Balanced Multi-Class Accuracy

---

### Detailed Comparison: Paper 1 (Li & Li) vs Paper 2 (Zhuang et al.)

#### Handling Data Imbalance

| Technique | Paper 1 (Li & Li) | Paper 2 (Zhuang et al.) |
|:---|:---|:---|
| **Class Weighting** | Yes. Explicitly uses class weights to balance dataset. | Yes (Extensive Analysis). Compared Class-Weighted CE vs Focal Loss. Class-Weighted gave higher absolute MCA; Focal Loss gave lower variance. |
| **Resampling** | Not mentioned. | Tried and Rejected. Neither undersampling nor oversampling brought obvious improvement — sometimes worse. |
| **Feature Space Manipulation** | Not used. | Experimental Only. Tried Triplet Loss and Contrastive Loss (Siamese). Failed — difficulty mining hard pairs in this domain (unlike faces). |

#### Model Architecture & Ensemble

| Aspect | Paper 1 (Li & Li) | Paper 2 (Zhuang et al.) |
|:---|:---|:---|
| **Base Models** | ResNet50, ResNet152, DenseNet201, Inception_V4 | **SENet-154** and **PNASNet-5-Large** |
| **Ensemble Logic** | No Ensemble. Individual models compared separately. | **Weighted Average Ensemble.** Formula: Σ wᵢsᵢ. Slightly better than direct average or hard voting. |
| **Why Specific Models?** | Common ImageNet backbones for reproducibility. | SENet/PNASNet chosen because they *"exhibit powerful abilities in learning features in the case of insufficient training data."* |

#### Segmentation-First vs End-to-End

| Criteria | Paper 1 (Crop Strategy) | Paper 2 (Ensemble Strategy) |
|:---|:---|:---|
| **Inference stages** | Two-stage (Seg → Class) | Single-stage forward pass |
| **Error propagation** | High — seg failure → class failure | Low |
| **Computational cost** | High prep (train 2 models); low inference | High inference (2 giant models per image) |
| **Modern relevance** | Low-Medium (Mask R-CNN + ResNet152 = 2018 standard) | Medium-High (SENet attention ≈ modern ViT logic) |
| **Performance ceiling** | ~80–85% range | **93.1%** (Winner-tier 2018) |

---

## Hybrid Plan: Targeting 80–85% Balanced Accuracy

**Philosophy:** Borrow Paper 2's single-stage principle (no separate segmentation model) but apply Paper 1's insight that background noise matters — using **Random Resized Crop** as a stochastic, cheap approximation of explicit lesion cropping.

### Component Decisions

| Component | Choice | Rationale |
|:---|:---|:---|
| **Backbones** | EfficientNet-B2, DenseNet121, ViT-Base, Swin-Tiny | Swin/ViT attention suppresses background (Paper 2 logic) without separate segmentation |
| **Loss** | Class-Balanced Focal Loss (γ=2) | Paper 2: lower variance across folds → more stable minority-class recall |
| **Augmentation** | Random Resized Crop + Mixup (minority only) | Cheap stochastic approximation of Paper 1's "crop the lesion" + synthetic signal for DF/VASC/AKIEC |
| **Ensemble** | 4-model weighted average, weights ∝ val bAcc | Middle ground: Paper 1's "no ensemble" is wasteful; Paper 2's manual tuning is replaced by automatic val-set derivation |
| **TTA** | 5× (H-flip, V-flip, 90°, 180° rotations) | +3–6% bAcc with zero retraining cost |

### Expected Gains by Component

| Priority | Change | Requires Retraining | Expected bAcc Gain |
|:---|:---|:---|:---|
| 1 | TTA (5× flips/rots) | No | +3–6% |
| 2 | Val-tuned ensemble weights | No | +1–3% |
| 3 | Class-Balanced Focal Loss (γ=2) | Yes | +1–4% |
| 4 | EfficientNet-B0 → B2 | Yes | +1–3% |
| 5 | Mixup for minority classes | Yes | +1–2% |
