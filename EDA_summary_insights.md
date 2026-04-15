# ISIC 2018 — EDA Summary & Insights

**Thesis:** A Multi-Dimensional Evaluation of Explainability Methods Across CNN and Vision Transformer Architectures for Skin Lesion Classification
**Generated:** April 2026

---

## A — Data Integrity

**Result: 17 / 18 checks passed — dataset is clean.**

All substantive integrity checks pass across every split:

- No corrupt images detected in train, val, or test
- All CSV row counts match the image file counts exactly
- One-hot label sums are valid for every row (exactly one class active per image)
- All image files referenced in CSVs exist on disk
- All 3,694 segmentation masks pair exactly to their corresponding images
- All 5 attribute mask types are present (`globules`, `milia_like_cyst`, `negative_network`, `pigment_network`, `streaks`)
- Total attribute files confirmed: 16,104

The single "fail" is informational — it is the numeric count of attribute files (16,104), not a real failure condition.

**Pipeline implication:** No filtering or file-repair step is needed before training. Proceed directly to Step A.3.

---

## B — Class Distribution & Imbalance

**Finding: Severe imbalance — maximum ratio of 58.3× (DF vs NV).**

### Training split (10,015 images)

| Class | Count | % of train | Imbalance vs NV |
|-------|-------|-----------|-----------------|
| NV (Melanocytic Nevus) | 6,705 | 66.9% | 1.0× (majority) |
| MEL (Melanoma) | 1,113 | 11.1% | 6.0× |
| BKL (Benign Keratosis) | 1,099 | 11.0% | 6.1× |
| BCC (Basal Cell Carcinoma) | 514 | 5.1% | 13.0× |
| AKIEC (Actinic Keratosis) | 327 | 3.3% | 20.5× |
| VASC (Vascular Lesion) | 142 | 1.4% | 47.2× |
| DF (Dermatofibroma) | 115 | 1.1% | **58.3×** |

### Label entropy by split (max = log₂(7) ≈ 2.81 bits)

| Split | Entropy |
|-------|---------|
| Train | 1.63 bits |
| Val | 1.73 bits |
| Test | 1.87 bits |

All splits are dominated by NV, but the class distributions are broadly consistent (Jensen-Shannon divergence between train and test = 0.005 — negligible shift in label space).

### Insights

**The 58.3× imbalance ratio is severe.** Two countermeasures are already planned in Step A.5 and A.6:

1. `WeightedRandomSampler` — rebalances each training batch at draw time
2. Class-weighted `CrossEntropyLoss` — scales the loss contribution of minority classes

**However, the EDA recommends upgrading to Focal Loss.** At a 58× imbalance, even weighted CE can fail to adequately penalise easy-majority misclassifications. Focal Loss adds a modulating factor `(1 − p)^γ` that further down-weights easy examples (high-confidence NV predictions), focusing training on hard minority cases. This is particularly important for DF (115 images) and VASC (142 images), whose XAI explanations will only be meaningful if the classifier learns them reliably.

> **Action in Step B.2:** Replace `nn.CrossEntropyLoss(weight=weights)` with a Focal Loss implementation, or use a library such as `torchvision.ops.sigmoid_focal_loss` adapted for multiclass.

---

## C — Lesion Groupings & Label Quality

**Finding: Patient-level leakage risk exists and cannot be fully removed from the official splits.**

| Metric | Value |
|--------|-------|
| Unique lesion IDs (unique patients/lesions) | 7,470 |
| Lesions with more than one image | 26.2% |
| Mean images per lesion | 1.34 |
| Maximum images per lesion | 6 |

Because the official ISIC 2018 train/test split is fixed and provided pre-divided, re-splitting is not possible without invalidating the benchmark. However, ~26% of lesions have multiple images, and some of those images appear in both the training and test sets. This inflates test performance by an unknown amount — the model has effectively seen the same patient's skin under slightly different conditions.

### Label quality breakdown

| Confirmation method | % of training labels |
|---------------------|----------------------|
| Histopathology (gold standard) | 53.3% |
| Serial imaging showing no change | 37.0% |
| Single-image expert consensus | 9.0% |
| Confocal microscopy with consensus dermoscopy | 0.7% |

**46.7% of training labels are not histopathology-confirmed.** Serial imaging and expert consensus labels may introduce label noise, particularly for ambiguous lesions. This could affect model calibration and, downstream, the reliability of XAI faithfulness metrics on those images.

### Insights

- The `lesion_id` column is already attached to `train_df` in Step A.3 for leakage tracking — document this in §5.1 of the thesis as a known limitation of using the official benchmark.
- Run a **confirmation-type sensitivity analysis** in Phase H: compare per-class XAI metric distributions separately for histopathology-confirmed vs non-confirmed images. If plausibility or faithfulness scores differ substantially, the label quality is confounding the results.

---

## D & E — Image Quality & Resolution

**Finding: No quality issues requiring intervention. Minor sharpness shift between train and test.**

### Normalisation statistics (from 2,000-image train sample)

| Channel | Mean | Std |
|---------|------|-----|
| R | 0.765 | — |
| G | 0.544 | — |
| B | 0.568 | — |

These are the values used in `A.Normalize()` in Steps A.5 and A.6. Expected values at inference: mean ≈ [0.76, 0.55, 0.57], std ≈ [0.14, 0.16, 0.18].

### Quality metrics by split

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Brightness | 0.613 | 0.620 | 0.620 |
| Contrast | 0.110 | 0.101 | 0.116 |
| Sharpness | 630.3 | 607.2 | **700.1** |

### Resolution

- All images have aspect ratio 1.33 (4:3), with zero standard deviation — perfectly uniform
- Upsample fraction to 224px: **0.0%** — every image is larger than 224px, so resizing is purely downsampling
- Direct resize to 224×224 is safe with no aliasing concern

### Insights

Test images are measurably sharper than training images (sharpness 700 vs 630, Cohen's d ≈ −0.21, p < 0.001). This is a **distribution shift in image quality**, not in label space. It has two implications:

1. Grad-CAM and HiResCAM heatmaps on the test set may appear more spatially precise than they would on in-distribution data, because the underlying feature maps are sharper.
2. This could marginally inflate plausibility (IoU/Dice) scores — mention this caveat when reporting Phase E results in §5.4.2.

> **No corrective action needed**, but note the sharpness shift as a limitation in the thesis.

---

## F — Artifact Burden

**Finding: Hair artifacts are the dominant confounder, particularly in test images. MEL has the highest overall artifact load.**

### Mean artifact scores by class (training set, normalised)

| Class | Hair (norm.) | Highlights (norm.) | Dark artifacts (norm.) | Blur score |
|-------|------------|-------------------|----------------------|------------|
| MEL | **1.00** | 0.74 | **1.00** | 258 |
| DF | 1.00 | **1.00** | 0.03 | **1000** |
| BKL | 0.86 | 0.04 | 0.54 | 136 |
| AKIEC | 0.79 | 0.43 | 0.42 | 189 |
| BCC | 0.59 | 0.15 | 0.10 | 289 |
| NV | 0.44 | 0.86 | 0.24 | 296 |
| VASC | 0.00 | 0.00 | 0.00 | 0.00 |

### Distribution shift: train → test (hair artifacts)

Hair score shift: Cohen's d = −0.208, p < 0.001. Test images have **more hair** than training images.

### Insights

- **MEL** has the highest combined hair + dark artifact burden. When Grad-CAM or LIME heatmaps for MEL highlight hair stripes or dark artifact edges rather than the lesion itself, it is not necessarily a model failure — it may reflect a genuine training signal (dermatologists are also trained to look around hair for lesion boundaries). Report XAI plausibility for MEL with this caveat.
- **DF** images are unusually sharp (blur_score = 769, normalised to 1.0). This means Grad-CAM on DF may produce sharper, more localised heatmaps not because the method is better, but because the input features are crisper.
- Standard augmentation (flip, rotate, colour jitter) is sufficient — the EDA does not recommend hair simulation augmentation despite the elevated test hair score, because the effect size is medium and the model's exposure to hair during training is already non-trivial.

---

## G — Plausibility Coverage

**Finding: Full mask coverage for L1 evaluation. Attribute coverage is uneven — pigment network dominates.**

### L1 coverage (coarse — binary lesion mask)

| Metric | Value |
|--------|-------|
| Images with lesion segmentation mask | 3,694 |
| Images with any attribute mask | 3,677 |
| Images with at least one non-empty attribute | 2,671 |

### L2 coverage (fine-grained — attribute masks)

| Attribute | Non-empty rate | Count |
|-----------|---------------|-------|
| Pigment network | **52.8%** | 1,950 |
| Milia-like cysts | 19.7% | 728 |
| Globules | 19.6% | 725 |
| Negative network | 6.6% | 244 |
| Streaks | 4.1% | 152 |

### Lesion size distribution

| Metric | Value |
|--------|-------|
| Mean lesion area fraction | 23.3% of image |
| Median lesion area fraction | 16.5% |
| Lesions covering < 5% of image | **19.14%** |

### Insights

**The 19.14% small-lesion fraction is a critical confound for plausibility metrics.** A heatmap that is correctly centred on a lesion covering 3% of the image will still score near-zero IoU if the binarisation threshold produces a heatmap blob larger than the tiny lesion, or if there is any spatial offset. This is a property of the metric, not the XAI method.

> **Action in Phase E:** Stratify all plausibility results (IoU, Dice, Pointing Game) into two groups: small lesions (area < 5% of image) and standard lesions (≥ 5%). Report separately and pool only when presenting overall summaries.

For L2 attribute-level evaluation, **pigment network is the only attribute with sufficient N (1,950) for reliable statistical comparisons**. Streaks (n = 152) and negative network (n = 244) are too sparse for per-class breakdowns — either aggregate them into an "other" group or report as exploratory only with confidence intervals.

---

## H — Duplicate Risk

**Finding: Minimal. 4 exact duplicates (within-split only), 5 near-duplicate train↔test pairs.**

### Exact duplicates

- Total: 4 images
- Cross-split (train appearing in test): 0

### Near-duplicate train↔test pairs (Hamming distance = 4 on perceptual hash)

| Train image | Test image | Hamming distance |
|-------------|-----------|-----------------|
| ISIC_0024352 | ISIC_0035065 | 4 |
| ISIC_0024375 | ISIC_0035614 | 4 |
| ISIC_0024404 | ISIC_0035157 | 4 |
| ISIC_0024541 | ISIC_0034917 | 4 |
| ISIC_0024594 | ISIC_0035343 | 4 |

### Insights

The 5 near-duplicate pairs (Hamming = 4, meaning 4 of 64 hash bits differ) represent visually very similar images. At 5 / 1,512 test images = **0.33%** of the test set, the impact on any aggregate metric is negligible. However, these pairs are worth one sentence in §5.1 as a known data quality note, and the near-duplicate test images should be excluded from any qualitative case-study examples shown in the thesis to avoid giving the impression that the model has been evaluated on its own training data.

---

## I — Distribution Shift Between Splits

**Finding: 8 of 14 comparisons show statistically significant shift. The val → test contrast gap is the largest effect.**

| Comparison | Feature | Cohen's d | p-value | Significance |
|------------|---------|-----------|---------|-------------|
| val → test | contrast | −0.337 | < 0.001 | significant |
| val → test | sharpness | −0.298 | < 0.001 | significant |
| train → test | sharpness | −0.210 | < 0.001 | significant |
| train → test | hair score | −0.208 | < 0.001 | significant |
| train → test | contrast | −0.141 | < 0.001 | significant |
| train → test | highlight score | −0.030 | 0.008 | significant |
| train → test | brightness | −0.091 | 0.018 | significant |
| val → test | brightness | +0.006 | 0.444 | not significant |
| all pairs | aspect ratio | 0.000 | 1.000 | not significant |

All shifts are small-to-medium in effect size (no d > 0.4). Aspect ratio is perfectly identical across splits.

### Insights

The val → test contrast gap (d = −0.337) is the largest shift in the dataset. This means the validation set's contrast distribution is noticeably different from the test set — which is relevant for training decisions: a model that is tuned to perform well on the validation set (via early stopping on val_loss) may encounter a slightly different image regime at test time.

For the XAI evaluation specifically, the sharpness and hair shifts (train → test, d ≈ 0.21) can produce **systematically sharper and more hair-confounded heatmaps on the test set** than would be produced on true in-distribution images. This should be mentioned when interpreting plausibility and robustness results.

> **No corrective action needed.** All effect sizes are modest. The validation loss is still a valid early stopping criterion. Document the shift in §5.1 as a dataset property.

---

## J — Actionable Decisions Summary

| Decision | Recommendation | Priority | Evidence |
|----------|---------------|----------|---------|
| Loss function | Use Focal Loss instead of plain weighted CE | **High** | 58.3× max imbalance ratio |
| Augmentation | Standard augmentation is sufficient | Low | Hair shift d = 0.208 (medium, not extreme) |
| Plausibility evaluation | Stratify results by lesion size | **High** | 19.1% of masks have lesion area < 5% |
| Label quality | Run sensitivity analysis by confirmation type | Medium | 46.7% of train labels are non-histopathology |
| Data cleaning | No filtering needed | — | Zero corrupt images detected |
| Resize policy | Direct resize to 224px is safe | — | All originals > 224px, 0% upsampling |
| Near-duplicates | Exclude from qualitative examples | Low | 5 pairs = 0.33% of test set |
| Distribution shift | Document, do not correct | Low | Max Cohen's d = 0.337 (moderate) |

---

## Overall Readiness Assessment

The ISIC 2018 dataset is well-suited for this thesis pipeline with no blocking issues. The key pre-training decisions are:

1. **Focal Loss** — replace weighted CE in Step B.2 to handle the 58.3× imbalance.
2. **Lesion-size stratification** — add a `lesion_size_group` column to the plausibility subset before Phase E and report metrics separately for small (< 5%) and standard (≥ 5%) lesions.
3. **Confirmation-type column** — ensure `diagnosis_confirm_type` is available in the eval DataFrames for the Phase H sensitivity analysis.
4. **Pigment network as primary L2 attribute** — use it as the main fine-grained plausibility signal; treat other attributes as supplementary.
5. **Document, don't fix** — the sharpness shift, hair shift, lesion leakage risk, and near-duplicate pairs are all minor enough that they require thesis documentation, not pipeline intervention.
