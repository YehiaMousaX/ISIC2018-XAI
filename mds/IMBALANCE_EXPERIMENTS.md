# Imbalance Strategy Experiments

Comparing two strategies for handling the ISIC 2018 class imbalance (NV ≈ 67% of train, imbalance ratio ≈ 58x).

Metric tracked: **macro-F1** and **per-class recall** (not accuracy — NV dominance masks minority failures).

---

## Strategy Definitions

### Combo A — Sampling-based
| Component | Setting |
|---|---|
| Sampler | `WeightedRandomSampler` (weight ∝ inverse class freq) |
| Loss | `nn.CrossEntropyLoss` (no class weights) |
| Augmentation | Full pipeline (flips, rotate ±180°, ColorJitter, Affine, RandomCrop) |
| Mixup | Disabled |
| Rationale | Balance batches at the data level; keep loss simple and interpretable |

### Combo B — Loss-based + regularisation
| Component | Setting |
|---|---|
| Sampler | Standard shuffle |
| Loss | `ClassBalancedFocalLoss` (γ=2, inverse-freq weights) |
| Augmentation | Full pipeline |
| Mixup | Enabled for batches containing minority classes (α=0.2) |
| Rationale | Focal loss focuses on hard/minority examples; Mixup adds synthetic minority signal |

---

## Results

### Combo A — WeightedRandomSampler + CrossEntropyLoss

*Paste results here after the run.*

```
EfficientNet-B2 bAcc = 
DenseNet-121    bAcc = 
ViT-Base/16     bAcc = 
Swin-Tiny       bAcc = 
Weighted Ensemble bAcc = 
```

**Per-class recall (paste classification report):**

```
              precision    recall  f1-score

         MEL
          NV
         BCC
       AKIEC
         BKL
          DF
        VASC
```

---

### Combo B — Focal Loss + Mixup (baseline, commit `380ec89`)

*Paste results here after the run — or copy from the last experiment output.*

```
EfficientNet-B2 bAcc = 
DenseNet-121    bAcc = 
ViT-Base/16     bAcc = 
Swin-Tiny       bAcc = 
Weighted Ensemble bAcc = 
```

**Per-class recall:**

```
              precision    recall  f1-score

         MEL
          NV
         BCC
       AKIEC
         BKL
          DF
        VASC
```

---

## Comparison

| Metric | Combo A | Combo B | Winner |
|---|---|---|---|
| Ensemble bAcc | — | — | — |
| MEL recall | — | — | — |
| NV recall | — | — | — |
| BCC recall | — | — | — |
| AKIEC recall | — | — | — |
| BKL recall | — | — | — |
| DF recall | — | — | — |
| VASC recall | — | — | — |
| macro-F1 | — | — | — |

**Conclusion:** *(fill after both runs)*

---

## Interpretation Guide

- If **Combo A wins** → imbalance was the primary bottleneck; sampling correction was sufficient.
- If **Combo B wins** → hard-example mining and loss regularisation mattered more than batch balance.
- If minority recall (DF, VASC, AKIEC) is still low in both → consider two-stage training or per-class threshold tuning.
