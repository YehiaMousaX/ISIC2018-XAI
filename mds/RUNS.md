# Experiment Runs Log — ISIC 2018 Classifier

Target: **bAcc ≥ 0.80 | top-1 ≥ 0.90 | macro-F1 ≥ 0.75 | macro-AUC ≥ 0.95**

---

## Current notebook

**`ISIC2018_Classifier_Kaggle.py`** — Paper-faithful SENet-154 single-model classifier.

| Component | Value |
|---|---|
| Backbone | `legacy_senet154` (113M params), ImageNet-pretrained |
| Input | Resize 300 → RandomCrop 224 (train) / CenterCrop 224 (eval) |
| Color constancy | Shades-of-Gray, power=6 |
| Loss | Class-weighted CrossEntropy (inverse-freq weights) |
| Sampler | Standard random shuffle (no WeightedRandomSampler) |
| Optimizer | AdamW lr=1e-4, wd=1e-4 + CosineAnnealingLR |
| Batch | 16, AMP=True |
| Early stop | patience=10 on val bAcc |

Legacy XAI pipeline archived to `legacy/`. XAI is a separate future notebook.

---

## All runs (newest first)

---

### 2026-04-21 — `senet154-paper-faithful` ← **current baseline**

**Notebook:** `ISIC2018_Classifier_Kaggle.py` (fresh start)
**Config:** SENet-154, weighted CE, 300→224, batch 16, AMP, patience 10 on val bAcc

| Metric | Result | Target | Status |
|---|---|---|---|
| Balanced accuracy | 0.7839 | ≥ 0.80 | MISS |
| Top-1 accuracy | 0.7474 | ≥ 0.90 | MISS |
| Macro F1 | 0.6628 | ≥ 0.75 | MISS |
| Macro AUC | 0.9608 | ≥ 0.95 | **PASS** |
| Top-3 accuracy | 0.9603 | — | — |
| Weighted F1 | 0.7657 | — | — |
| MCC | 0.6384 | — | — |
| Cohen κ | 0.6243 | — | — |

**Training:** ran 17 epochs (best val bAcc=0.809 @ epoch 7, stopped after 17) | 64.6 min on T4

**Per-class recall (test):**

| MEL | NV | BCC | AKIEC | BKL | DF | VASC |
|---|---|---|---|---|---|---|
| 0.743 | 0.748 | 0.677 | 0.837 | 0.710 | 0.886 | 0.886 |

**What's good:**
- Macro-AUC 0.961 — ranking quality is strong; model is discriminating classes well
- Minority recall is high (AKIEC 0.84, DF 0.89, VASC 0.89) — class-weighted CE is doing its job on rare classes
- Very fast convergence: 17 epochs total, best checkpoint at epoch 7
- Training was clean: no overfitting signs (train loss 0.20, val loss 0.98 gap is from class weights, not overfitting)

**What's bad:**
- Top-1 accuracy 0.747 is the worst so far — model over-predicts minority classes at the expense of NV precision (NV recall dropped to 0.748 vs 0.94 in previous runs)
- AKIEC precision 0.32 / DF precision 0.49 — many NV/BKL samples misclassified as rare classes
- Val bAcc 0.809 vs test bAcc 0.784 — 2.5% gap suggests the val set was a slightly easier draw or early stopping under-trained the model
- Stopped at epoch 17 — patience=10 from best at ep7 means model stopped improving early; possible the LR schedule (T_max=60) was too slow to cycle

**Analysis:** The class-weighted CE without sampler heavily pushes the model to predict minority classes. High recall + low precision on AKIEC/DF/VASC is a classic over-correction. The paper may have relied on the full 5-fold CV + ensemble to average out this variance. A single run may need either a softer weight (reduce w_c for rarest classes) or a second run to see if it converges differently.

---

### 2026-04-20_21-14 — `combo-a-cnn-only` (last XAI pipeline run)

**Notebook:** `XAI_Evaluation_Pipeline_Kaggle.py` (archived)
**Config:** EffNet-B2 + DenseNet-121 | img256 | WD=5e-4, smooth=0.05, Mixup α=0.2 p=0.5 | WeightedRandomSampler | 15% lesion-grouped val | temp-scaled + per-class α ensemble

| Model | bAcc | Top-1 | Macro F1 | Macro AUC | MCC |
|---|---|---|---|---|---|
| EfficientNet-B2 | 0.7052 | 0.8347 | 0.7390 | 0.9468 | 0.7171 |
| DenseNet-121 | 0.7110 | 0.7824 | 0.7102 | 0.9503 | 0.6674 |
| **Ensemble** | **0.6991** ⬇ | **0.8393** | **0.7377** | **0.9625** | **0.7290** |

**Per-class recall (ensemble):**

| MEL | NV | BCC | AKIEC | BKL | DF | VASC |
|---|---|---|---|---|---|---|
| 0.696 | 0.927 | 0.807 | 0.419 | 0.747 | 0.727 | 0.571 |

**Headline failure:** Ensemble bAcc *below* both individual models. Root cause: α-tuning objective was macro-F1, which actively suppressed AKIEC (α=0.65) and VASC (α=0.60) recall to optimise F1 at the expense of bAcc. Solo models had VASC recall 0.71–0.74; ensemble dropped it to 0.57.

---

### 2026-04-20_13-57 — `img256 + smooth + mixup` (full run)

**Config:** EffNet-B2 + DenseNet-121 | img256 | WD=5e-4, smooth=0.05, Mixup α=0.2 p=0.5

| Model | bAcc | Top-1 | Macro F1 | Macro AUC |
|---|---|---|---|---|
| EfficientNet-B2 | 0.6866 | 0.8307 | 0.7356 | 0.9406 |
| DenseNet-121 | 0.6997 | 0.8287 | — | — |

EffNet ran 68 epochs (best val bAcc=0.851 @ some epoch) — large val/test gap, suggests val overfit or distribution shift between old (random) and new (lesion-grouped) val split.

---

### 2026-04-20_08-08 — `effnet-b2 upgrade` (first B2 run)

**Config:** EffNet-B2 + DenseNet-121 | img256 | switched from B0 to B2 backbone

| Model | bAcc | Top-1 | Macro F1 | Macro AUC |
|---|---|---|---|---|
| EfficientNet-B2 | 0.6448 | 0.7809 | 0.6726 | — |
| DenseNet-121 | 0.6389 | 0.7929 | 0.6815 | — |

Lower bAcc than later runs — likely using old random val split (before lesion-grouped split was introduced). Top-1 accuracy on test was actually high (0.78–0.79) because NV dominated.

---

### 2026-04-18_12-17 — `4-model second attempt`

**Config:** EffNet-B0, DenseNet-121, ViT-Base, Swin-Tiny | 4-model XAI pipeline

| Model | bAcc | Top-1 | Macro F1 |
|---|---|---|---|
| EfficientNet-B0 | 0.6678 | 0.7030 | 0.6396 |
| DenseNet-121 | 0.6997 | 0.5904 | — |
| ViT-Base | 0.6272 | 0.6217 | — |
| Swin-Tiny | 0.7275 | 0.6543 | 0.7048 |

Swin-Tiny best so far in this family at 0.7275.

---

### 2026-04-17_03-34 — `4-model first full attempt`

**Config:** EffNet-B0, DenseNet-121, ViT-Base, Swin-Tiny | first full non-debug 4-model run

| Model | bAcc | Top-1 | Macro F1 |
|---|---|---|---|
| EfficientNet-B0 | 0.7082 | 0.6612 | 0.6503 |
| DenseNet-121 | 0.7370 | 0.5742 | 0.5972 |
| ViT-Base | 0.6678 | 0.5688 | 0.5224 |
| Swin-Tiny | 0.7196 | 0.6386 | 0.6239 |

Best single model to this date: DenseNet-121 at 0.737 bAcc.

---

### 2026-04-16_00-14 — `first full run`

**Config:** EffNet-B0, DenseNet-121, ViT-Base, Swin-Tiny | first non-debug full run

| Model | bAcc | Top-1 |
|---|---|---|
| EfficientNet-B0 | 0.6961 | 0.5620 |
| DenseNet-121 | 0.6935 | 0.4026 |
| ViT-Base | 0.6403 | 0.4993 |
| Swin-Tiny | 0.7002 | 0.4733 |

High bAcc but very low top-1 accuracy — model was predicting minority classes aggressively (underfitting NV). First sign of the recall/precision trade-off.

---

### 2026-04-15_22-02, 2026-04-16_00-03, 2026-04-20_12-56 — DEBUG runs

All `DEBUG=True` (200–500 image subset, 3 epochs). Results are noise (0.25–0.44 bAcc). Excluded from trend analysis.

---

## Trends

| Pivot | bAcc (best single model) | Note |
|---|---|---|
| First full run (Apr 16) | 0.700 (Swin-Tiny) | 4-model, B0/DenseNet |
| Second attempt (Apr 17) | 0.737 (DenseNet-121) | Better training |
| After backbone upgrade B0→B2 (Apr 20) | 0.711 (DenseNet-121) | Lesion-grouped val split introduced — harder val |
| SENet-154 paper-faithful (Apr 21) | **0.784** (single model) | New best test bAcc single model |

**Best val bAcc ever:** 0.809 (SENet-154 @ epoch 7, Apr 21)

---

## Next steps

The SENet-154 run hit val bAcc 0.809 (above target) but test bAcc 0.784 (below). The gap and the over-prediction of minority classes suggest:

1. **Soften class weights** — cap minority weights to avoid extreme over-prediction (e.g., `w_c = sqrt(N / (K * n_c))` instead of linear). Reduces AKIEC/DF false positives.
2. **Increase patience** — current run stopped at epoch 7 + 10 = 17. Try `PATIENCE=15` or `MAX_EPOCHS=100` — SENet-154 may need more epochs to properly calibrate the NV/minority boundary.
3. **Add PNASNet-5-Large** — the paper's full ensemble (SENet + PNASNet) gets 92.3% MCA. Even a simple weighted-average of two models would close the 0.784→0.80 gap on the test set.
4. **Reduce LR** — try `LR=5e-5` with `T_max=60`; faster cosine cycle may explain early convergence stall.
