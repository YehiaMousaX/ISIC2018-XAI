# Experiment Runs Log — ISIC 2018 Classifier

**Targets:** bAcc ≥ 0.80 | top-1 ≥ 0.90 | macro-F1 ≥ 0.75 | macro-AUC ≥ 0.95

---

## Headline results

| Run | Date | bAcc | top-1 | macro-F1 | macro-AUC | Epochs | Notes |
|---|---|---|---|---|---|---|---|
| **senet154-v3** (planned) | — | — | — | — | — | — | freeze-then-unfreeze + dropout + stronger WD |
| senet154-v2-sqrt-weights | 2026-04-22 | 0.7570 | 0.8016 | 0.7297 | 0.9691 | 23 (best@8) | sqrt weights fixed top-1/F1; overfitting killed bAcc after ep8 |
| senet154-v1-paper-faithful | 2026-04-21 | 0.7839 | 0.7474 | 0.6628 | 0.9608 | 17 (best@7) | best test bAcc so far; oscillation from 59× weight ratio |

---

## Per-class recall — key runs (test set)

| Run | MEL | NV | BCC | AKIEC | BKL | DF | VASC |
|---|---|---|---|---|---|---|---|
| senet154-v3 | — | — | — | — | — | — | — |
| senet154-v2 | 0.795 | 0.823 | 0.742 | 0.512 | 0.797 | 0.773 | 0.857 |
| senet154-v1 | 0.743 | 0.748 | 0.677 | 0.837 | 0.710 | 0.886 | 0.886 |

---

## Config diff — v1 → v2 → v3

| Param | v1 | v2 | v3 (planned) |
|---|---|---|---|
| Class weights | linear inv-freq (59×) | sqrt inv-freq (7.7×) | sqrt inv-freq (keep) |
| `COSINE_T_MAX` | 60 | 30 | 30 (keep) |
| `PATIENCE` | 10 | 15 | 15 (keep) |
| `MAX_EPOCHS` | 60 | 80 | 80 (keep) |
| Backbone freeze | none | none | **freeze 5 ep, then unfreeze** |
| Dropout | none | none | **drop_rate=0.3** |
| Weight decay | 1e-4 | 1e-4 | **1e-3** |

---

## v2 diagnosis — why it regressed on bAcc despite better top-1/F1

**What got better:** top-1 0.747→0.802, macro-F1 0.663→0.730, macro-AUC 0.961→0.969.
The sqrt weights fixed the NV recall collapse (0.748→0.823) and balanced the class predictions.

**What got worse:** test bAcc 0.784→0.757. Root cause: **overfitting.**

| Signal | Evidence |
|---|---|
| Train loss dropped 28× | ep1=0.977 → ep23=0.034 |
| Val loss rose after ep8 | ep8=0.590 → ep23=0.788 |
| Best val bAcc also at ep8 | peaked 0.811, then oscillated down to 0.750–0.784 |
| Val-test gap widened | v1 gap=2.5pp → v2 gap=5.5pp |

113M-param SENet-154 is massively overparameterised for 8,750 training images (99 DF, 143 VASC). Without strong regularisation, the backbone memorises the training distribution by epoch 10 and generalisation collapses. The class-weighted loss amplifies minority gradients, accelerating this memorisation for the rarest classes.

The same oscillation pattern appears in both runs (best checkpoint at epoch 7–8, then val bAcc falls despite val loss staying low), which confirms the issue is in the model, not the LR schedule.

---

## v2 epoch-by-epoch

| Ep | train loss | val loss | val bAcc | val top-1 | val F1 | LR |
|---|---|---|---|---|---|---|
| 1 | 0.9771 | 0.7213 | 0.6752 | 0.7778 | 0.5988 | 9.97e-05 |
| 2 | 0.6957 | 0.7465 | 0.6979 | 0.7682 | 0.5942 | 9.89e-05 |
| 3 | 0.5797 | 0.6444 | 0.7175 | 0.8141 | 0.6854 | 9.76e-05 |
| 4 | 0.5096 | 0.6002 | 0.7362 | 0.8258 | 0.7146 | 9.57e-05 |
| 5 | 0.4603 | 0.6330 | 0.7565 | 0.8285 | 0.7417 | 9.34e-05 |
| 6 | 0.4030 | 0.6880 | 0.7444 | 0.8155 | 0.7026 | 9.05e-05 |
| 7 | 0.3719 | 0.6020 | 0.7717 | 0.8292 | 0.7190 | 8.73e-05 |
| **8** | **0.3444** | **0.5903** | **0.8106** ← best | **0.8134** | **0.7518** | **8.36e-05** |
| 9 | 0.3047 | 0.6224 | 0.7568 | 0.8512 | 0.7551 | 7.96e-05 |
| 10 | 0.2628 | 0.5897 | 0.7710 | 0.8505 | 0.7592 | 7.52e-05 |
| 11 | 0.2250 | 0.6107 | 0.7825 | 0.8690 | 0.7740 | 7.06e-05 |
| 12 | 0.1816 | 0.6109 | 0.7840 | 0.8560 | 0.7691 | 6.58e-05 |
| 13 | 0.1923 | 0.7557 | 0.7636 | 0.8477 | 0.7442 | 6.08e-05 |
| 14 | 0.1620 | 0.7318 | 0.7654 | 0.8416 | 0.7516 | 5.57e-05 |
| 15 | 0.1295 | 0.7762 | 0.7545 | 0.8429 | 0.7352 | 5.05e-05 |
| 16 | 0.1136 | 0.8133 | 0.7739 | 0.8608 | 0.7806 | 4.53e-05 |
| 17 | 0.0907 | 0.8615 | 0.7480 | 0.8676 | 0.7643 | 4.02e-05 |
| 18 | 0.0854 | 0.7318 | 0.7774 | 0.8539 | 0.7633 | 3.52e-05 |
| 19 | 0.0673 | 0.8132 | 0.7535 | 0.8621 | 0.7546 | 3.04e-05 |
| 20 | 0.0646 | 0.7705 | 0.7497 | 0.8628 | 0.7643 | 2.58e-05 |
| 21 | 0.0485 | 0.8580 | 0.7626 | 0.8697 | 0.7727 | 2.14e-05 |
| 22 | 0.0434 | 0.8312 | 0.7497 | 0.8628 | 0.7572 | 1.74e-05 |
| 23 | 0.0343 | 0.7876 | 0.7696 | 0.8663 | 0.7698 | 1.37e-05 |

---

## v1 epoch-by-epoch

| Ep | train loss | val loss | val bAcc | val top-1 | val F1 | LR |
|---|---|---|---|---|---|---|
| 1 | 1.1801 | 0.9827 | 0.6948 | 0.6646 | 0.5264 | 9.99e-05 |
| 2 | 0.8475 | 0.7804 | 0.7218 | 0.7051 | 0.5458 | 9.97e-05 |
| 3 | 0.7129 | 0.6992 | 0.7330 | 0.7318 | 0.6285 | 9.94e-05 |
| 4 | 0.6275 | 0.8220 | 0.7323 | 0.7942 | 0.6656 | 9.89e-05 |
| 5 | 0.5640 | 0.7127 | 0.7392 | 0.7565 | 0.6585 | 9.83e-05 |
| 6 | 0.5315 | 0.6754 | 0.7550 | 0.8011 | 0.6856 | 9.76e-05 |
| **7** | **0.4774** | **0.6220** | **0.8089** ← best | **0.7668** | **0.6528** | **9.67e-05** |
| 8 | 0.4710 | 0.7234 | 0.7700 | 0.7394 | 0.6793 | 9.57e-05 |
| 9 | 0.4449 | 0.6680 | 0.7807 | 0.8381 | 0.7459 | 9.46e-05 |
| 10 | 0.3731 | 0.8022 | 0.7866 | 0.7497 | 0.6975 | 9.34e-05 |
| 11 | 0.3272 | 0.8469 | 0.7438 | 0.8532 | 0.7493 | 9.20e-05 |
| 12 | 0.3067 | 0.7790 | 0.7412 | 0.8018 | 0.6820 | 9.05e-05 |
| 13 | 0.3346 | 0.8449 | 0.7162 | 0.7956 | 0.6628 | 8.90e-05 |
| 14 | 0.3053 | 0.9755 | 0.7469 | 0.8066 | 0.7071 | 8.73e-05 |
| 15 | 0.2743 | 0.9667 | 0.7603 | 0.8086 | 0.7072 | 8.55e-05 |
| 16 | 0.2622 | 0.6905 | 0.8028 | 0.8333 | 0.7657 | 8.36e-05 |
| 17 | 0.2017 | 0.9837 | 0.7317 | 0.8416 | 0.7495 | 8.17e-05 |
