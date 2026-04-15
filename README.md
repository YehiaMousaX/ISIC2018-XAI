# Multi-Dimensional XAI Evaluation Pipeline — ISIC 2018

> **Thesis:** *A Multi-Dimensional Evaluation of Explainability Methods Across CNN and Vision Transformer Architectures for Skin Lesion Classification*

Evaluate six XAI methods across four architectures on the ISIC 2018 skin lesion dataset.
Developed locally, trained on Kaggle (T4 GPU).

---

## Architectures

| Model | Family |
|-------|--------|
| EfficientNet-B0 | CNN |
| DenseNet-121 | CNN |
| ViT-Base/16 | Vision Transformer |
| Swin-Tiny | Vision Transformer |

## XAI Methods

Grad-CAM · HiResCAM · Attention Rollout · LIME · KernelSHAP · Integrated Gradients

## Evaluation Dimensions

| Dimension | Metrics |
|-----------|---------|
| Faithfulness | AOPC, Insertion/Deletion AUC |
| Plausibility | IoU, Dice, Pointing Game |
| Robustness | Max-Sensitivity |
| Complexity | Saliency Entropy |

---

## Repository Structure

```
├── XAI_Evaluation_Pipeline_Kaggle.ipynb   # Main pipeline notebook
├── EDA_ISIC2018.ipynb                     # Exploratory data analysis
├── IMPLEMENTATION_PLAN.md                 # Phase-by-phase implementation plan
├── EDA_summary_insights.md                # EDA findings summary
├── DATASET.md                             # Full dataset reference
├── SETUP.md                               # Setup & Kaggle deployment guide
├── Makefile                               # Shortcuts (make push, make status, ...)
├── run.ps1                                # Windows PowerShell run script
└── scripts/
    ├── run.sh                             # Full cycle: commit → GitHub → Kaggle
    ├── push_to_kaggle.sh                  # Push notebook to Kaggle kernel
    ├── wait_for_kernel.sh                 # Poll Kaggle until run completes
    ├── pipeline.env.example               # Config template (copy → pipeline.env)
    └── kernel-meta/
        └── kernel-metadata.json           # Kaggle kernel config
```

> **Data is not in this repo.** It is uploaded separately as a Kaggle dataset (`yehiasamir/isic2018-dataset`) if run on kaggle or in the root as "data" if run local.

---

## Quick Start

### Local development

```bash
git clone https://github.com/YehiaMousaX/ISIC2018-XAI.git
cd ISIC2018-XAI
# Place your Data/ folder here, then open the notebook
jupyter notebook XAI_Evaluation_Pipeline_Kaggle.ipynb
```

### Kaggle training

See [SETUP.md](SETUP.md) for the full guide. Short version:

1. Upload `Data.zip` to Kaggle as dataset `isic2018-dataset`
2. `cp scripts/pipeline.env.example scripts/pipeline.env`
3. `bash scripts/push_to_kaggle.sh`
4. `bash scripts/run.sh "first run"`

---

## Pipeline Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **A** | Data loading, EDA, preprocessing | ✅ Done |
| **B** | Model training (4 architectures, Focal Loss) | ✅ Done |
| **C** | XAI method implementation | 🔲 Next |
| **D** | Faithfulness evaluation (AOPC, Ins/Del AUC) | 🔲 |
| **E** | Plausibility evaluation (IoU, Dice, Pointing Game) | 🔲 |
| **F** | Robustness & Complexity | 🔲 |
| **G** | Cross-method analysis & visualisation | 🔲 |
| **H** | Ablation studies | 🔲 |
| **I** | Final outputs & export | 🔲 |
