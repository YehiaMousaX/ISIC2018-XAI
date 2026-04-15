# One-Time Setup

Run these steps once after cloning. Then use `bash scripts/run.sh` every day.

---

## 1. Kaggle CLI

```bash
pip install kaggle
# Place your kaggle.json at ~/.kaggle/kaggle.json  (never commit this)
chmod 600 ~/.kaggle/kaggle.json
kaggle config view
```

## 2. Configure the ISIC pipeline

Create your local config file and edit it:

```bash
cp scripts/pipeline.env.example scripts/pipeline.env
```

Update at least these fields in `scripts/pipeline.env`:

- `KAGGLE_USER`
- `KERNEL_SLUG`
- `KERNEL_ID`
- `NOTEBOOK`
- `DATASET_SOURCES` (must include your ISIC images dataset slug)

Example:

```bash
KAGGLE_USER=yehiasamir
KERNEL_SLUG=isic2018-xai-evaluation
KERNEL_ID=yehiasamir/isic2018-xai-evaluation
NOTEBOOK=XAI_Evaluation_Pipeline_Kaggle.ipynb
DATASET_SOURCES=sani84/isic-2018-classification,yehiasamir/isic2018-prepared,yehiasamir/timm-pretrained-weights
```

## 3. Create the Kaggle kernel (first time only)

Push the notebook to create the kernel entry on Kaggle:

```bash
bash scripts/push_to_kaggle.sh
```

If the kernel doesn't exist, Kaggle creates it. If it exists, it's updated.

## 4. Make scripts executable (Git Bash)

```bash
chmod +x scripts/*.sh
```

## 5. Run local notebooks first

Before doing any Kaggle run, complete the local workflow:
1. Open and run `EDA.ipynb` to explore the dataset
2. Run your preprocessing/split notebook to generate `prepared/*.csv`
3. Verify `prepared/` has split files and metadata required by the training notebook

## 6. Add prepared/ as a Kaggle dataset (if not already done)

The `prepared/` CSVs must be available to the Kaggle kernel as a dataset source.
Upload them once:
```bash
# Create dataset metadata
kaggle datasets init -p prepared/
# Edit prepared/dataset-metadata.json with your username and slug
kaggle datasets create -p prepared/
```
Then add the slug to `DATASET_SOURCES` in `scripts/pipeline.env`.

---

## Daily Loop

| Step | Command |
|---|---|
| Edit notebook | — |
| Full cycle | `bash scripts/run.sh "note"` or `.\run.ps1 "note"` |
| Push only | `make push` |
| Check status only | `make status` |
| Wait for run | `make wait` |
| Pull results | `make pull` |
| Capture error | `make log-error` |
| Compare all runs | `bash scripts/compare.sh` |

## Handing an Error to Claude

1. Run fails → `errors/latest_error.txt` is auto-written.
2. Tell Claude:

   > "New error in `errors/latest_error.txt`. Please read it and fix the notebook."

3. Claude reads both files, patches the notebook, commits.
