# Setup Guide

---

## 1. Upload your data to Kaggle (one-time)

Go to **kaggle.com/datasets** → **New Dataset** and upload your `Data.zip`.

- Give it the title **`isic2018-dataset`** — this sets the slug to `yehiasamir/isic2018-dataset`.
- If you use a different title, open `XAI_Evaluation_Pipeline_Kaggle.ipynb` → cell **A.1** and update `KAGGLE_DATASET_SLUG` to match.

The notebook auto-detects whether your zip extracted with or without a `Data/` wrapper,
so either zip structure works.

---

## 2. Install the Kaggle CLI

```bash
pip install kaggle
```

Place your API token at `~/.kaggle/kaggle.json` (download from kaggle.com → Settings → API):

```bash
chmod 600 ~/.kaggle/kaggle.json
kaggle config view   # should print your username
```

---

## 3. Create your local config

```bash
cp scripts/pipeline.env.example scripts/pipeline.env
```

The defaults already match your setup (`yehiasamir`, `isic2018-xai-evaluation`).
No edits needed unless you renamed something.

---

## 4. Push the notebook to Kaggle (first time)

```bash
chmod +x scripts/*.sh
bash scripts/push_to_kaggle.sh
```

This creates the kernel on Kaggle and attaches `yehiasamir/isic2018-dataset` as the data source.

---

## 5. Run

```bash
bash scripts/run.sh "initial run"
# or on Windows PowerShell:
.\run.ps1 "initial run"
```

This commits any local changes, pushes to GitHub, pushes the notebook to Kaggle, and starts the kernel run.

---

## Daily workflow

| Action | Command |
|--------|---------|
| Full cycle (commit → push → Kaggle run) | `bash scripts/run.sh "message"` or `.\run.ps1 "message"` |
| Push notebook only | `make push` |
| Check Kaggle run status | `make status` |
| Wait for run to finish | `make wait` |
| Pull output files | `make pull` |
| Capture error log | `make log-error` |

---

## Editing the pipeline

**Always edit `XAI_Evaluation_Pipeline_Kaggle.py`, never the `.ipynb` directly.**

The `.py` file is the source of truth — it is easier to read, diff, and edit. After any change, sync it to the notebook with:

```bash
jupytext --to notebook --set-kernel python3 --output XAI_Evaluation_Pipeline_Kaggle.ipynb XAI_Evaluation_Pipeline_Kaggle.py
```

> `--set-kernel python3` embeds the kernel spec — **required** or Kaggle's papermill will error with `No kernel name found`.
> `--output` overwrites the existing notebook. Cell outputs from previous Kaggle runs are discarded (use `--update` instead if you want to preserve them).
>
> **You don't need to run this manually** — `run.sh` does it automatically at step 0 before every push.

Then run the normal cycle:

```bash
bash scripts/run.sh "describe your change"
```

---

## Handing an error to Claude

1. Run fails → `errors/latest_error.txt` is written automatically.
2. Tell Claude:

   > "New error in `errors/latest_error.txt`. Please read it and fix the notebook."

3. Claude reads the error, patches **`XAI_Evaluation_Pipeline_Kaggle.py`**, then converts to `.ipynb` and commits.
