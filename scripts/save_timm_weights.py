"""
save_timm_weights.py — One-time script to cache timm pretrained weights locally.

Usage:
    python scripts/save_timm_weights.py efficientnet_b0
    python scripts/save_timm_weights.py resnet50
    python scripts/save_timm_weights.py vit_base_patch16_224

Saves weights to:  scripts/timm-weights/<arch>.pth

Then upload the whole scripts/timm-weights/ folder to Kaggle as a private
dataset named "timm-pretrained-weights" (slug: timm-pretrained-weights):
    kaggle datasets create -p scripts/timm-weights --dir-mode zip
Or update if it already exists:
    kaggle datasets version -p scripts/timm-weights -m "<arch> added"

The notebook loads weights from /kaggle/input/timm-pretrained-weights/<arch>.pth
and never downloads from HuggingFace at runtime.
"""
import sys
import timm
import torch
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / 'timm-weights'
WEIGHTS_DIR.mkdir(exist_ok=True)

archs = sys.argv[1:] if len(sys.argv) > 1 else ['efficientnet_b0']

for arch in archs:
    out = WEIGHTS_DIR / f'{arch}.pth'
    if out.exists():
        print(f'[skip] {arch}.pth already exists')
        continue
    print(f'Downloading {arch} from HuggingFace...')
    model = timm.create_model(arch, pretrained=True, num_classes=0)  # num_classes=0 → feature extractor, saves the backbone weights
    torch.save(model.state_dict(), out)
    size_mb = out.stat().st_size / 1e6
    print(f'Saved {arch}.pth  ({size_mb:.1f} MB)')

print(f'\nAll weights in: {WEIGHTS_DIR}')
print('Next: upload to Kaggle —')
print(f'  kaggle datasets create -p {WEIGHTS_DIR} --dir-mode zip')
print(f'  or: kaggle datasets version -p {WEIGHTS_DIR} -m "added {archs}"')
