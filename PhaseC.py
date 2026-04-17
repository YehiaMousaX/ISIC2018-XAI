
# %% [markdown]
# ---
# # Phase C — XAI Method Implementation
# > **Thesis §5.3 — XAI Methods**
# 
# Implements all six explanation methods behind a single `generate_heatmap()` interface.
# Function definitions always load. Only the visual validation cell (C.4) is gated.
# 
# | Method | Type | Architectures |
# |--------|------|---------------|
# | Grad-CAM | Gradient × Activation | CNN only |
# | HiResCAM | Gradient × Activation | CNN only |
# | Attention Rollout | Attention propagation | ViT only |
# | Integrated Gradients | Path gradient | All |
# | LIME | Perturbation (superpixels) | All |
# | KernelSHAP | Perturbation (superpixels) | All |
# 
# ### Steps
# C.1 Grad-CAM helpers · C.2 Attention Rollout · C.3 Integrated Gradients · C.4 LIME & KernelSHAP · C.5 Unified interface · C.6 Visual validation

# %%
# Set to True once models are trained and you are ready to run Phase C.
RUN_PHASE_C = False

if not RUN_PHASE_C:
    print("⏸  Phase C is disabled. "
          "Set RUN_PHASE_C = True to run XAI generation.")


# %% [markdown]
# ## C.1 — Grad-CAM Helpers
# Returns the correct `nn.Module` target layer for `pytorch-grad-cam`.
# These run for all architectures; ViTs simply don’t use them.

# %%
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_gradcam_target_layer(model, arch_key):
    """Return the nn.Module list for the last conv/feature layer."""
    if arch_key == "efficientnet_b0":
        return [model.conv_head]
    elif arch_key == "densenet121":
        return [model.features.denseblock4.denselayer16.conv2]
    else:
        raise ValueError(f"No Grad-CAM target defined for {arch_key}")


print("C.1 ✔ Grad-CAM helpers defined.")


# %% [markdown]
# ## C.2 — Attention Rollout
# Rolls up attention matrices across all transformer layers (Abnar & Zuidema 2020).
# `vit_base_16`: hooks into each block’s attention module.
# `swin_tiny`: shifted-window attention is non-uniform — falls back to Layer GradCAM on the last norm layer as a pragmatic substitute.

# %%
def _attention_rollout(model, img_tensor, arch_key):
    """Attention Rollout for ViT-Base/16.
    Returns heatmap np.ndarray (num_patches, num_patches).
    """
    model.eval()
    attention_maps = []
    hooks = []

    def _hook_fn(module, inp, out):
        # Recompute softmax attention from qkv weights (timm ViT layout)
        if not hasattr(module, "qkv"):
            return
        B, N, C = inp[0].shape
        qkv = module.qkv(inp[0]).reshape(
            B, N, 3, module.num_heads, C // module.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        attention_maps.append(attn.detach().cpu().numpy())

    if arch_key == "vit_base_16":
        for block in model.blocks:
            hooks.append(block.attn.register_forward_hook(_hook_fn))
    elif arch_key == "swin_tiny":
        return _swin_gradcam_fallback(model, img_tensor)

    with torch.no_grad():
        _ = model(img_tensor)
    for h in hooks:
        h.remove()

    if not attention_maps:
        return np.zeros((IMG_SIZE, IMG_SIZE))

    # Rollout: multiply attention matrices layer by layer, add residual
    result = None
    for attn in attention_maps:          # attn: (B, heads, N, N)
        attn_avg = attn.mean(axis=1)[0]  # avg over heads: (N, N)
        attn_avg = attn_avg + np.eye(attn_avg.shape[0])  # residual
        attn_avg = attn_avg / attn_avg.sum(axis=-1, keepdims=True)
        result = attn_avg if result is None else result @ attn_avg

    # CLS row [0] attends to all patch tokens [1:]
    cls_attn    = result[0, 1:]
    num_patches = int(np.sqrt(cls_attn.shape[0]))
    return cls_attn.reshape(num_patches, num_patches)


def _swin_gradcam_fallback(model, img_tensor):
    """Layer GradCAM on Swin’s last norm layer (pragmatic substitute)."""
    model.eval()
    target_layer = [model.layers[-1].blocks[-1].norm1]
    with GradCAM(model=model, target_layers=target_layer) as cam:
        with torch.no_grad():
            tc = model(img_tensor).argmax(1).item()
        heatmap = cam(
            input_tensor=img_tensor,
            targets=[ClassifierOutputTarget(tc)]
        )[0]
    return heatmap


print("C.2 ✔ Attention Rollout defined.")


# %% [markdown]
# ## C.3 — Integrated Gradients
# Pure PyTorch — no Captum dependency (Captum pins numpy<2.0, breaking albumentations).
# Path integral from a zero baseline to the input; averages gradients over `steps` interpolations.
# Works for both CNN and ViT architectures.

# %%
def _integrated_gradients(model, img_tensor, target_class, steps=50):
    """Integrated Gradients (Sundararajan et al. 2017) — pure PyTorch.
    Baseline: zero tensor (black image in normalised space).
    Returns heatmap np.ndarray (H, W).
    """
    model.eval()
    baseline = torch.zeros_like(img_tensor).to(DEVICE)
    alphas   = torch.linspace(0, 1, steps, device=DEVICE)

    grads = []
    for alpha in alphas:
        interp = (baseline + alpha * (img_tensor - baseline)).requires_grad_(True)
        output = model(interp)
        score  = output[0, target_class]
        score.backward()
        grads.append(interp.grad.detach().clone())

    # Trapezoidal average of gradients
    avg_grads = torch.stack(grads).mean(dim=0)
    ig = (img_tensor - baseline) * avg_grads          # (1, 3, H, W)
    heatmap = ig.squeeze().abs().mean(dim=0).cpu().numpy()  # (H, W)
    return heatmap


print("C.3 ✔ Integrated Gradients defined.")


# %% [markdown]
# ## C.4 — LIME & KernelSHAP
# Both methods treat the model as a black box and perturb superpixels.
# LIME uses weighted linear regression on the neighbourhood; KernelSHAP uses Shapley-weighted least squares.
# Both use SLIC superpixels as the feature space.

# %%
from skimage.segmentation import slic


def _make_predict_fn(model):
    """Shared predict function for LIME and KernelSHAP."""
    _mean = np.array(stats["mean"], dtype=np.float32)
    _std  = np.array(stats["std"],  dtype=np.float32)

    def predict_fn(images):
        # images: (N, H, W, 3) float32 in [0, 1]
        batch = torch.stack([
            torch.from_numpy((img - _mean) / _std).permute(2, 0, 1).float()
            for img in images
        ]).to(DEVICE)
        with torch.no_grad():
            return torch.softmax(model(batch), dim=1).cpu().numpy()
    return predict_fn


def _lime_heatmap(model, img_tensor, target_class):
    """LIME attribution — returns (H, W) weight map over superpixels."""
    from lime import lime_image
    _mean  = np.array(stats["mean"], dtype=np.float32)
    _std   = np.array(stats["std"],  dtype=np.float32)
    img_np = np.clip(
        img_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * _std + _mean,
        0, 1
    ).astype(np.float32)

    predict_fn  = _make_predict_fn(model)
    explainer   = lime_image.LimeImageExplainer(verbose=False)
    explanation = explainer.explain_instance(
        img_np, predict_fn,
        top_labels=NUM_CLASSES,
        hide_color=0,
        num_samples=LIME_SAMPLES,
    )

    # Build pixel-level weight map from segment weights
    heatmap = np.zeros(img_np.shape[:2], dtype=np.float32)
    for seg_id, weight in explanation.local_exp[target_class]:
        heatmap[explanation.segments == seg_id] = weight
    return heatmap


def _kernelshap_heatmap(model, img_tensor, target_class):
    """KernelSHAP attribution — returns (H, W) Shapley value map."""
    import shap
    _mean  = np.array(stats["mean"], dtype=np.float32)
    _std   = np.array(stats["std"],  dtype=np.float32)
    img_np = np.clip(
        img_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * _std + _mean,
        0, 1
    ).astype(np.float32)

    segments   = slic(img_np, n_segments=50, compactness=10, start_label=0)
    n_segments = int(segments.max()) + 1

    def seg_predict_fn(z):
        # z: (N, n_segments) binary — 1 = keep, 0 = black out
        images = []
        for mask_vec in z:
            img = img_np.copy()
            for seg_id in range(n_segments):
                if mask_vec[seg_id] == 0:
                    img[segments == seg_id] = 0.0
            images.append(img)
        batch = torch.stack([
            torch.from_numpy((img - _mean) / _std).permute(2, 0, 1).float()
            for img in images
        ]).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        return probs[:, target_class]

    background = np.ones((1, n_segments))  # all segments present
    explainer  = shap.KernelExplainer(seg_predict_fn, background)
    shap_vals  = explainer.shap_values(
        np.ones((1, n_segments)), nsamples=SHAP_SAMPLES, silent=True
    )

    # Map segment Shapley values back to pixels
    heatmap = np.zeros(img_np.shape[:2], dtype=np.float32)
    for seg_id in range(n_segments):
        heatmap[segments == seg_id] = shap_vals[0, seg_id]
    return heatmap


print("C.4 ✔ LIME & KernelSHAP defined.")


# %% [markdown]
# ## C.5 — Unified `generate_heatmap()` Interface
# Single entry point for all downstream phases. Returns a normalised `[0, 1]` attribution map of shape `(H, W)` regardless of method.
# 
# | Arg | Type | Notes |
# |-----|------|-------|
# | `model` | `nn.Module` | Must be in eval mode on DEVICE |
# | `img_tensor` | `(1,3,H,W)` tensor | Normalised, on DEVICE |
# | `method` | str | See table above |
# | `arch_key` | str | Key from `ARCHITECTURES` |
# | `target_class` | int \| None | None → uses predicted class |

# %%
from skimage.transform import resize as sk_resize


def get_applicable_methods(arch_key):
    """Return the XAI methods valid for this architecture."""
    family = ARCHITECTURES[arch_key]["family"]
    if family == "cnn":
        return ["gradcam", "hirescam",
                "integrated_gradients", "lime", "kernelshap"]
    else:
        return ["attention_rollout",
                "integrated_gradients", "lime", "kernelshap"]


def generate_heatmap(model, img_tensor, method, arch_key, target_class=None):
    """
    Generate a normalised [0, 1] attribution heatmap.

    Args:
        model        : trained nn.Module (eval mode, on DEVICE)
        img_tensor   : (1, 3, H, W) tensor on DEVICE
        method       : "gradcam" | "hirescam" | "attention_rollout"
                       "integrated_gradients" | "lime" | "kernelshap"
        arch_key     : key from ARCHITECTURES dict
        target_class : int or None (None → argmax of model output)

    Returns:
        heatmap : np.ndarray shape (H, W), values in [0, 1]
    """
    model.eval()
    family = ARCHITECTURES[arch_key]["family"]
    H, W   = img_tensor.shape[2], img_tensor.shape[3]

    if target_class is None:
        with torch.no_grad():
            target_class = model(img_tensor).argmax(1).item()

    # ── Dispatch ─────────────────────────────────────────────────────────────────
    if method == "gradcam":
        assert family == "cnn", "Grad-CAM requires a CNN architecture."
        with GradCAM(model=model,
                     target_layers=get_gradcam_target_layer(model, arch_key)) as cam:
            heatmap = cam(
                input_tensor=img_tensor,
                targets=[ClassifierOutputTarget(target_class)]
            )[0]

    elif method == "hirescam":
        assert family == "cnn", "HiResCAM requires a CNN architecture."
        with HiResCAM(model=model,
                      target_layers=get_gradcam_target_layer(model, arch_key)) as cam:
            heatmap = cam(
                input_tensor=img_tensor,
                targets=[ClassifierOutputTarget(target_class)]
            )[0]

    elif method == "attention_rollout":
        assert family == "vit", "Attention Rollout requires a ViT architecture."
        heatmap = _attention_rollout(model, img_tensor, arch_key)

    elif method == "integrated_gradients":
        heatmap = _integrated_gradients(model, img_tensor, target_class)

    elif method == "lime":
        heatmap = _lime_heatmap(model, img_tensor, target_class)

    elif method == "kernelshap":
        heatmap = _kernelshap_heatmap(model, img_tensor, target_class)

    else:
        raise ValueError(f"Unknown method: {method!r}. "
                         f"Choose from: gradcam, hirescam, attention_rollout, "
                         f"integrated_gradients, lime, kernelshap")

    # ── Normalise to [0, 1] ─────────────────────────────────────────────────────────────
    heatmap = heatmap.astype(np.float32)
    mn, mx  = heatmap.min(), heatmap.max()
    heatmap = (heatmap - mn) / (mx - mn) if mx > mn else np.zeros_like(heatmap)

    # Resize to input spatial dims if method returned a different size
    if heatmap.shape != (H, W):
        heatmap = sk_resize(heatmap, (H, W), order=1, anti_aliasing=True)

    return heatmap.astype(np.float32)


print("C.5 ✔ generate_heatmap() unified interface defined.")
print("Applicable methods per family:")
for ak in ARCHITECTURES:
    print(f"  {ak:20s}: {get_applicable_methods(ak)}")


# %% [markdown]
# ## C.6 — Visual Validation
# > **Gated by `RUN_PHASE_C`** — set it to `True` in the config cell above.
# 
# Picks one correctly-classified test image per architecture, generates all applicable
# heatmaps, and renders them as a grid. Must pass before proceeding to Phase D.
# 
# **Pass criteria:**
# - No cell errors
# - Heatmaps are not all-zero or all-one
# - Grad-CAM / HiResCAM highlight lesion area, not background
# - Attention Rollout shows coherent spatial structure
# - LIME produces superpixel-granularity highlights

# %%
if not RUN_PHASE_C:
    print("⏸  Skipped. Set RUN_PHASE_C = True to run visual validation.")
else:
    import matplotlib.pyplot as plt
    _mn  = np.array(stats["mean"])
    _std = np.array(stats["std"])

    def denorm(t):
        return np.clip(t.permute(1, 2, 0).cpu().numpy() * _std + _mn, 0, 1)

    # One correctly classified image per architecture
    n_methods_max = 5   # gradcam/hirescam + 3 shared = 5 cols for CNNs
    fig, axes = plt.subplots(
        len(ARCHITECTURES), n_methods_max + 1,
        figsize=(4 * (n_methods_max + 1), 4 * len(ARCHITECTURES))
    )

    for row, arch_key in enumerate(ARCHITECTURES):
        model   = trained_models[arch_key].to(DEVICE).eval()
        methods = get_applicable_methods(arch_key)

        # Pick first correctly-classified image for this arch
        subset    = eval_subsets[arch_key]
        sample_id = subset.iloc[0]["image_id"]
        split_dir = TEST_IMG
        img_raw   = np.array(
            Image.open(f"{split_dir}/{sample_id}.jpg").convert("RGB")
                 .resize((IMG_SIZE, IMG_SIZE))
        ) / 255.0

        img_tensor = eval_transform(
            image=(img_raw * 255).astype(np.uint8)
        )["image"].unsqueeze(0).to(DEVICE)
        true_cls = subset.iloc[0]["label_name"]
        pred_cls = CLASS_NAMES[subset.iloc[0]["pred"]]

        # Original image
        axes[row, 0].imshow(img_raw)
        axes[row, 0].set_title(
            f"{arch_key}\ntrue={true_cls} pred={pred_cls}", fontsize=8
        )
        axes[row, 0].axis("off")

        for col, method in enumerate(methods):
            heatmap = generate_heatmap(
                model, img_tensor, method, arch_key,
                target_class=subset.iloc[0]["pred"]
            )
            ax = axes[row, col + 1]
            ax.imshow(img_raw)
            ax.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax.set_title(method, fontsize=9)
            ax.axis("off")

        # Hide unused columns
        for col in range(len(methods) + 1, n_methods_max + 1):
            axes[row, col].axis("off")

        model = model.cpu()
        torch.cuda.empty_cache()

    fig.suptitle("XAI Method Visual Validation — One Image per Architecture",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_ROOT, "xai_visual_validation.png"),
        dpi=120, bbox_inches="tight"
    )
    plt.show()
    print("✔ Visual validation complete.")



