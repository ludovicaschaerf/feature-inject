# FeatureInject: Cross-Architectural Layer-Wise Feature Injection for Diffusion Editing

**Training-free, prompt-guided editing and style transfer that adapt to any diffusion architecture.**  
*(Companion code for “One Size Does Not Fit All: Cross-Architectural Layer-Wise Representations in Diffusion Models,”)*

---

## 🧠 Overview

FeatureInject identifies where **semantic and stylistic information** live in diffusion backbones (U-Net, SDXL, DiT) and injects only the **right layers** to perform clean, structure-preserving edits and style transfers—without retraining.

Unlike methods assuming uniform feature distributions, FeatureInject shows that **representations vary strongly by architecture**, and that optimal injection layers differ between SD1.4/2/XL, Kandinsky, SD3.5-Turbo, and Flux.

---

## ⚙️ Installation

```bash
conda create -n finject python=3.10 -y
conda activate finject
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors pillow opencv-python matplotlib tqdm einops scikit-image
```

---

## 🚀 Quickstart

Run the notebook:

```bash
jupyter notebook test.ipynb
```

Or use the CLI:

```bash
python main.py   --model "stabilityai/stable-diffusion-xl-base-1.0"   --mode edit   --source "a zebra in a city, realistic"   --edit "a parrot in a forest, neon colors"   --seed 123   --steps 50   --out outputs/sdxl_edit.png
```
(TODO insert mode into main.py)

**Modes:** `edit` (semantic) or `style` (appearance).  
FeatureInject automatically selects layer sets per model but you can override them.

---

## 🧩 Supported Models

- Stable Diffusion 1.4, 2, XL (+Turbo)  
- Kandinsky 2.x  
- SD3.5-Turbo, Flux (DiT backbones)

---

## 📊 Reproducing Paper Results

1. Run the probing script to map “mixing” layers (structure vs. style).  
2. Use identified layers for editing or style transfer.  
3. Metrics: SSIM, keypoints, HSV histogram, LBP texture, CLIP sim.

---

## 📚 Citation

```bibtex
@inproceedings{featureinject,
  title     = {One Size Does Not Fit All: Cross-Architectural Layer-Wise Representations in Diffusion Models},
  booktitle = {},
  year      = {},
  note      = {}
}
```

---

**License:** Research only. Check model-specific terms before redistribution.
