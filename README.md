This repository contains the code of the paper  
**“MedOpenSeg: Open-World Medical Segmentation with Memory-Augmented Transformers.”** _Accepted at **BMVC 2025**_ 🎉

*Luisa Vargas<sup>\*</sup><sup>1</sup>, Eleonora Poeta<sup>\*</sup><sup>2</sup>, Tania Cerquitelli<sup>2</sup>, Elena Baralis<sup>2</sup>, Maria A. Zuluaga<sup>1</sup>*

<sub><sub>\* Denotes equal contribution</sub></sub>
---

## 📦 Quick Installation
```bash
# Install PyTorch (recommended version: 2.1.0, adjust CUDA as needed)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Minimal extra packages
pip install numpy scipy nibabel SimpleITK tqdm monai
```

---

## 📂 Datasets
Experiments are conducted on:
- **AMOS 2022** (multi-organ CT, 15 abdominal organs)  
- **BTCV (Synapse)** (multi-organ abdominal CT)  
- **MSD – Pancreas** (pancreas + tumors)

---

## 🚀 Usage

**Training**
```bash
python trainV.py --config configs/amos.yaml --data_root DATA/AMOS --exp_name amos_os
```

**Inference (masks + novelty maps)**
```bash
python inference_V.py   --config configs/amos.yaml   --checkpoint runs/amos_os/best.ckpt   --data_root DATA/AMOS   --save_dir outputs/amos_os
```

---

## 📖 How to cite
If this work helps your research, please cite:

---

🙏 Thank you for your interest in **MedOpenSeg**!
