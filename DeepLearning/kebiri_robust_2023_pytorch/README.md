# Robust Estimation of the Microstructure of the Early Developing Brain Using Deep Learning

This folder contains our reimplementation of the deep learning model for white matter fiber estimation in newborn and baby brains using PyTorch.

The model is described in the following papers:

- H. Kebiri, A. Gholipour, R. Lin, L. Vasung, D. Karimi, and M. Bach Cuadra, "Robust Estimation of the Microstructure of the Early Developing Brain Using Deep Learning," in _26th International Conference on Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023_, Oct. 2023, pp. 293–303, doi: 10.1007/978-3-031-43990-2_28.
- H. Kebiri, A. Gholipour, R. Lin, L. Vasung, C. Calixto, Ž. Krsnik, D. Karimi, and M. Bach Cuadra, "Deep learning microstructure estimation of developing brains from diffusion MRI: A newborn and fetal study," _Medical Image Analysis_, vol. 95, p. 103186, Jul. 2024, doi: 10.1016/j.media.2024.103186.

## Requirements

Dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
```

- Python 3
- Dipy
- Nibabel
- NumPy
- PyTorch
- PyTorch Lightning
- MONAI

## Structure

```plaintext
├── data.py                   # Data module
├── model.py                  # Model
├── nets.py                   # Neural networks
├── patched_data_module.py    # Patched data module
├── patched_model.py          # Patched model
├── patched_train.py          # Patched training script
├── patcher.py                # Patching script
├── requirements.txt          # Dependencies
└── train.py                  # Training script
```
