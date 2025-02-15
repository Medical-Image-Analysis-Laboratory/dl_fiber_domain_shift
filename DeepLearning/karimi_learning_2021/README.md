# Learning to estimate the fiber orientation distribution function from diffusion-weighted MRI

This folder contains our reimplementation of the MLP model for estimating the fiber orientation distribution function (fODF) from diffusion-weighted MRI using TensorFlow 2. The original code can be found at https://github.com/bchimagine/fODF_deep_learning

The model is described in the following paper:

- D. Karimi, L. Vasung, C. Jaimes, F. Machado-Rivas, S. K. Warfield, and A. Gholipour, ‘Learning to estimate the fiber orientation distribution function from diffusion-weighted MRI’, NeuroImage, vol. 239, p. 118316, Oct. 2021, doi: 10.1016/j.neuroimage.2021.118316.

## Requirements

- Python 3
- Dipy
- Nibabel
- NumPy
- TensorFlow

## Structure

```plaintext
.
├── data.py   # Data
├── mlp.py    # Model
├── train.py  # Training script
└── utils.py  # Utility functions
```