# Robust Estimation of the Microstructure of the Early Developing Brain Using Deep Learning

This folder contains our implementation of the deep learning model for white matter fiber estimation in newborn and baby brains. The model is described in the following papers:

- H. Kebiri, A. Gholipour, R. Lin, L. Vasung, D. Karimi, and M. Bach Cuadra, "Robust Estimation of the Microstructure of the Early Developing Brain Using Deep Learning," in _26th International Conference on Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023_, Oct. 2023, pp. 293–303, doi: 10.1007/978-3-031-43990-2_28.
- H. Kebiri, A. Gholipour, R. Lin, L. Vasung, C. Calixto, Ž. Krsnik, D. Karimi, and M. Bach Cuadra, "Deep learning microstructure estimation of developing brains from diffusion MRI: A newborn and fetal study," _Medical Image Analysis_, vol. 95, p. 103186, Jul. 2024, doi: 10.1016/j.media.2024.103186.

## Requirements

- Python 3
- Dipy
- Nibabel
- NumPy
- TensorFlow

## Structure

- `data.py`: The script for loading and preprocessing the data.
- `model.py`: The script for defining the deep learning model.
- `train.py`: The script for training the deep learning model.
- `utils.py`: The script for utility functions.
- `requirements.txt`: The list of required Python packages.
