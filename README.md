# Cross-Age and Cross-Site Domain Shift Impacts on Deep Learning-Based White Matter Fiber Estimation in Newborn and Baby Brains

This repository contains the code for the paper **"[Cross-Age and Cross-Site Domain Shift Impacts on Deep Learning-Based White Matter Fiber Estimation in Newborn and Baby Brains](https://arxiv.org/abs/2312.14773)"** by Rizhong Lin, Ali Gholipour, Jean-Philippe Thiran, Davood Karimi, Hamza Kebiri, and Meritxell Bach Cuadra, which is accepted by the [_21st IEEE International Symposium on Biomedical Imaging (ISBI 2024)_](https://biomedicalimaging.org/2024/).

## Abstract

Deep learning models have shown great promise in estimating tissue microstructure from limited diffusion magnetic resonance imaging data. However, these models face domain shift challenges when test and train data are from different scanners and protocols, or when the models are applied to data with inherent variations such as the developing brains of infants and children scanned at various ages. Several techniques have been proposed to address some of these challenges, such as data harmonization or domain adaptation in the adult brain. However, those techniques remain unexplored for the estimation of fiber orientation distribution functions in the rapidly developing brains of infants. In this work, we extensively investigate the age effect and domain shift within and across two different cohorts of 201 newborns and 165 babies using the Method of Moments and fine-tuning strategies. Our results show that reduced variations in the microstructural development of babies in comparison to newborns directly impact the deep learning models' cross-age performance. We also demonstrate that a small number of target domain samples can significantly mitigate domain shift problems.

## Structure

- [`MethodOfMoments`](./MethodOfMoments): The implementation of the Method of Moments (MoM) for harmonizing diffusion MRI data across different sites. The method is described in the following paper:
  - K. M. Huynh, G. Chen, Y. Wu, D. Shen, and P.-T. Yap, "Multi-Site Harmonization of Diffusion MRI Data via Method of Moments," _IEEE Transactions on Medical Imaging_, vol. 38, no. 7, pp. 1599–1609, Jul. 2019, doi: 10.1109/TMI.2019.2895020.
- [`DeepLearning`](./DeepLearning): The implementation of the deep learning model for white matter fiber estimation in newborn and baby brains. The model is described in the following papers:
  - H. Kebiri, A. Gholipour, R. Lin, L. Vasung, D. Karimi, and M. Bach Cuadra, "Robust Estimation of the Microstructure of the Early Developing Brain Using Deep Learning," in _26th International Conference on Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023_, Oct. 2023, pp. 293–303, doi: 10.1007/978-3-031-43990-2_28.
  - H. Kebiri, A. Gholipour, R. Lin, L. Vasung, C. Calixto, Ž. Krsnik, D. Karimi, and M. Bach Cuadra, "Deep learning microstructure estimation of developing brains from diffusion MRI: A newborn and fetal study," _Medical Image Analysis_, vol. 95, p. 103186, Jul. 2024, doi: 10.1016/j.media.2024.103186.

## Data

The data used in this study are from the Developing Human Connectome Project (dHCP) and the Baby Connectome Project (BCP). The dHCP data are available at https://www.humanconnectome.org/study/lifespan-developing-human-connectome-project, and the BCP data are available at https://www.humanconnectome.org/study/lifespan-baby-connectome-project.

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{lin_cross-age_2024,
  title     = {Cross-{Age} and {Cross}-{Site} {Domain} {Shift} {Impacts} on {Deep} {Learning}-{Based} {White} {Matter} {Fiber} {Estimation} in {Newborn} and {Baby} {Brains}},
  doi       = {10.48550/arXiv.2312.14773},
  author    = {Lin, Rizhong and Gholipour, Ali and Thiran, Jean-Philippe and Karimi, Davood and Kebiri, Hamza and Bach Cuadra, Meritxell},
  year      = 2024,
  month     = may,
  booktitle = {21st {IEEE} {International} {Symposium} on {Biomedical} {Imaging} ({ISBI})}
}

@article{kebiri_deep_2024,
  title   = {Deep learning microstructure estimation of developing brains from diffusion {MRI}: A newborn and fetal study},
  url     = {https://www.sciencedirect.com/science/article/abs/pii/S1361841524001117},
  doi     = {10.1016/j.media.2024.103186},
  author  = {Kebiri, Hamza and Gholipour, Ali and Lin, Rizhong and Vasung, Lana and Calixto, Camilo and Krsnik, Željka and Karimi, Davood and Bach Cuadra, Meritxell},
  year    = {2024},
  month   = jul,
  journal = {Medical Image Analysis},
  volume  = {95},
  pages   = {103186},
  issn    = {1361-8415},
}

@inproceedings{kebiri_robust_2023,
  title     = {Robust {Estimation} of the {Microstructure} of the {Early} {Developing} {Brain} {Using} {Deep} {Learning}},
  url       = {http://link.springer.com/chapter/10.1007/978-3-031-43990-2_28},
  doi       = {10.1007/978-3-031-43990-2_28},
  author    = {Kebiri, Hamza and Gholipour, Ali and Lin, Rizhong and Vasung, Lana and Karimi, Davood and Bach Cuadra, Meritxell},
  year      = 2023,
  month     = oct,
  booktitle = {26th {International} {Conference} on {Medical} {Image} {Computing} and {Computer} {Assisted} {Intervention} -- {MICCAI} 2023},
  pages     = {293--303}
}
```

## Acknowledgments

We acknowledge access to the facilities and expertise of the CIBM Center for Biomedical Imaging, a Swiss research center of excellence founded and supported by Lausanne University Hospital (CHUV), University of Lausanne (UNIL), Ecole polytechnique fédérale de Lausanne (EPFL), University of Geneva (UNIGE), Geneva University Hospitals (HUG) and the Leenaards and Jeantet Foundations.

This research was partly funded by the Swiss National Science Foundation (grants 182602 and 215641); also by the National Institute of Neurological Disorders and Stroke, and the Eunice Kennedy Shriver National Institute of Child Health and Human Development of the National Institutes of Health (NIH) of the United States (award numbers R01NS106030, R01NS128281 and R01HD110772).

## Contact

Please feel free to contact Rizhong Lin at

```python
# Python
'rizhong.lin@$.#'.replace('$', 'epfl').replace('#', 'ch')
```
