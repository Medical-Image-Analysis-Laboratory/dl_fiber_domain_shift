# Cross-Age and Cross-Site Domain Shift Impacts on Deep Learning-Based White Matter Fiber Estimation in Newborn and Baby Brains

This repository contains the code for the paper **"[Cross-Age and Cross-Site Domain Shift Impacts on Deep Learning-Based White Matter Fiber Estimation in Newborn and Baby Brains](https://arxiv.org/abs/2312.14773)"** by Rizhong Lin, Ali Gholipour, Jean-Philippe Thiran, Davood Karimi, Hamza Kebiri, and Meritxell Bach Cuadra, which is accepted by the [_21st IEEE International Symposium on Biomedical Imaging (ISBI 2024)_](https://biomedicalimaging.org/2024/).
The code will be available soon.

## Abstract

Deep learning models have shown great promise in estimating tissue microstructure from limited diffusion magnetic resonance imaging data. However, these models face domain shift challenges when test and train data are from different scanners and protocols, or when the models are applied to data with inherent variations such as the developing brains of infants and children scanned at various ages. Several techniques have been proposed to address some of these challenges, such as data harmonization or domain adaptation in the adult brain. However, those techniques remain unexplored for the estimation of fiber orientation distribution functions in the rapidly developing brains of infants. In this work, we extensively investigate the age effect and domain shift within and across two different cohorts of 201 newborns and 165 babies using the Method of Moments and fine-tuning strategies. Our results show that reduced variations in the microstructural development of babies in comparison to newborns directly impact the deep learning models' cross-age performance. We also demonstrate that a small number of target domain samples can significantly mitigate domain shift problems.

## Code Availability

Detailed instructions for using the code, along with requirements for the computational environment, will be provided upon the code's release.

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

<!-- ## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->

## Acknowledgments

We acknowledge access to the facilities and expertise of the CIBM Center for Biomedical Imaging, a Swiss research center of excellence founded and supported by Lausanne University Hospital (CHUV), University of Lausanne (UNIL), Ecole polytechnique fédérale de Lausanne (EPFL), University of Geneva (UNIGE), Geneva University Hospitals (HUG) and the Leenaards and Jeantet Foundations.

This research was partly funded by the Swiss National Science Foundation (grants 182602 and 215641); also by the National Institute of Neurological Disorders and Stroke, and the Eunice Kennedy Shriver National Institute of Child Health and Human Development of the National Institutes of Health (NIH) of the United States (award numbers R01NS106030, R01NS128281 and R01HD110772).

<!-- ## Contact

If you have any questions, please feel free to contact Rizhong Lin at

```python
'rizhong.lin@$.#'.replace('$', 'epfl').replace('#', 'ch')
``` -->
