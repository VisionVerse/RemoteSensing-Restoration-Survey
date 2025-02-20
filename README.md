# Remote Sensing Image-Restoration-Survey

:fire::fire: In this review, we have systematically examined **over 150 papers** :page_with_curl::page_with_curl::page_with_curl:, summarizing and analyzing :star2:**more than 30** blind motion deblurring methods. 

:fire::fire::fire: Extensive qualitative and quantitative comparisons have been conducted against the current SOTA methods on four datasets, highlighting their limitations and pointing out future research directions.

:fire::fire::fire::fire: The latest deblurring papers of [CVPR 2024](https://cvpr.thecvf.com/Conferences/2024/AcceptedPapers) have been included~


## Content:

1. <a href="#survey">Remote Sensing Image Datasets</a>
2. <a href="#cnnmodels"> Remote Sensing Image Dehazing Methods Based on Traditional Image Processing(Image enhancement)</a>
3.  <a href="#rnnmodels"> Remote Sensing Image Dehazing Methods Based on Physical Models</a>
4. <a href="#ganmodels"> Remote Sensing Image Dehazing Methods Based on Prior Knowledge </a>
5. <a href="#tmodels"> CNN - based Remote Sensing Image Dehazing Methods </a>
6. <a href="#diffmodels"> Transformer - based Remote Sensing Image Dehazing Methods </a>
7. <a href="#datasets"> Diffusion-based Remote Sensing Image Dehazing Methods</a>
8. <a href="#evaluation"> Evaluation </a>
9. <a href="#citation"> Citation </a>

------

# 0. Remote Sensing Image Datasets:  <a id="survey" class="anchor" href="#survey" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
**No.** |**Dataset** | **Year** | **Pub.** |**Size** |  **Types** | **Train/Val/Test**  | **Download**
:-: | :-: | :-: | :-:  | :-:  | :-: | :- | :-:
01   | [**SateHaze1k**](https://openaccess.thecvf.com/content_WACV_2020/papers/Huang_Single_Satellite_Optical_Imagery_Dehazing_using_SAR_Image_Prior_Based_WACV_2020_paper.pdf) | 2017 | WACV | 400*3 | Synthetic | -  | [link](https://www.dropbox.com/s/k2i3p7puuwl2g59/Haze1k.zip?dl=0) 
02   | [**RICE**](https://arxiv.org/abs/1901.00600) | 2019 | - | 950 | Synthetic | - | [link](https://github.com/BUPTLdy/RICE_DATASET) 
03 | [**AID**](https://ieeexplore.ieee.org/document/7907303) | 2017 | TGRS | 10000 | Synthetic | -  | [link](https://opendatalab.org.cn/OpenDataLab/AID) 
04 | [**RS-Haze**](https://ieeexplore.ieee.org/document/10076399) | 2023 | TIP | 54000 | Real | 51300/0/2700 | [link](https://github.com/IDKiro/DehazeFormer) 
05 | [**I-HAZE**](https://arxiv.org/abs/1804.05091) | 2018 | ACIVS | 35 | Real | -  | [link](https://data.vision.ee.ethz.ch/cvl/ntire18//i-haze/) 
06 | [**O-HAZE**](https://ieeexplore.ieee.org/document/8575270) | 2018 | CVPRW | 45 | Real | -                  | [link](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/) 
07 | [**RESIDE**](https://ieeexplore.ieee.org/document/8451944) | 2018 | TIP | 13990 | Synthetic | 13000/0/990 | [link](https://sites.google.com/view/reside-dehaze-datasets) 

------










# 1. Remote Sensing Image Dehazing Methods Based on Traditional Image Processing(Image enhancement):  <a id="cnnmodels" class="anchor" href="#CNNmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2024-05-14) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2023 | Efficient Dehazing Method | JSTARS | Efficient Dehazing Method for Outdoor and Remote Sensing Images | [Paper](https://ieeexplore.ieee.org/document/10122596)/[Project] 
02 | 2019 | AHE(直方图均衡化) | APCC | Single Image Dehazing Based on Adaptive Histogram Equalization and Linearization of Gamma Correction | [Paper](https://ieeexplore.ieee.org/document/9026457)/[Project]
03 | 2022 | CLAHEMSF | MTA | Single image haze removal using contrast limited adaptive histogram equalization based multiscale fusion technique | [Paper](https://link.springer.com/article/10.1007/s11042-021-11890-0)/[Project]
04 | 2020 | URSHR | IEEE Access | A New Haze Removal Algorithm for Single Urban Remote Sensing Image | [Paper](https://ieeexplore.ieee.org/abstract/document/9102275)/Project
05 | 2018 |  | GRSL | A Framework for Outdoor RGB Image Enhancement and Dehazing | [Paper](https://ieeexplore.ieee.org/abstract/document/8331851)/[Project]
                                   

















# 2. Remote Sensing Image Dehazing Methods Based on Physical Models(Image Restoration):  <a id="rnnmodels" class="anchor" href="#RNNmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2024-05-14) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2023 | SRD | Remote Sensing | Remote Sensing Image Haze Removal Based on Superpixel | [Paper](https://www.mdpi.com/2072-4292/15/19/4680)/[Project]
02 | 2023 | Efficient Dehazing Method | JSTARS | Efficient Dehazing Method for Outdoor and Remote Sensing Images | [Paper](https://ieeexplore.ieee.org/document/10122596)/[Project] 
03 | 2023 | RLDP | Remote Sensing | Single Remote Sensing Image Dehazing Using Robust Light-Dark Prior | [Paper](https://www.mdpi.com/2072-4292/15/4/938)/[Project]
04 | 2019 | DCP | CVPR | Single image haze removal using dark channel prior | [Paper](https://ieeexplore.ieee.org/document/5206515)/[Project] 







# 3. Remote Sensing Image Dehazing Methods Based on Prior Knowledge:  <a id="ganmodels" class="anchor" href="#GANmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2025-02-12) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2024 | ALFE | TGRS | A Remote Sensing Image Dehazing Method Based on Heterogeneous Priors | [Paper](https://ieeexplore.ieee.org/document/10476500 )/[Project] 
02 | 2023 | HALP | TGRS | Remote Sensing Image Dehazing Using  Heterogeneous Atmospheric Light Prior | [Paper](https://ieeexplore.ieee.org/document/10050029)/[Project](https://github.com/foreverfruit/HALP)
03 | 2022 | GPD-Net | GRSL | Single Remote Sensing Image Dehazing Using Gaussian and Physics-Guided Process | [Paper](https://ieeexplore.ieee.org/document/9780137)/[Project]
04 | 2023 | saliency-guided parallel learning mechanism | GRSL | UAV Image Haze Removal Based on Saliency- Guided Parallel Learning Mechanism | [Paper](https://ieeexplore.ieee.org/document/10016637)/[Project]
05 | 2023 | RLDP | Remote Sensing | Single Remote Sensing Image Dehazing Using Robust Light-Dark Prior | [Paper](https://www.mdpi.com/2072-4292/15/4/938)/[Project]
 06 | 2019 | DADN |Remote Sensing|Single Remote Sensing Image Dehazing Using a Prior-Based Dense Attentive Network|[Paper](https://www.mdpi.com/2072-4292/11/24/3008)/[Project]














# 4. CNN - based Remote Sensing Image Dehazing Methods:  <a id="tmodels" class="anchor" href="#Tmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2024-05-14) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2021 |Uformer| CVPR | Uformer: A general u-shaped transformer for image restoration | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Uformer_A_General_U-Shaped_Transformer_for_Image_Restoration_CVPR_2022_paper.pdf)/[Project](https://github.com/ZhendongWang6/Uformer)
02 | 2022 | Restormer | CVPR | Restormer: Efficient transformer for high-resolution image restoration | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf)/[Project](https://github.com/swz30/Restormer)
03 | 2022 | Stripformer | ECCV | Stripformer: Strip transformer for fast image deblurring | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-19800-7_9)/[Project](https://github.com/pp00704831/Stripformer-ECCV-2022-)
04 | 2022 | Stoformer | NeurIPS | Stochastic Window Transformer for Image Restoration | [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/3ca6d336ddaa316a6ae953a20b9477cf-Paper-Conference.pdf)/[Project](https://github.com/jiexiaou/Stoformer)
05 | 2023 | Sharpformer | TIP | SharpFormer: Learning Local Feature Preserving Global Representations for Image Deblurring | [Paper](https://ieeexplore.ieee.org/document/10124841)/[Project](https://github.com/qingsenyangit/SharpFormer)
06 | 2023 | FFTformer | CVPR | Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Efficient_Frequency_Domain-Based_Transformers_for_High-Quality_Image_Deblurring_CVPR_2023_paper.pdf)/[Project](https://github.com/kkkls/fftformer)
07 | 2023 | BiT | CVPR | Blur Interpolation Transformer for Real-World Motion from Blur | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhong_Blur_Interpolation_Transformer_for_Real-World_Motion_From_Blur_CVPR_2023_paper.pdf)/[Project](https://github.com/zzh-tech/bit)
08 | 2024 | | CVPR | Efficient Multi-scale Network with Learnable Discrete Wavelet Transform for Blind Motion Debluring| [Paper]/[Project]
09 | 2024 | | TNNLS |Image Deblurring by Exploring In-Depth Properties of Transformer| [Paper]/[Project](https://github.com/erfect2020/TransformerPerceptualLoss)








# 5. Transformer - based Remote Sensing Image Dehazing Methods:  <a id="diffmodels" class="anchor" href="#Diffmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2025-02-12) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2024 | Dehaze-TGGAN | TGRS | Dehaze-TGGAN: Transformer-Guide Generative Adversarial Networks With Spatial-Spectrum Attention for Unpaired Remote Sensing Dehazing | [Paper](https://ieeexplore.ieee.org/document/10614150)/[Project]
02 | 2024 |PGSformer| GRSL | Prompt-Guided Sparse Transformer for Remote Sensing Image Dehazing | [Paper](https://ieeexplore.ieee.org/document/10648722)/[Project]
03 | 2023 | ARDD-Net | GRSL | Remote Sensing Image Dehazing Using Adaptive Region-Based Diffusion Models | [Paper](https://ieeexplore.ieee.org/document/10233893)/[Project] 
04 | 2023 | RSDformer | GRSL | Learning an Effective Transformer for Remote Sensing Satellite Image Dehazing |[Paper](https://ieeexplore.ieee.org/document/10265239)/[Project](https://github.com/MingTian99/RSDformer)
05 | 2024 | ASTA | GRSL | Additional Self-Attention Transformer With Adapter for Thick Haze Removal |[Paper](https://ieeexplore.ieee.org/document/10443626)/[Project](https://github.com/Eric3200C/ASTA)



------













#  6. Diffusion-based Remote Sensing Image Dehazing Methods:  <a id="datasets" class="anchor" href="#datasets" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2024-01-08) :balloon:
| **No.** | **Year** | **Model** | **Pub.** | **Title**                                                    |                          **Links**                           |
| :-----: | :------: | :-------: | :------- | :----------------------------------------------------------- | :----------------------------------------------------------: |
|   01    |   2024   | ADND-Net  | GRSL     | Diffusion Models Based Null-Space Learning for Remote Sensing Image Dehazing | [Paper](https://ieeexplore.ieee.org/document/10445700)/[Project] |

------

# 7. Evaluation:  <a id="evaluation" class="anchor" href="#evaluation" aria-hidden="true"><span class="octicon octicon-link"></span></a>  

* For evaluation on **GoPro** results in MATLAB, modify './out/...' to the corresponding path
```matlab
evaluation_GoPro.m
```
* For evaluation on **HIDE** results in MATLAB, modify './out/...' to the corresponding path
```matlab
evaluation_HIDE.m
```
* For evaluation on **RealBlur_J** results, modify './out/...' to the corresponding path
```python
python evaluate_RealBlur_J.py
```
* For evaluation on **RealBlur_R** results, modify './out/...' to the corresponding path
```python
python evaluate_RealBlur_R.py
```


------

# Citation: <a id="citation" class="anchor" href="#citation" aria-hidden="true"><span class="octicon octicon-link"></span></a> 

If you find [our survey paper](https://doi.org/10.1007/s00371-024-03632-8) and evaluation code are useful, please cite the following paper:
```BibTeX
@article{xiang2024deep,
  title={Deep learning in motion deblurring: current status, benchmarks and future prospects},
  author={Xiang, Yawen and Zhou, Heng and Li, Chengyang and Sun, Fangwei and Li, Zhongbo and Xie, Yongqiang},
  journal={The Visual Computer},
  pages={1--27},
  year={2024},
  publisher={Springer}
}
```


--------------------------------------------------------------------------------------

# :clap::clap::clap: Thanks to the above authors for their excellent work！
