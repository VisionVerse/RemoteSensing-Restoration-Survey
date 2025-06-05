# Remote Sensing Image Dehazing: Recent Progress, Challenges, and Future Directions

:fire::fire: In this review, we have systematically examined **over 100 papers** :page_with_curl::page_with_curl::page_with_curl:, summarizing and analyzing :star2:**more than** 30 Remote Sensing Image Restoration methods. 

:fire::fire::fire: Extensive qualitative and quantitative comparisons have been conducted against the current SOTA methods on four datasets, highlighting their limitations and pointing out future research directions.


## Content:

0. <a href="#Datasets">Remote Sensing Image Datasets</a>
1. <a href="#Traditional Methods"> Traditional Remote Sensing Image Restoration Methods</a>
2. <a href="#CNNmodels"> CNN - based Remote Sensing Image Restoration Methods </a>
3. <a href="#GANmodels">GAN - based Remote Sensing Image Restoration Methods </a>
4. <a href="#Transformer">Transformer - based Remote Sensing Image Restoration Methods </a>
5. <a href="#Diffusion"> Diffusion-based Remote Sensing Image Restoration Methods</a>
6. <a href="#evaluation"> Evaluation </a>
7. <a href="#citation"> Citation </a>

------


# 0. Remote Sensing Image Datasets:  <a id="Datasets" class="anchor" href="#Datasets" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
**No.** |**Dataset** | **Year** | **Pub.** |**Size** |  **Types** | **Download**
:-: | :-: | :-: | :-:  | :-:  | :-: | :-:
01   | [**SateHaze1k**](https://openaccess.thecvf.com/content_WACV_2020/papers/Huang_Single_Satellite_Optical_Imagery_Dehazing_using_SAR_Image_Prior_Based_WACV_2020_paper.pdf) | 2017 | WACV | 400*3 | Synthetic | -  | [link](https://www.dropbox.com/s/k2i3p7puuwl2g59/Haze1k.zip?dl=0) 
02   | [**RICE**](https://arxiv.org/abs/1901.00600) | 2019 | arXiv | 950 | Synthetic  | [link](https://github.com/BUPTLdy/RICE_DATASET) 
03 | [**AID**](https://ieeexplore.ieee.org/document/7907303) | 2017 | TGRS | 10000 | Synthetic  | [link](https://opendatalab.org.cn/OpenDataLab/AID) 
04 | [**RS-Haze**](https://ieeexplore.ieee.org/document/10076399) | 2023 | TIP | 54000 | Real | [link](https://github.com/IDKiro/DehazeFormer) 
05 | [**I-HAZE**](https://arxiv.org/abs/1804.05091) | 2018 | ACIVS | 35 | Real  | [link](https://data.vision.ee.ethz.ch/cvl/ntire18//i-haze/) 
06 | [**O-HAZE**](https://ieeexplore.ieee.org/document/8575270) | 2018 | CVPRW | 45 | Real | [link](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/) 
07 | [**RESIDE**](https://ieeexplore.ieee.org/document/8451944) | 2018 | TIP | 13990 | Synthetic | [link](https://sites.google.com/view/reside-dehaze-datasets) 
08 | [**RSID**](https://ieeexplore.ieee.org/abstract/document/10149032) | 2023 | TGRS | 1000 | Synthetic | [link](https://github.com/chi-kaichen/Trinity-Net) 
09 | [**UBCSet**](https://ieeexplore.ieee.org/abstract/document/10149032) | 2024 | ISPRS | 5911 | Synthetic | [link](https://github.com/Liying-Xu/TCBC) 
10 | [**WHUS2-CR**](https://github.com/Neooolee/WHUS2-CR) | 2021 | - | 24450 | real |[link](https://github.com/Neooolee/WHUS2-CR)
11 | [**SEN12MS-CR**](https://patricktum.github.io/cloud_removal/sen12mscr/) | 2018 | - | 122218 | real  |[link](https://patricktum.github.io/cloud_removal/sen12mscr/)

------


# 1. Traditional Methods:  <a id="Traditional Methods" class="anchor" href="#Traditional Methods" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2025-06-04) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2015 | DHIM | IEEE Signal Processing Letters | Haze removal for a single remote sensing image based on deformed haze imaging model |[Paper](https://ieeexplore.ieee.org/abstract/document/7105841)/[Project]
02 | 2017 | HTM | Signal Processing | Haze removal for a single visible remote sensing image |[Paper](https://www.sciencedirect.com/science/article/pii/S0165168417300464)/[Project]
03 | 2018 |  | GRSL | A Framework for Outdoor RGB Image Enhancement and Dehazing | [Paper](https://ieeexplore.ieee.org/abstract/document/8331851)/[Project]
04 | 2018 | SMIDCP | GRSL | Haze and thin cloud removal via sphere model improved dark channel prior | [Paper](https://ieeexplore.ieee.org/document/8500152)/[Project] 
05 | 2018 | SMIDCP | GRSL | Haze and thin cloud removal via sphere model improved dark channel prior | [Paper](https://ieeexplore.ieee.org/document/8500152)/[Project] 
06 | 2019 | AHE | APCC | Single Image Dehazing Based on Adaptive Histogram Equalization and Linearization of Gamma Correction | [Paper](https://ieeexplore.ieee.org/document/9026457)/[Project]
07 | 2019 | DADN |Remote Sensing|Single Remote Sensing Image Dehazing Using a Prior-Based Dense Attentive Network|[Paper](https://www.mdpi.com/2072-4292/11/24/3008)/[Project]
08 | 2020 | URSHR | IEEE Access | A New Haze Removal Algorithm for Single Urban Remote Sensing Image | [Paper](https://ieeexplore.ieee.org/abstract/document/9102275)/[Project]
09 | 2020 | CR-GAN-PM| ISPRS | Thin cloud removal in optical remote sensing images based on generative adversarial networks and physical model of cloud distortion |[Paper](https://www.sciencedirect.com/science/article/pii/S0924271620301787)/[Project](https://github.com/Neooolee/CR-GAN-PM)
10 | 2021 | Kang et al. | TGRS | Fog Model-Based Hyperspectral Image Defogging |[Paper](https://ieeexplore.ieee.org/document/9511329)/[Project]
11 | 2021 | MDCP | GRSL | A novel thin cloud removal method based on multiscale dark channel prior| [Paper](https://ieeexplore.ieee.org/document/9381399)/[Project] 
12 | 2022 | CLAHEMSF | MTA | Single image haze removal using contrast limited adaptive histogram equalization based multiscale fusion technique | [Paper](https://link.springer.com/article/10.1007/s11042-021-11890-0)/[Project]
13 | 2022 | GPD-Net | GRSL | Single Remote Sensing Image Dehazing Using Gaussian and Physics-Guided Process | [Paper](https://ieeexplore.ieee.org/document/9780137)/[Project]
14 | 2022 | EVPM | Information Sciences | Local patchwise minimal and maximal values prior for single optical remote sensing image dehazing |[Paper](https://www.sciencedirect.com/science/article/pii/S0020025522004534)/[Project]
15 | 2022 | IdeRs | Information Sciences | IDeRs: Iterative dehazing method for single remote sensing image |[Paper](https://www.sciencedirect.com/science/article/pii/S0020025519301732)/[Project]
16 | 2023 | saliency-guided parallel learning mechanism | GRSL | UAV Image Haze Removal Based on Saliency- Guided Parallel Learning Mechanism | [Paper](https://ieeexplore.ieee.org/document/10016637)/[Project]
17 | 2023 | ED | JSTARS | Efficient Dehazing Method for Outdoor and Remote Sensing Images | [Paper](https://ieeexplore.ieee.org/document/10122596)/[Project] 
18 | 2023 | SRD | Remote Sensing | Remote Sensing Image Haze Removal Based on Superpixel | [Paper](https://www.mdpi.com/2072-4292/15/19/4680)/[Project]
19 | 2023 | Efficient Dehazing Method | JSTARS | Efficient Dehazing Method for Outdoor and Remote Sensing Images | [Paper](https://ieeexplore.ieee.org/document/10122596)/[Project] 
20 | 2023 | RLDP | Remote Sensing | Single Remote Sensing Image Dehazing Using Robust Light-Dark Prior | [Paper](https://www.mdpi.com/2072-4292/15/4/938)/[Project]
21 | 2023 | HALP | TGRS | Remote Sensing Image Dehazing Using  Heterogeneous Atmospheric Light Prior | [Paper](https://ieeexplore.ieee.org/document/10050029)/[Project](https://github.com/foreverfruit/HALP)
22 | 2024 | ALFE | TGRS | A Remote Sensing Image Dehazing Method Based on Heterogeneous Priors | [Paper](https://ieeexplore.ieee.org/document/10476500 )/[Project] 

------


# 2. CNN-based Models:  <a id="CNNmodels" class="anchor" href="#CNNmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2025-06-04) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2016 | Ren et al. | ECCV | Single image dehazing via multi-scale convolutional neural networks | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_10)/[Project]
02 | 2019 |RSC-Net | ISPRS |Thin cloud removal with residual symmetrical concatenation network| [Paper](https://www.sciencedirect.com/science/article/pii/S092427161930125X)/[Project]
03 | 2020 | RSDehazeNet | TGRS | RSDehazeNet: Dehazing network with channel refinement for multispectral remote sensing images | [Paper](https://ieeexplore.ieee.org/document/9134800)/[Project](https://github.com/tianqiong123/RSDehazeNet)
04 | 2020 | FCTF-Net | GRSL | A coarse-to-fine two-stage attentive network for haze removal of remote sensing images | [Paper](https://ieeexplore.ieee.org/document/9136742)/[Project](https://github.com/Neooolee/FCTF-Net?tab=readme-ov-file)
05 | 2020 |UCR | TGRS |Single image cloud removal using U-Net and generative adversarial networks| [Paper](https://ieeexplore.ieee.org/document/9224941)/[Project]
06 | 2020 |DSen2-CR | ISPRS |Cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion| [Paper](https://www.sciencedirect.com/science/article/pii/S0924271620301398)/[Project]
07 | 2021 | Zi et al.| JSTARS | Thin cloud removal for multispectral remote sensing images using convolutional neural networks combined with an imaging model | [Paper](https://ieeexplore.ieee.org/document/9384224)/[Project]
08 | 2022 |DCIL | TGRS |Dense haze removal based on dynamic collaborative inference learning for remote sensing images| [Paper](https://ieeexplore.ieee.org/document/9895281)/[Project](https://github.com/Shan-rs/DCI-Net)
09 | 2022 |SG-Net | ISPRS |A spectral grouping-based deep learning model for haze removal of hyperspectral images| [Paper](https://www.sciencedirect.com/science/article/pii/S0924271622001046)/[Project](https://github.com/SZU-AdvTech-2022/158-A-Spectral-Grouping-based-Deep-Learning-Model-for-Haze-Removal-of-Hyperspectral-Images)
10 | 2022 |GLF-CR | ISPRS |GLF-CR: SAR-enhanced cloud removal with global–local fusion| [Paper](https://www.sciencedirect.com/science/article/pii/S0924271622002064)/[Project](https://github.com/xufangchn/GLF-CR)
11 | 2022 |mutually beneficial guides | ISPRS |Semi-supervised thin cloud removal with mutually beneficial guides| [Paper](https://www.sciencedirect.com/science/article/pii/S0924271622002350)/[Project]
12 | 2022 |NE module | CVPRW |Nonuniformly Dehaze Network for Visible Remote Sensing Images| [Paper](https://ieeexplore.ieee.org/document/9857255)/[Project]
13 | 2023 | MSDA-CR | GRSL | Cloud removal in optical remote sensing imagery using multiscale distortion-aware networks | [Paper](https://ieeexplore.ieee.org/document/9686746)/[Project]
14 | 2023 |EMPF-Net | TGRS |Encoder-free multiaxis physics-aware fusion network for remote sensing image dehazing| [Paper](https://ieeexplore.ieee.org/document/10287960)/[Project](https://github.com/chdwyb/EMPF-Net)
15 | 2023 |SFAN | TGRS |Spatial-frequency adaptive remote sensing image dehazing with mixture of experts| [Paper](https://ieeexplore.ieee.org/abstract/document/10679156)/[Project](https://github.com/it-hao/SFAN)
16 | 2023 |PSMB-Net | TGRS |Partial siamese with multiscale bi-codec networks for remote sensing image haze removal| [Paper](https://ieeexplore.ieee.org/abstract/document/10268954)/[Project](https://github.com/thislzm/PSMB-Net)
17 | 2023 |HS2P | Information Fusion |HS2P: Hierarchical spectral and structure-preserving fusion network for multimodal remote sensing image cloud and shadow removal| [Paper](https://www.sciencedirect.com/science/article/pii/S1566253523000453)/[Project](https://github.com/weifanyi515/HS2P)
18 | 2023 |CP-FFCN | ISPRS |Blind single-image-based thin cloud removal using a cloud perception integrated fast Fourier convolutional network| [Paper](https://www.sciencedirect.com/science/article/pii/S0924271623002903)/[Project]
19 | 2024 |EDED-Net | Remote Sensing |End-to-end detail-enhanced dehazing network for remote sensing images| [Paper](https://www.mdpi.com/2072-4292/16/2/225)/[Project]
20 | 2024 |ConvIR | TPAMI |Revitalizing Convolutional Network for Image Restoration| [Paper](https://ieeexplore.ieee.org/abstract/document/10571568)/[Project](https://github.com/c-yn/ConvIR)
21 | 2024 |PhDnet | Information Fusion |PhDnet: A novel physic-aware dehazing network for remote sensing images| [Paper](https://www.sciencedirect.com/science/article/pii/S1566253524000551#b12)/[Project](https://github.com/colacomo/PhDnet)
22 | 2024 |HyperDehazeNet | ISPRS |HyperDehazing: A hyperspectral image dehazing benchmark dataset and a deep learning model for haze removal| [Paper](https://www.sciencedirect.com/science/article/pii/S0924271624003721)/[Project]
23 | 2024 |HDRSA-Net | ISPRS |HDRSA-Net: Hybrid dynamic residual self-attention network for SAR-assisted optical image cloud and shadow removal| [Paper](https://www.sciencedirect.com/science/article/pii/S0924271624004039)/[Project](https://github.com/RSIIPAC/LuojiaSET-OSFCR)
24 | 2024 |ICL-Net | JSTARS |ICL-Net: Inverse cognitive learning network for remote sensing image dehazing| [Paper](https://ieeexplore.ieee.org/document/10665990)/[Project]
25 | 2024 |C2AIR | WACV |C2AIR: Consolidated Compact Aerial Image Haze Removal| [Paper](https://ieeexplore.ieee.org/document/10484204)/[Project](https://github.com/AshutoshKulkarni4998/C2AIR)
25 | 2025 |BMFH-Net | TCSVT |Bidirectional-Modulation Frequency-Heterogeneous Network for Remote Sensing Image Dehazing| [Paper](https://ieeexplore.ieee.org/document/11006655)/[Project](https://github.com/zqf2024/BMFH-Net)
26 | 2025 |HPN-CR | TGRS | HPN-CR: Heterogeneous Parallel Network for SAR-Optical Data Fusion Cloud Removal| [Paper](https://ieeexplore.ieee.org/document/10906642)/[Project](https://github.com/G-pz/HPN-CR)
27 | 2025 |DDIA-CFR | Information Fusion | Breaking through clouds: A hierarchical fusion network empowered by dual-domain cross-modality interactive attention for cloud-free image reconstruction| [Paper](https://www.sciencedirect.com/science/article/pii/S1566253524004275)/[Project]
28 | 2025 |SMDCNet | ISPRS |Cloud removal with optical and SAR imagery via multimodal similarity attention| [Paper](https://www.sciencedirect.com/science/article/pii/S0924271625001856)/[Project]
29 | 2025 |MIMJT | ECCV |Satellite Image Dehazing Via Masked Image Modeling and Jigsaw Transformation| [Paper](https://link.springer.com/chapter/10.1007/978-3-031-91838-4_27)/[Project]



------


# 3. GAN-based Methods:  <a id="GANmodels" class="anchor" href="#GANmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  

:rocket::rocket::rocket:Update (in 2025-06-04) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2018 | Cloud-GAN | IGARSS | Cloud-gan: Cloud removal for sentinel-2 imagery using a cyclic consistent generative adversarial networks | [Paper](https://ieeexplore.ieee.org/document/8519033)/[Project]
02 | 2020 | CR-GAN-PM | ISPRS | Thin cloud removal in optical remote sensing images based on generative adversarial networks and physical model of cloud distortion | [Paper](https://www.sciencedirect.com/science/article/pii/S0924271620301787)/[Project](https://github.com/Neooolee/CR-GAN-PM)
03 | 2020 |UCR | TGRS |Single image cloud removal using U-Net and generative adversarial networks| [Paper](https://ieeexplore.ieee.org/document/9224941)/[Project]
04 | 2020 | SpA-GAN| arXiv | Cloud Removal for Remote Sensing Imagery via Spatial Attention Generative Adversarial Network | [Paper](https://arxiv.org/abs/2009.13015)/[Project](https://github.com/Penn000/SpA-GAN_for_cloud_removal)
05 | 2020 | FCTF-Net | GRSL | A coarse-to-fine two-stage attentive network for haze removal of remote sensing images | [Paper](https://ieeexplore.ieee.org/document/9136742)/[Project](https://github.com/Neooolee/FCTF-Net?tab=readme-ov-file)
06 | 2020 | Huang et al. | WACV | Single Satellite Optical Imagery Dehazing using SAR Image Prior Based on conditional Generative Adversarial Networks | [Paper](https://ieeexplore.ieee.org/document/9093471)/[Project]
07 | 2021 | Darbaghshahi et al. | TGRS | Cloud removal in remote sensing images using generative adversarial networks and SAR-to-optical image translation | [Paper](https://ieeexplore.ieee.org/abstract/document/9627647)/[Project]
08 | 2021 | SkyGAN | WACV | Domain-Aware Unsupervised Hyperspectral Reconstruction for Aerial Image Dehazing | [Paper](https://ieeexplore.ieee.org/document/9423159)/[Project]
09 | 2022 |Dehaze-AGGAN| TGRS | Dehaze-AGGAN: Unpaired remote sensing image dehazing using enhanced attention-guide generative adversarial networks | [Paper](https://ieeexplore.ieee.org/document/9881213)/[Project]
10 | 2023 | MSDA-CR | GRSL | Cloud removal in optical remote sensing imagery using multiscale distortion-aware networks | [Paper](https://ieeexplore.ieee.org/document/9686746)/[Project]
11 | 2025 | MT_GAN | ISPRS | MT_GAN: A SAR-to-optical image translation method for cloud removal | [Paper](https://www.sciencedirect.com/science/article/pii/S0924271625001479)/[Project](https://github.com/NUAA-RS/MT_GAN)

------


# 4. Transformer-based Methods:  <a id="Transformer" class="anchor" href="#Transformer" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2025-06-04) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2022 | TransRA |Multidimensional Systems and Signal Processing | TransRA: Transformer and residual attention fusion for single remote sensing image dehazing | [Paper](https://link.springer.com/article/10.1007/s11045-022-00835-x)/[Project] 
02 | 2023 | DehazeFormer |TIP | Vision transformers for single image dehazing | [Paper](https://ieeexplore.ieee.org/abstract/document/10076399)/[Project](https://github.com/IDKiro/DehazeFormer)
03 | 2023 | FormerCR |Remote Sensing | Former-CR: A transformer-based thick cloud removal method with optical and SAR imagery | [Paper](https://www.mdpi.com/2072-4292/15/5/1196)/[Project]
04 | 2023 | RSDformer | GRSL | Learning an Effective Transformer for Remote Sensing Satellite Image Dehazing |[Paper](https://ieeexplore.ieee.org/document/10265239)/[Project](https://github.com/MingTian99/RSDformer)
05 | 2023 | Trinity-Net |TGRS | Trinity-Net: Gradient-guided Swin transformer-based remote sensing image dehazing and beyond | [Paper](https://ieeexplore.ieee.org/abstract/document/10149032)/[Project](https://github.com/chi-kaichen/Trinity-Net)
06 | 2023 | AIDTransformer | WACV | Aerial Image Dehazing with Attentive Deformable Transformers |[Paper](https://ieeexplore.ieee.org/document/10030985)/[Project](https://github.com/AshutoshKulkarni4998/AIDTransformer)
07 | 2024 | DCR-GLFT |TGRS | Density-aware Cloud Removal of Remote Sensing Imagery Using a Global-Local Fusion Transformer | [Paper](https://ieeexplore.ieee.org/document/10713444)/[Project]
08 | 2024 |SSGT | JSTARS | SSGT: Spatio-Spectral Guided Transformer for Hyperspectral Image Fusion Joint with Cloud Removal | [Paper](https://ieeexplore.ieee.org/document/10648722)/[Project]
09 | 2024 |PGSformer| GRSL | Prompt-Guided Sparse Transformer for Remote Sensing Image Dehazing | [Paper](https://ieeexplore.ieee.org/document/10648722)/[Project]
10 | 2024 | ASTA | GRSL | Additional Self-Attention Transformer With Adapter for Thick Haze Removal |[Paper](https://ieeexplore.ieee.org/document/10443626)/[Project](https://github.com/Eric3200C/ASTA)
11 | 2024 | Dehaze-TGGAN | TGRS | Dehaze-TGGAN: Transformer-Guide Generative Adversarial Networks With Spatial-Spectrum Attention for Unpaired Remote Sensing Dehazing | [Paper](https://ieeexplore.ieee.org/document/10614150)/[Project]
12 | 2025 | DehazeXL | CVPR | Tokenize Image Patches: Global Context Fusion for Effective Haze Removal in Large Images |[Paper](https://arxiv.org/abs/2504.09621)/[Project](https://github.com/CastleChen339/DehazeXL)
13 | 2025 | DecloudFormer | Pattern Recognition | DecloudFormer: Quest the key to consistent thin cloud removal of wide-swath multi-spectral images |[Paper](https://www.sciencedirect.com/science/article/pii/S0031320325003243)/[Project](https://github.com/N1rv4n4/DecloudFormer)


------


#  5. Diffusion-based Methods:  <a id="Diffusion" class="anchor" href="#Diffusion" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2025-06-04) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2023 | ARDD-Net | GRSL | Remote Sensing Image Dehazing Using Adaptive Region-Based Diffusion Models | [Paper](https://ieeexplore.ieee.org/document/10233893)/[Project]
02 | 2023 |SeqDMs | Remote Sensing | Cloud removal in remote sensing using sequential-based diffusion models | [Paper](https://www.mdpi.com/2072-4292/15/11/2861)/[Project]
03 | 2024 |ADND-Net | GRSL | Diffusion Models Based Null-Space Learning for Remote Sensing Image Dehazing | [Paper](https://ieeexplore.ieee.org/document/10445700)/[Project] 
04 | 2024 |RSHazeDiff | T-ITS | RSHazeDiff: A unified Fourier-aware diffusion model for remote sensing image dehazing | [Paper](https://ieeexplore.ieee.org/document/10747754)/[Project](https://github.com/jm-xiong/RSHazeDiff)
05 | 2024 |IDF-CR | TGRS | IDF-CR: Iterative diffusion process for divide-and-conquer cloud removal in remote-sensing images | [Paper](https://ieeexplore.ieee.org/abstract/document/10474382)/[Project](https://github.com/SongYxing/IDF-CR)
06 | 2025 |EMRDM | CVPR | Effective Cloud Removal for Remote Sensing Images by an Improved Mean-Reverting Denoising Model with Elucidated Design Space | [Paper](https://arxiv.org/abs/2503.23717)/[Project](https://github.com/Ly403/EMRDM)

------


# 6. Evaluation:  <a id="evaluation" class="anchor" href="#evaluation" aria-hidden="true"><span class="octicon octicon-link"></span></a>  

* For evaluation on **RICE**, **StateHaze1k** results, modify 'test_original' and 'test_restored' to the corresponding path
```python
python evaluate.py -to [ground-truth image path] -td [restored image path]
```

------


# Citation: <a id="citation" class="anchor" href="#citation" aria-hidden="true"><span class="octicon octicon-link"></span></a> 

<!--If you find [our survey paper](https://doi.org/10.1007/s00371-024-03632-8) and evaluation code are useful, please cite the following paper:
```BibTeX

```-->

--------------------------------------------------------------------------------------

# :clap::clap::clap: Thanks to the above authors for their excellent work！
