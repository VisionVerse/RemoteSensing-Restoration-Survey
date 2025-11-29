# Remote Sensing Image Dehazing: A Systematic Review of Progress, Challenges, and Prospects [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)![Stars](https://img.shields.io/github/stars/VisionVerse/RemoteSensing-Restoration-Survey)



For the purposes of this review, **we adopt an inclusive definition of “dehazing” that encompasses all methodologies designed to mitigate the effects of fog, haze, and optically cloud layers.**
In this review, we have systematically examined **over 200 papers** :scroll::scroll::scroll:, summarizing and analyzing **more than** 80 Remote Sensing Image Dehazing methods. 

:heart_eyes_cat: If this work is helpful for you, please help star this repo. Thanks! 

## :mega: News
- **2025/08/29**: Added 2 TGRS 2025 papers, 1 TGRS 2024 paper.
- **2025/07/26**: Added 2 JSTARS 2025 papers, 1 GRSL 2025 paper.
- **2025/07/22**: Added 3 TGRS 2025 papers, 1 EAAI 2025 paper, 1 ISPRS P&RS 2024 paper and 1 Signal Processing 2025 paper.
- **2025/06/28**: Paper submitted.
- **2025/05/15**: Added 2 CVPR 2025 papers.


## :pizza: Introduction
Remote sensing images (RSIs) are frequently degraded by atmospheric interferences such as haze, fog, and thin clouds, which significantly hinder surface observation and subsequent analytical tasks. 
As a vital preprocessing step, image dehazing plays a critical role in improving data usability for applications including environmental monitoring, disaster response, and military reconnaissance.
As show in Fig. 1, this paper presents a comprehensive and in-depth survey of recent developments in RSIs dehazing, covering traditional image enhancement and physics models, as well as state-of-the-art deep learning frameworks, including deep convolution, adversarial generation, vision Transformer, and diffusion generation.
We not only categorize and summarize representative algorithms but also provide a detailed comparative analysis of their strengths and limitations from algorithmic, theoretical, and application perspectives.
To provide a practical reference for performance evaluation, we conduct large-scale quantitative and qualitative experiments on four public datasets: SateHaze1k, RICE, LHID, and DHID, enabling fair and reproducible benchmarking of various dehazing methods.
In addition, we discuss key technical challenges in Fig.2, such as dynamic atmospheric modeling, multi-modal data fusion, lightweight model design, data scarcity, and joint degradation scenarios, and propose future research directions.


![avatar](/Taxonomy.png)
**Fig 1.** Taxonomy of Remote Sensing Image Dehazing Methods.


## Content:

0. <a href="#Datasets">Remote Sensing Image Datasets</a>
1. <a href="#Traditional Methods"> Traditional Remote Sensing Image Restoration Methods</a>
2. <a href="#CNNmodels"> Deep Convolution for Remote Sensing Image Dehazing </a>
3. <a href="#GANmodels"> Adversarial Generation for Remote Sensing Image Dehazing </a>
4. <a href="#Transformer"> Vision Transformer for Remote Sensing Image Dehazing </a>
5. <a href="#Diffusion"> Diffusion Generation for Remote Sensing Image Dehazing </a>
6. <a href="#prospects"> Current Challenges and Future Prospects </a>
7. <a href="#evaluation"> Evaluation </a>


------


# :open_file_folder: Remote Sensing Image Datasets:  <a id="Datasets" class="anchor" href="#Datasets" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
**No.** |**Dataset** | **Year** | **Pub.** |**Number** | **Image Size** |  **Types** | **Download**
:-: | :-: | :-: | :-:  | :-:  | :-:  | :-: | :-:
01   | [**SateHaze1k**](https://ieeexplore.ieee.org/document/9093471) | 2017 | WACV | 400*3 | 512×512 | Synthetic| [link](https://www.dropbox.com/s/k2i3p7puuwl2g59/Haze1k.zip?dl=0) 
02   | [**RICE**](https://arxiv.org/abs/1901.00600) | 2019 | arXiv | 950| 512×512 | Synthetic  | [link](https://github.com/BUPTLdy/RICE_DATASET) 
03 | [**AID**](https://ieeexplore.ieee.org/document/7907303) | 2017 | TGRS | 10000 | 600×600 | Synthetic  | [link](https://opendatalab.org.cn/OpenDataLab/AID) 
04 | [**RS-Haze**](https://ieeexplore.ieee.org/document/10076399) | 2023 | TIP | 54000 | 512×512 | Real | [link](https://github.com/IDKiro/DehazeFormer) 
05 | [**LHID**](https://ieeexplore.ieee.org/document/9895281) | 2022 | TGRS | 31017 | 512×512 | Synthetic | [link](https://github.com/Shan-rs/DCI-Net?tab=readme-ov-file) 
06 | [**DHID**](https://ieeexplore.ieee.org/document/9895281) | 2022 | TGRS | 14990 | 512×512 | Synthetic | [link](https://github.com/Shan-rs/DCI-Net?tab=readme-ov-file) 
07 | [**HN-Snowy**](https://www.sciencedirect.com/science/article/pii/S0924271623002903) | 2022 | ISPRS P&RS | 1237 | 256×256 | Synthetic | [link](https://github.com/Merryguoguo/CP-FFCN) 
08 | [**CUHK-CR**](https://ieeexplore.ieee.org/document/10552304) | 2024 | TGRS | 1227 | 256×256 | Synthetic | [link](https://ieeexplore.ieee.org/document/10552304) 
09 | [**HRSI**](https://ieeexplore.ieee.org/document/10658989) | 2024 | TGRS | 796 | 512×512-4000×4000 | Synthetic | [link](https://ieeexplore.ieee.org/document/10658989) 
10 | [**RSID**](https://ieeexplore.ieee.org/abstract/document/10149032) | 2023 | TGRS | 1000 | 256×256 | Synthetic | [link](https://github.com/chi-kaichen/Trinity-Net) 
11 | [**UBCSet**](https://www.sciencedirect.com/science/article/pii/S0924271624003460) | 2024 | ISPRS P&RS | 5911 | 256×256 | Synthetic | [link](https://github.com/Liying-Xu/TCBC) 
12 | [**WHUS2-CR**](https://github.com/Neooolee/WHUS2-CR) | 2021 | - | 24450 | 64×64-384×384 | real |[link](https://github.com/Neooolee/WHUS2-CR)
13 | [**SEN12MS-CR**](https://patricktum.github.io/cloud_removal/sen12mscr/) | 2018 | - | 122218 | 256×256 | real  |[link](https://patricktum.github.io/cloud_removal/sen12mscr/)

------


# 1. Traditional Image Enhancement and Physics Model for Remote Sensing Image Dehazing:  <a id="Traditional Methods" class="anchor" href="#Traditional Methods" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2025-07-26) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2015 | DHIM | SPL | Haze removal for a single remote sensing image based on deformed haze imaging model |[Paper](https://ieeexplore.ieee.org/abstract/document/7105841)/[Project]
02 | 2017 | GRS-HTM | Signal Processing | Haze removal for a single visible remote sensing image |[Paper](https://www.sciencedirect.com/science/article/pii/S0165168417300464)/[Project]
03 | 2018 | HMF | GRSL | A Framework for Outdoor RGB Image Enhancement and Dehazing | [Paper](https://ieeexplore.ieee.org/abstract/document/8331851)/[Project]
04 | 2018 | SMIDCP | GRSL | Haze and thin cloud removal via sphere model improved dark channel prior | [Paper](https://ieeexplore.ieee.org/document/8500152)/[Project] 
05 | 2019 | AHE | APCC | Single Image Dehazing Based on Adaptive Histogram Equalization and Linearization of Gamma Correction | [Paper](https://ieeexplore.ieee.org/document/9026457)/[Project]
06 | 2019 | DADN |Remote Sensing|Single Remote Sensing Image Dehazing Using a Prior-Based Dense Attentive Network|[Paper](https://www.mdpi.com/2072-4292/11/24/3008)/[Project]
07 | 2019 | IDeRs | Information Sciences | IDeRs: Iterative dehazing method for single remote sensing image |[Paper](https://www.sciencedirect.com/science/article/pii/S0020025519301732)/[Project]
08 | 2020 | CR-GAN-PM| ISPRS P&RS | Thin cloud removal in optical remote sensing images based on generative adversarial networks and physical model of cloud distortion |[Paper](https://www.sciencedirect.com/science/article/pii/S0924271620301787)/[Project](https://github.com/Neooolee/CR-GAN-PM)
09 | 2021 | HID | TGRS | Fog Model-Based Hyperspectral Image Defogging |[Paper](https://ieeexplore.ieee.org/document/9511329)/[Project]
10 | 2021 | MDCP | GRSL | A novel thin cloud removal method based on multiscale dark channel prior| [Paper](https://ieeexplore.ieee.org/document/9381399)/[Project] 
11 | 2022 | CLAHEMSF | MTA | Single image haze removal using contrast limited adaptive histogram equalization based multiscale fusion technique | [Paper](https://link.springer.com/article/10.1007/s11042-021-11890-0)/[Project]
12 | 2022 | GPD-Net | GRSL | Single Remote Sensing Image Dehazing Using Gaussian and Physics-Guided Process | [Paper](https://ieeexplore.ieee.org/document/9780137)/[Project]
13 | 2022 | EVPM | Information Sciences | Local patchwise minimal and maximal values prior for single optical remote sensing image dehazing |[Paper](https://www.sciencedirect.com/science/article/pii/S0020025522004534)/[Project]
14 | 2023 | SGPLM | GRSL | UAV Image Haze Removal Based on Saliency- Guided Parallel Learning Mechanism | [Paper](https://ieeexplore.ieee.org/document/10016637)/[Project]
15 | 2023 | ED | JSTARS | Efficient Dehazing Method for Outdoor and Remote Sensing Images | [Paper](https://ieeexplore.ieee.org/document/10122596)/[Project] 
16 | 2023 | SRD | Remote Sensing | Remote Sensing Image Haze Removal Based on Superpixel | [Paper](https://www.mdpi.com/2072-4292/15/19/4680)/[Project]
17 | 2023 | RLDP | Remote Sensing | Single Remote Sensing Image Dehazing Using Robust Light-Dark Prior | [Paper](https://www.mdpi.com/2072-4292/15/4/938)/[Project]
18 | 2023 | HALP | TGRS | Remote Sensing Image Dehazing Using  Heterogeneous Atmospheric Light Prior | [Paper](https://ieeexplore.ieee.org/document/10050029)/[Project](https://github.com/foreverfruit/HALP)
19 | 2024 | ALFE | TGRS | A Remote Sensing Image Dehazing Method Based on Heterogeneous Priors | [Paper](https://ieeexplore.ieee.org/document/10476500 )/[Project] 

------


# 2. Deep Convolution for Remote Sensing Image Dehazing:  <a id="CNNmodels" class="anchor" href="#CNNmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2025-07-26) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2016 | MSDN | ECCV | Single image dehazing via multi-scale convolutional neural networks | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_10)/[Project]
02 | 2019 |RSC-Net | ISPRS P&RS |Thin cloud removal with residual symmetrical concatenation network| [Paper](https://www.sciencedirect.com/science/article/pii/S092427161930125X)/[Project]
03 | 2020 | RSDehazeNet | TGRS | RSDehazeNet: Dehazing network with channel refinement for multispectral remote sensing images | [Paper](https://ieeexplore.ieee.org/document/9134800)/[Project](https://github.com/tianqiong123/RSDehazeNet)
04 | 2020 | FCTF-Net | GRSL | A coarse-to-fine two-stage attentive network for haze removal of remote sensing images | [Paper](https://ieeexplore.ieee.org/document/9136742)/[Project](https://github.com/Neooolee/FCTF-Net?tab=readme-ov-file)
05 | 2020 |UCR | TGRS |Single image cloud removal using U-Net and generative adversarial networks| [Paper](https://ieeexplore.ieee.org/document/9224941)/[Project]
06 | 2020 |DSen2-CR | ISPRS P&RS |Cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion| [Paper](https://www.sciencedirect.com/science/article/pii/S0924271620301398)/[Project]
07 | 2021 | CNNIM| JSTARS | Thin cloud removal for multispectral remote sensing images using convolutional neural networks combined with an imaging model | [Paper](https://ieeexplore.ieee.org/document/9384224)/[Project]
08 | 2022 |DCIL | TGRS |Dense haze removal based on dynamic collaborative inference learning for remote sensing images| [Paper](https://ieeexplore.ieee.org/document/9895281)/[Project](https://github.com/Shan-rs/DCI-Net)
09 | 2022 |SG-Net | ISPRS P&RS |A spectral grouping-based deep learning model for haze removal of hyperspectral images| [Paper](https://www.sciencedirect.com/science/article/pii/S0924271622001046)/[Project](https://github.com/SZU-AdvTech-2022/158-A-Spectral-Grouping-based-Deep-Learning-Model-for-Haze-Removal-of-Hyperspectral-Images)
10 | 2022 |GLF-CR | ISPRS P&RS |GLF-CR: SAR-enhanced cloud removal with global–local fusion| [Paper](https://www.sciencedirect.com/science/article/pii/S0924271622002064)/[Project](https://github.com/xufangchn/GLF-CR)
11 | 2022 |MBG-CR | ISPRS P&RS |Semi-supervised thin cloud removal with mutually beneficial guides| [Paper](https://www.sciencedirect.com/science/article/pii/S0924271622002350)/[Project]
12 | 2022 |NE module | CVPRW |Nonuniformly Dehaze Network for Visible Remote Sensing Images| [Paper](https://ieeexplore.ieee.org/document/9857255)/[Project]
13 | 2023 | MSDA-CR | GRSL | Cloud removal in optical remote sensing imagery using multiscale distortion-aware networks | [Paper](https://ieeexplore.ieee.org/document/9686746)/[Project]
14 | 2023 |EMPF-Net | TGRS |Encoder-free multiaxis physics-aware fusion network for remote sensing image dehazing| [Paper](https://ieeexplore.ieee.org/document/10287960)/[Project](https://github.com/chdwyb/EMPF-Net)
15 | 2023 |PSMB-Net | TGRS |Partial siamese with multiscale bi-codec networks for remote sensing image haze removal| [Paper](https://ieeexplore.ieee.org/abstract/document/10268954)/[Project](https://github.com/thislzm/PSMB-Net)
16 | 2023 |HS2P | Information Fusion |HS2P: Hierarchical spectral and structure-preserving fusion network for multimodal remote sensing image cloud and shadow removal| [Paper](https://www.sciencedirect.com/science/article/pii/S1566253523000453)/[Project](https://github.com/weifanyi515/HS2P)
17 | 2023 |CP-FFCN | ISPRS P&RS | Blind single-image-based thin cloud removal using a cloud perception integrated fast Fourier convolutional network| [Paper](https://www.sciencedirect.com/science/article/pii/S0924271623002903)/[Project]
18 | 2023 | GHRN | JAG | Incorporating inconsistent auxiliary images in haze removal of very high resolution images | [Paper](https://www.sciencedirect.com/science/article/pii/S1569843223001395)/[Project]
19 | 2024 |SFAN | TGRS |Spatial-frequency adaptive remote sensing image dehazing with mixture of experts| [Paper](https://ieeexplore.ieee.org/abstract/document/10679156)/[Project](https://github.com/it-hao/SFAN)
20 | 2024 |EDED-Net | Remote Sensing |End-to-end detail-enhanced dehazing network for remote sensing images| [Paper](https://www.mdpi.com/2072-4292/16/2/225)/[Project]
21 | 2024 |ConvIR | TPAMI | Revitalizing Convolutional Network for Image Restoration| [Paper](https://ieeexplore.ieee.org/abstract/document/10571568)/[Project](https://github.com/c-yn/ConvIR)
22 | 2024 |PhDnet | Information Fusion | PhDnet: A novel physic-aware dehazing network for remote sensing images| [Paper](https://www.sciencedirect.com/science/article/pii/S1566253524000551#b12)/[Project](https://github.com/colacomo/PhDnet)
23 | 2024 |HyperDehazeNet | ISPRS P&RS | HyperDehazing: A hyperspectral image dehazing benchmark dataset and a deep learning model for haze removal| [Paper](https://www.sciencedirect.com/science/article/pii/S0924271624003721)/[Project]
24 | 2024 |HDRSA-Net | ISPRS P&RS |HDRSA-Net: Hybrid dynamic residual self-attention network for SAR-assisted optical image cloud and shadow removal| [Paper](https://www.sciencedirect.com/science/article/pii/S0924271624004039)/[Project](https://github.com/RSIIPAC/LuojiaSET-OSFCR)
25 | 2024 |ICL-Net | JSTARS |ICL-Net: Inverse cognitive learning network for remote sensing image dehazing| [Paper](https://ieeexplore.ieee.org/document/10665990)/[Project]
26 | 2024 |C2AIR | WACV |C2AIR: Consolidated Compact Aerial Image Haze Removal| [Paper](https://ieeexplore.ieee.org/document/10484204)/[Project](https://github.com/AshutoshKulkarni4998/C2AIR)
27 | 2024 |AU-Net | TGRS |Dehazing Network: Asymmetric Unet Based on Physical Model| [Paper](https://ieeexplore.ieee.org/document/10415396)/[Project](https://github.com/Dudujia160918/Dehazing_Network_AU-Net)
28 | 2025 |BMFH-Net | TCSVT |Bidirectional-Modulation Frequency-Heterogeneous Network for Remote Sensing Image Dehazing| [Paper](https://ieeexplore.ieee.org/document/11006655)/[Project](https://github.com/zqf2024/BMFH-Net)
29 | 2025 |HPN-CR | TGRS | HPN-CR: Heterogeneous Parallel Network for SAR-Optical Data Fusion Cloud Removal| [Paper](https://ieeexplore.ieee.org/document/10906642)/[Project](https://github.com/G-pz/HPN-CR)
30 | 2025 |DDIA-CFR | Information Fusion | Breaking through clouds: A hierarchical fusion network empowered by dual-domain cross-modality interactive attention for cloud-free image reconstruction| [Paper](https://www.sciencedirect.com/science/article/pii/S1566253524004275)/[Project]
31 | 2025 |SMDCNet | ISPRS P&RS |Cloud removal with optical and SAR imagery via multimodal similarity attention| [Paper](https://www.sciencedirect.com/science/article/pii/S0924271625001856)/[Project]
32 | 2025 |MIMJT | ECCV |Satellite Image Dehazing Via Masked Image Modeling and Jigsaw Transformation| [Paper](https://link.springer.com/chapter/10.1007/978-3-031-91838-4_27)/[Project]
33 | 2025 |MCAF-Net | TGRS |Real-World Remote Sensing Image Dehazing: Benchmark and Baseline| [Paper](https://ieeexplore.ieee.org/abstract/document/11058953)/[Project](https://github.com/lwCVer/RRSHID)
34 | 2025 |DFDNet | JSTARS |Density-Guided and Frequency Modulation Dehazing Network for Remote Sensing Images| [Paper](https://ieeexplore.ieee.org/abstract/document/10946836)/[Project]
35 | 2025 |CCHD | GRSL |CCHD: Chain Connection and Hybrid Dense Attention for Remote Sensing Dehazing | [Paper](https://ieeexplore.ieee.org/abstract/document/11003077)/[Project]




------


# 3. Adversarial Generation for Remote Sensing Image Dehazing:  <a id="GANmodels" class="anchor" href="#GANmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  

:rocket::rocket::rocket:Update (in 2025-07-26) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2018 | Cloud-GAN | IGARSS | Cloud-gan: Cloud removal for sentinel-2 imagery using a cyclic consistent generative adversarial networks | [Paper](https://ieeexplore.ieee.org/document/8519033)/[Project]
02 | 2020 | CR-GAN-PM | ISPRS P&RS | Thin cloud removal in optical remote sensing images based on generative adversarial networks and physical model of cloud distortion | [Paper](https://www.sciencedirect.com/science/article/pii/S0924271620301787)/[Project](https://github.com/Neooolee/CR-GAN-PM)
03 | 2020 |UCR | TGRS |Single image cloud removal using U-Net and generative adversarial networks| [Paper](https://ieeexplore.ieee.org/document/9224941)/[Project]
04 | 2020 | SpA-GAN| arXiv | Cloud Removal for Remote Sensing Imagery via Spatial Attention Generative Adversarial Network | [Paper](https://arxiv.org/abs/2009.13015)/[Project](https://github.com/Penn000/SpA-GAN_for_cloud_removal)
05 | 2020 | FCTF-Net | GRSL | A coarse-to-fine two-stage attentive network for haze removal of remote sensing images | [Paper](https://ieeexplore.ieee.org/document/9136742)/[Project](https://github.com/Neooolee/FCTF-Net?tab=readme-ov-file)
06 | 2020 | SScGAN | WACV | Single Satellite Optical Imagery Dehazing using SAR Image Prior Based on conditional Generative Adversarial Networks | [Paper](https://ieeexplore.ieee.org/document/9093471)/[Project]
07 | 2021 | SAR2Opt-GAN-CR | TGRS | Cloud removal in remote sensing images using generative adversarial networks and SAR-to-optical image translation | [Paper](https://ieeexplore.ieee.org/abstract/document/9627647)/[Project]
08 | 2021 | SkyGAN | WACV | Domain-Aware Unsupervised Hyperspectral Reconstruction for Aerial Image Dehazing | [Paper](https://ieeexplore.ieee.org/document/9423159)/[Project]
09 | 2022 |Dehaze-AGGAN| TGRS | Dehaze-AGGAN: Unpaired remote sensing image dehazing using enhanced attention-guide generative adversarial networks | [Paper](https://ieeexplore.ieee.org/document/9881213)/[Project]
10 | 2023 | MSDA-CR | GRSL | Cloud removal in optical remote sensing imagery using multiscale distortion-aware networks | [Paper](https://ieeexplore.ieee.org/document/9686746)/[Project]
11 | 2024 | TC-BC | ISPRS P&RS | A thin cloud blind correction method coupling a physical model with unsupervised deep learning for remote sensing imagery | [Paper](https://www.sciencedirect.com/science/article/pii/S0924271624003460)/[Project](https://github.com/Liying-Xu/TCBC)
12 | 2025 | MT_GAN | ISPRS P&RS | MT_GAN: A SAR-to-optical image translation method for cloud removal | [Paper](https://www.sciencedirect.com/science/article/pii/S0924271625001479)/[Project](https://github.com/NUAA-RS/MT_GAN)
13 | 2025 | UTCR-Dehaze | EAAI | UTCR-Dehaze: U-Net and transformer-based cycle-consistent generative adversarial network for unpaired remote sensing image dehazing | [Paper](https://www.sciencedirect.com/science/article/pii/S0952197625013879)/[Project]
14 | 2025 | Dehazing-DiffGAN | TGRS | Dehazing-DiffGAN: Sequential Fusion of Diffusion Models and GANs for High-Fidelity Remote Sensing Image Dehazing | [Paper](https://ieeexplore.ieee.org/document/11142885)/[Project](https://github.com/Fan-pixel/Dehazing-DiffGAN)


------


# 4. Vision Transformer for Remote Sensing Image Dehazing:  <a id="Transformer" class="anchor" href="#Transformer" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2025-07-26) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2022 | TransRA |Multidimensional Systems and Signal Processing | TransRA: Transformer and residual attention fusion for single remote sensing image dehazing | [Paper](https://link.springer.com/article/10.1007/s11045-022-00835-x)/[Project] 
02 | 2023 | DehazeFormer |TIP | Vision transformers for single image dehazing | [Paper](https://ieeexplore.ieee.org/abstract/document/10076399)/[Project](https://github.com/IDKiro/DehazeFormer)
03 | 2023 | FormerCR |Remote Sensing | Former-CR: A transformer-based thick cloud removal method with optical and SAR imagery | [Paper](https://www.mdpi.com/2072-4292/15/5/1196)/[Project]
04 | 2023 | RSDformer | GRSL | Learning an Effective Transformer for Remote Sensing Satellite Image Dehazing |[Paper](https://ieeexplore.ieee.org/document/10265239)/[Project](https://github.com/MingTian99/RSDformer)
05 | 2023 | Trinity-Net |TGRS | Trinity-Net: Gradient-guided Swin transformer-based remote sensing image dehazing and beyond | [Paper](https://ieeexplore.ieee.org/abstract/document/10149032)/[Project](https://github.com/chi-kaichen/Trinity-Net)
06 | 2023 | AIDTransformer | WACV | Aerial Image Dehazing with Attentive Deformable Transformers |[Paper](https://ieeexplore.ieee.org/document/10030985)/[Project](https://github.com/AshutoshKulkarni4998/AIDTransformer)
07 | 2024 | DCR-GLFT |TGRS | Density-aware Cloud Removal of Remote Sensing Imagery Using a Global-Local Fusion Transformer | [Paper](https://ieeexplore.ieee.org/document/10713444)/[Project]
08 | 2024 |SSGT | JSTARS | SSGT: Spatio-Spectral Guided Transformer for Hyperspectral Image Fusion Joint with Cloud Removal | [Paper](https://ieeexplore.ieee.org/document/10706710)/[Project]
09 | 2024 |PGSformer| GRSL | Prompt-Guided Sparse Transformer for Remote Sensing Image Dehazing | [Paper](https://ieeexplore.ieee.org/document/10648722)/[Project]
10 | 2024 | ASTA | GRSL | Additional Self-Attention Transformer With Adapter for Thick Haze Removal |[Paper](https://ieeexplore.ieee.org/document/10443626)/[Project](https://github.com/Eric3200C/ASTA)
11 | 2024 | Dehaze-TGGAN | TGRS | Dehaze-TGGAN: Transformer-Guide Generative Adversarial Networks With Spatial-Spectrum Attention for Unpaired Remote Sensing Dehazing | [Paper](https://ieeexplore.ieee.org/document/10614150)/[Project]
12 | 2024 | PCSformer | TGRS | Proxy and Cross-Stripes Integration Transformer for Remote Sensing Image Dehazing | [Paper](https://ieeexplore.ieee.org/document/10677537)/[Project](https://github.com/SmileShaun/PCSformer)
13 | 2025 | DehazeXL | CVPR | Tokenize Image Patches: Global Context Fusion for Effective Haze Removal in Large Images |[Paper](https://arxiv.org/abs/2504.09621)/[Project](https://github.com/CastleChen339/DehazeXL)
14 | 2025 | DecloudFormer | Pattern Recognition | DecloudFormer: Quest the key to consistent thin cloud removal of wide-swath multi-spectral images |[Paper](https://www.sciencedirect.com/science/article/pii/S0031320325003243)/[Project](https://github.com/N1rv4n4/DecloudFormer)
15 | 2025 | CINet | TGRS | Cross-Level Interaction and Intralevel Fusion Network for Remote Sensing Image Dehazing |[Paper](https://ieeexplore.ieee.org/abstract/document/11026022)/[Project]
16 | 2025 | MABDT | Signal Processing | MABDT: Multi-scale attention boosted deformable transformer for remote sensing image dehazing |[Paper](https://www.sciencedirect.com/science/article/pii/S0165168424003888)/[Project](https://github.com/ningjin00/MABDT)
17 | 2025 | CLEAR-Net | JSTARS | CLEAR-Net: A Cascaded Local and External Attention Network for Enhanced Dehazing of Remote Sensing Images |[Paper](https://ieeexplore.ieee.org/document/10855474)/[Project]


------


#  5. Diffusion Generation for Remote Sensing Image Dehazing:  <a id="Diffusion" class="anchor" href="#Diffusion" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2025-07-26) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2023 | ARDD-Net | GRSL | Remote Sensing Image Dehazing Using Adaptive Region-Based Diffusion Models | [Paper](https://ieeexplore.ieee.org/document/10233893)/[Project]
02 | 2023 |SeqDMs | Remote Sensing | Cloud removal in remote sensing using sequential-based diffusion models | [Paper](https://www.mdpi.com/2072-4292/15/11/2861)/[Project]
03 | 2024 |ADND-Net | GRSL | Diffusion Models Based Null-Space Learning for Remote Sensing Image Dehazing | [Paper](https://ieeexplore.ieee.org/document/10445700)/[Project] 
04 | 2024 |RSHazeDiff | T-ITS | RSHazeDiff: A unified Fourier-aware diffusion model for remote sensing image dehazing | [Paper](https://ieeexplore.ieee.org/document/10747754)/[Project](https://github.com/jm-xiong/RSHazeDiff)
05 | 2024 |IDF-CR | TGRS | IDF-CR: Iterative diffusion process for divide-and-conquer cloud removal in remote-sensing images | [Paper](https://ieeexplore.ieee.org/abstract/document/10474382)/[Project](https://github.com/SongYxing/IDF-CR)
06 | 2025 |EMRDM | CVPR | Effective Cloud Removal for Remote Sensing Images by an Improved Mean-Reverting Denoising Model with Elucidated Design Space | [Paper](https://arxiv.org/abs/2503.23717)/[Project](https://github.com/Ly403/EMRDM)
07 | 2025 |DFG-DDM | TGRS | DFG-DDM: Deep Frequency-Guided Denoising Diffusion Model for Remote Sensing Image Dehazing | [Paper](https://ieeexplore.ieee.org/abstract/document/11071917)/[Project](https://github.com/Junjie-LLL/DFG-DDM)
08 | 2025 |DS-RDMPD | TGRS | A Dual-Stage Residual Diffusion Model with Perceptual Decoding for Remote Sensing Image Dehazing | [Paper](https://ieeexplore.ieee.org/abstract/document/11130517)/[Project](https://github.com/Aaronwangz/DS-RDMPD)



------

# 6. :surfer: Current Challenges and Future Prospects  <a id="prospects" class="anchor" href="#citation" aria-hidden="true"><span class="octicon octicon-link"></span></a> 
![avatar](/prospects.png)
**Fig 2.** The outlines of current challenges and future prospects in RS image restoration. 
    We reorganize the representative challenges into three overarching directions: 
Dynamic-Aware Restoration, Multi-modal Generalization, and Efficiency-Oriented Design.


------


# :bar_chart: Evaluation:  <a id="evaluation" class="anchor" href="#evaluation" aria-hidden="true"><span class="octicon octicon-link"></span></a>  

 For evaluation on **Dehazed** results, modify 'test_original' and 'test_restored' to the corresponding path
```python
python evaluate.py --train_folder [restored image path] --target_folder [ground-truth image path]
```

Make sure the file structure is consistent with the following:
```python
dataset
├── ASTA
│   ├── RRSHID-M
│   ├── RRSHID-TK
│   ├── RRSHID-TN
│   ├── SH-M
│   ├── SH-TK
│   └── SH-TN
│       └── 1.png, 2.png, ...
│
├── DCIL
│   ├── RRSHID-M
│   ├── RRSHID-TK
│   ├── RRSHID-TN
│   ├── SH-M
│   ├── SH-TK
│   └── SH-TN
│       └── 1.png, 2.png, ...
│
├── DehazeFormer
│   ├── RRSHID-M
│   ├── RRSHID-TK
│   ├── RRSHID-TN
│   ├── SH-M
│   ├── SH-TK
│   └── SH-TN
│       └── 1.png, 2.png, ...
│
├── DehazeXL
│   ├── RRSHID-M
│   ├── RRSHID-TK
│   ├── RRSHID-TN
│   ├── SH-M
│   ├── SH-TK
│   └── SH-TN
│       └── 1.png, 2.png, ...
│
├── LFD-Net
│   ├── RRSHID-M
│   ├── RRSHID-TK
│   ├── RRSHID-TN
│   ├── SH-M
│   ├── SH-TK
│   └── SH-TN
│       └── 1.png, 2.png, ...
│
├── PSMB
│   ├── RRSHID-M
│   ├── RRSHID-TK
│   ├── RRSHID-TN
│   ├── SH-M
│   ├── SH-TK
│   └── SH-TN
│       └── 1.png, 2.png, ...
│
├── SpA-GAN
│   ├── RRSHID-M
│   ├── RRSHID-TK
│   ├── RRSHID-TN
│   ├── SH-M
│   ├── SH-TK
│   └── SH-TN
│       └── 1.png, 2.png, ...
│
└── Trinity-Net
|   ├── RRSHID-M
|   ├── RRSHID-TK
|   ├── RRSHID-TN
|   ├── SH-M
|   ├── SH-TK
|   └── SH-TN
│       └── 1.png, 2.png, ...
|
├── RRSHID-M-GT
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── RRSHID-TK-GT
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── RRSHID-TN-GT
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── SH-M-GT
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── SH-TK-GT
│   ├── 1.png
│   ├── 2.png
│   └── ...
└── SH-TN-GT
    ├── 1.png
    ├── 2.png
    └── ...
```
**Table 1.** 
Quantitative performance at PSNR (dB) and SSIM of remote sensing image restoration algorithms evaluated on the SateHaze1k (SH-TN, SH-M, SH-TK) and RICE datasets.

| Methods      | Category    | SH-TN       |            | SH-M        |            | SH-TK       |            | RICE        |            |
|--------------|-------------|-------------|------------|-------------|------------|-------------|------------|-------------|------------|
|              |             | PSNR        | SSIM       | PSNR        | SSIM       | PSNR        | SSIM       | PSNR        | SSIM       |
| SMIDCP       | Traditional | 13.639      | 0.833      | 15.990      | 0.863      | 14.956      | 0.757      | 16.573      | 0.712      |
| EVPM         | Traditional | 20.426      | 0.891      | 20.656      | 0.918      | 16.647      | 0.787      | 15.217      | 0.742      |
| IeRs         | Traditional | 15.048      | 0.772      | 14.763      | 0.785      | 11.754      | 0.702      | 15.750      | 0.611      |
| GRS-HTM      | Traditional | 15.489      | 0.762      | 15.071      | 0.784      | 10.473      | 0.462      | 18.278      | 0.825      |
| SRD          | Traditional | 21.327      | 0.896  | 20.774      | 0.930      | 17.265      | 0.814      | 20.550      | 0.896  |
| DHIM         | Traditional | 19.445      | 0.891      | 19.916      | 0.917      | 16.595      | 0.810      | 19.240      | 0.882      |
| EMPF-Net     | CNN         | **27.400**  | _0.960_      | **31.450**  | 0.975      | 26.330      | 0.928      | 35.845      | **0.979**  |
| SFAN         | CNN         | 23.688      | **0.963**  | _28.191_  | **0.977**  | 23.006      | **0.942**  | 35.374      | 0.941      |
| ICL-Net      | CNN         | 24.590      | 0.923      | 25.670      | 0.937      | 21.780      | 0.859      | **36.940**  | _0.960_      |
| FCTF-Net     | CNN         | 23.590      | 0.913      | 22.880      | 0.927      | 20.030      | 0.816      | 25.535      | 0.870      |
| PSMB-Net     | CNN         | 22.946      | 0.949      | 27.921      | 0.960      | 21.273      | 0.919      | 28.057      | 0.893      |
| DCIL         | CNN         | 20.187      | 0.947      | 27.431      | 0.964      | 21.450      | 0.926      | 27.720      | 0.876      |
| EDED-Net     | CNN         | 24.605      | 0.893      | 25.360      | 0.913      | 22.418      | 0.846      | 31.907      | 0.945      |
| TransRA      | Transformer | 25.200      | 0.930      | 26.500      | 0.947      | 22.730      | 0.875      | 31.130      | 0.955      |
| PGSformer    | Transformer | 25.534  | 0.918      | 26.622      | 0.933      | 23.596      | 0.863      | 34.404      | 0.948      |
| Trinity-Net  | Transformer | 21.304      | 0.946      | 26.473      | _0.963_  | 20.756      | 0.915      | 29.248      | 0.908      |
| RSDformer    | Transformer | 24.210      | 0.912      | 26.241      | 0.934      | 23.011      | 0.853      | 33.013      | 0.953      |
| ARDD-Net     | Diffusion   | 26.840      | 0.926      | 26.470      | 0.932      | _26.830_  | 0.932  | -           | -          |
| ADND-Net     | Diffusion   | _26.910_  | 0.927      | 26.670      | 0.936      | **26.940**  | _0.936_  | -           | -          |
| RSHazeDiff   | Diffusion   | -           | -          | -           | -          | -           | -          | _36.560_  | 0.958      |

---------------------

Go to [Reproducibility Check list](ReproducibilityChecklist.md)


---------------------

# :books: Citation: <a id="citation" class="anchor" href="#citation" aria-hidden="true"><span class="octicon octicon-link"></span></a> 

- If you find [our survey paper] and evaluation code are useful, please cite the following paper:
```BibTeX
@article{zhou2025remote,
  title={Remote Sensing Image Dehazing: A Systematic Review of Progress, Challenges, and Prospects},
  author={Zhou, Heng and Liu, Xiaoxiong and Zhang, Zhenxi and Li, Chengyang and Yun, Jieheng and Yang, Yunchu and Tian, Chunna and Wu, Xiao-Jun},
  journal={arXiv preprint arXiv:},
  pages={1--56},
  year={2025},
}
```

[![Star History Chart](https://api.star-history.com/svg?repos=VisionVerse/RemoteSensing-Restoration-Survey&type=Date)](https://www.star-history.com/#VisionVerse/RemoteSensing-Restoration-Survey)


--------------------------------------------------------------------------------------

# :clap::clap::clap: Thanks to the above authors for their excellent work！
