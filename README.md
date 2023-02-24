# Rain2Avoid Self Supervised Single Image Deraining
Pytorch Implementation of "Rain2Avoid: Self-Supervised Single Image Deraining"

## Introduction
The single image deraining task aims to remove rain from
a single image, attracting much attention in the field. Recent research on this topic primarily focuses on discriminative deep learning methods, which train models on rainy images with their clean counterparts. However, collecting such paired images for training takes much work. Thus, we present Rain2Avoid (R2A), a training scheme that requires only rainy images for image deraining. We propose a locally dominant gradient prior to reveal possible rain streaks and overlook those rain pixels while training with the input rainy image directly. Understandably, R2A may not perform as well as deraining methods that supervise their models with rain-free ground truth. However, R2A favors when training image pairs are unavailable and can self-supervise only one rainy image for deraining. Experimental results show that the proposed method performs favorably against state-of-the-art few-shot deraining and self-supervised denoising methods.

## Framework

![image](https://github.com/ytpeng-aimlab/Rain2Avoid-Self-Supervised-Single-Image-Deraining/blob/master/img/framework.png)

## Installation
```
pip install -r requirements.txt
```

## Dataset
* **Rain100L**: 200 training pairs and 100 test pairs *[[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)][[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)](2017 CVPR)*

* **Rain1400(DDN-Data)**: 12600 training pairs and 1400 test pairs *[[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf)][[dataset](https://xueyangfu.github.io/projects/cvpr2017.html)] (2017 CVPR)*

- Please prepare your datsets follow the struture as bellow.
```
./dataset/Rain100L
+--- test
|   +--- rainy
|   +--- gt
```

## Command
### Method 1: (Generate stochastic derained references off-line)
- Locally Dominant Gradient Prior (LDGP)
```
python generate_ldgp.py --rainy_data_path "./dataset/Rain100L/test/input/" --ldgp_result_path "./dataset/Rain100L/test/ldgp/"
```
- Generate Stochastic Derained References
```
python generate_sdr.py --rainy_data_path "./dataset/Rain100L/test/input/" --ldgp_data_path "./dataset/Rain100L/test/ldgp/" --sdr_result_path "./dataset/Rain100L/test/sdr/"
```
- Self-Supervised
```
python train_unet.py --rainy_data_path "./dataset/Rain100L/test/" --sdr_data_path "./dataset/Rain100L/test/sdr/" --result_path "./result/Rain100L/test/"
```
### Method 2: (Generate stochastic derained references while training (on-line))
- Locally Dominant Gradient Prior (LDGP)
```
python generate_ldgp.py --rainy_data_path "./dataset/Rain100L/test/input/" --ldgp_result_path "./dataset/Rain100L/test/ldgp/"
```
- Self-Supervised
```
python train_unet_v2.py --rainy_data_path "./dataset/Rain100L/test/" --sdr_data_path "./dataset/Rain100L/test/sdr/" --result_path "./result/Rain100L/test/"
```

### Evaluation
```
python cal_psnr_ssim.py
```

## LDGP

<details>
<summary><strong>LDGP on Rain100L&DDN-SIRR</strong> (click to expand) </summary>
<img src = "https://github.com/ytpeng-aimlab/Rain2Avoid-Self-Supervised-Single-Image-Deraining/blob/master/img/img2.png"> 
</details>


## Deraining Results
<details>
<summary><strong>Qualitative Result on Rain100L</strong> (click to expand) </summary>
<img src = "https://github.com/ytpeng-aimlab/Rain2Avoid-Self-Supervised-Single-Image-Deraining/blob/master/img/img1.png"> 
</details>

<details>
<summary><strong>Quantitative Result on Rain100L</strong> (click to expand) </summary>
<img src = "https://github.com/ytpeng-aimlab/Rain2Avoid-Self-Supervised-Single-Image-Deraining/blob/master/img/result_rain100L.png"> 
</details>

<details>
<summary><strong>Quantitative Result on Rain800, DDN-SIRR</strong> (click to expand) </summary>
<img src = "https://github.com/ytpeng-aimlab/Rain2Avoid-Self-Supervised-Single-Image-Deraining/blob/master/img/result_DDN.png"> 
</details>

## Citation
