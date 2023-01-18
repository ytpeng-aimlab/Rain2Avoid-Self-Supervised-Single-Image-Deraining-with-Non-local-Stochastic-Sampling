# Rain2Avoid Self Supervised Single Image Deraining
Pytorch Implementation of "Rain2Avoid: Self-Supervised Single Image Deraining"

## Introduction
The single image deraining task aims to remove rain from
a single image, attracting much attention in the field. Recent research on this topic primarily focuses on discriminative deep learning methods, which train models on rainy images with their clean counterparts. However, collecting such paired images for training takes much work. Thus, we present Rain2Avoid (R2A), a training scheme that requires only rainy images for image deraining. We propose a locally dominant gradient prior to reveal possible rain streaks and overlook those rain pixels while training with the input rainy image directly. Understandably, R2A may not perform as well as deraining methods that supervise their models with rain-free ground truth. However, R2A favors when training image pairs are unavailable and can self-supervise only one rainy image for deraining. Experimental results show that the proposed method performs favorably against state-of-the-art few-shot deraining and self-supervised denoising methods.

## Framework

## Installation
```
pip install -r requirements.txt
```

## Command
- Locally Dominant Gradient Prior (LDGP)
```
python LDGP.py 
```
- Generate Pseudo Ground Truth
```
python pgt.py
```
- Self-Supervised
```
python train_unet.py
```

## Deraining Results

![alt text](https://github.com/ytpeng-aimlab/Rain2Avoid-Self-Supervised-Single-Image-Deraining/blob/master/img/Figure1.png?raw=true)


## Citation