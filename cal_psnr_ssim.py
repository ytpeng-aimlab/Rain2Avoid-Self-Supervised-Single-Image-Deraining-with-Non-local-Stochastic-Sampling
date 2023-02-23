from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
import cv2
import argparse

def avg(score):
    return sum(score)/len(score)

parser = argparse.ArgumentParser()
parser.add_argument("--rainy_data_path", type=str, default="./dataset/Rain100L/test/input/", help='Path to rainy data')
parser.add_argument("--gt_data_path", type=str, default="./dataset/Rain100L/test/target/", help='Path to sdr data')
parser.add_argument("--result_path", type=str, default="./result/Rain100L/test/", help='Path to save result')
opt = parser.parse_args()


if __name__ == "__main__":

    input_path  = opt.rainy_data_path
    gt_path     = opt.gt_dat_path    
    derain_path = opt.result_path
    gt_folder   = os.listdir(gt_path)
    
    before_psnr_list = []
    before_ssim_list = []
    after_psnr_list = []
    after_ssim_list = []

    for i in range(len(gt_folder)):
        gt      = cv2.imread(gt_path + gt_folder[i])
        input   = cv2.imread(input_path + gt_folder[i])
        derain  = cv2.imread(derain_path + gt_folder[i])
        before_psnr = compare_psnr(gt,input)
        before_ssim = ssim(gt,input, multichannel=True)
        after_psnr = compare_psnr(gt,derain)
        after_ssim = ssim(gt,derain, multichannel=True)
        before_psnr_list.append(before_psnr)
        before_ssim_list.append(before_ssim)
        after_psnr_list.append(after_psnr)
        after_ssim_list.append(after_ssim)
        
        print(before_psnr, before_ssim, after_psnr, after_ssim)

    print("Before Avgrage PSNR:", avg(before_psnr_list))
    print("Before Avgrage SSIM:", avg(before_ssim_list))
    print("After Average PSNR:", avg(after_psnr_list))
    print("After Average SSIM", avg(after_ssim_list))



