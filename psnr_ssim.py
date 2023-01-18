from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
import cv2


def avg(score):
    return sum(score)/len(score)


if __name__ == "__main__":

    input_path  = './dataset/Rain100L/test/rainy/'
    gt_path     = './dataset/Rain100L/test/gt/'    
    derain_path = './dataset/Rain100L/test/rainy/'
    
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



