import os 
import cv2
import numpy as np
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rainy_data_path", type=str, default="./dataset/Rain100L/test/input/", help='Path to rainy data')
parser.add_argument("--ldgp_data_path", type=str, default="./dataset/Rain100L/test/ldgp/", help='Path to ldgp data')
parser.add_argument("--sdr_result_path", type=str, default="./dataset/Rain100L/test/sdr/", help='Path to save sdr data')
parser.add_argument("--kernel_size", type=int, default=7, help='K')
parser.add_argument("--num", type=int, default=50, help='The numer of sdr for each images')
opt = parser.parse_args()

ks= opt.kernel_size
padding_num = ks//2
random_num = opt.num
input_path = opt.rainy_data_path
mask_path = opt.ldgp_data_path
result_path = opt.sdr_result_path

try:
    os.mkdir(result_path)
except:
    pass

input_folder = os.listdir(input_path)
ldgp_folder = os.listdir(mask_path)

for idx in range(len(input_folder)):    
    print("Stochastic Derained References: ", input_folder[idx])
    
    # Create Folder to place sdr img
    dir_path = os.path.join(result_path,input_folder[idx][:-4])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Read Image
    img = cv2.imread(os.path.join(input_path, input_folder[idx]))
    mask = cv2.imread(os.path.join(mask_path, ldgp_folder[idx]), cv2.IMREAD_GRAYSCALE)
    mask = cv2.bitwise_not(mask)
    h = mask.shape[0]
    w = mask.shape[1]
    
    # Create images w/ padding
    images = []
    for n in range(random_num):
        images.append( cv2.copyMakeBorder(copy.deepcopy(img),padding_num,padding_num,padding_num,padding_num, cv2.BORDER_CONSTANT, value=0))
    mask = cv2.copyMakeBorder(mask, padding_num,padding_num,padding_num,padding_num, cv2.BORDER_CONSTANT, value=0)
    img = cv2.copyMakeBorder(img, padding_num,padding_num,padding_num,padding_num, cv2.BORDER_CONSTANT, value=0)
    
    # Start to replace
    for j in range(padding_num, mask.shape[0]-padding_num+1):
        for i in range(padding_num, mask.shape[1]-padding_num+1):
            if mask[j,i] < 255:
                c_m = mask[j-padding_num:j+padding_num+1,i-padding_num:i+padding_num+1]
                neighbor = []
                for c_m_j in range(c_m.shape[0]):
                    for c_m_i in range(c_m.shape[1]):
                        if c_m[c_m_j,c_m_i]==255:
                            neighbor.append(ks*c_m_j+c_m_i)
                try:sample = np.random.choice(neighbor, random_num)
                except:break
                
                for n in range(random_num):
                    pix = sample[n]
                    images[n][j,i,:] = img[j+(pix//ks-padding_num),i+(pix%ks-padding_num),:]
                
    for l in range(random_num):
        cv2.imwrite(os.path.join(dir_path,input_folder[idx][:-4]+"-"+str(l)+'.png'),images[l][padding_num:padding_num+h, padding_num:padding_num+w])            
    
    
    
