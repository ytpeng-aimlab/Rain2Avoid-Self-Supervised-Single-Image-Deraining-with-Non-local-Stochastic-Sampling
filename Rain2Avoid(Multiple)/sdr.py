import cv2
import os
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input_path", type=str, default="./dataset/Rain12/input/", help='input path')
parser.add_argument("--ldgp_path", type=str, default="./dataset/Rain12/ldgp/", help='input path')
parser.add_argument("--save_path", type=str, default="./dataset/Rain12/sdr/" , help='sdr save path')
parser.add_argument("--fuse_save_path", type=str, default="./dataset/Rain12/sdr(fuse)/" , help='fuse sdr save path')
parser.add_argument("--pow", type=int, default=0.5, help='eta in nss')
parser.add_argument("--threshold", type=int, default=10, help='the threshold of ldgp')
parser.add_argument("--K", type=int, default=7, help="neighbor size")
parser.add_argument("--k", type=int, default=3, help="similarity size")
parser.add_argument("--sdr_num", type=int, default=50, help='the number of sdr')

opt = parser.parse_args()

input_path = opt.input_path
ldgp_path = opt.ldgp_path
target_path = opt.save_path
target_path2 = opt.fuse_save_path
sdr_num = opt.sdr_num
ldgp_intensity_threshold = opt.threshold

def make_folder(path):
   try:
      os.makedirs(path)
   except:
      pass

def all_rain(grid):
   return np.all(grid>0)

def get_mask_and_size(rainy_image, ldgp_image, j, i, small_window_size=opt.k, center=True):
   
   h, w = ldgp_image.shape[0], ldgp_image.shape[1]
   ldgp_image[ldgp_image>0] = 1

   # determine the small window size 
   while(1):
      small_padding_size = small_window_size//2
      mask = ldgp_image[max(j-small_padding_size,0):min(j+small_padding_size+1,h), max(i-small_padding_size,0):min(i+small_padding_size+1,w)]
      if all_rain(mask)==False:
         break
      else:
         small_window_size+=2
   
   # initial return mask and rain with the small window size
   return_mask = np.zeros([small_window_size, small_window_size])
   return_rain = np.zeros([small_window_size, small_window_size, 3])
   
   for local_j in range(small_window_size):
      for local_i in range(small_window_size):
         global_j = local_j-small_padding_size+j
         global_i = local_i-small_padding_size+i
         
         # boundary check
         if global_j>h-1 or global_j<0 or global_i>w-1 or global_i<0:
            pass         
         
         elif center:
            return_mask[local_j][local_i] = ldgp_image[global_j, global_i]
            return_rain[local_j, local_i,:] = rainy_image[global_j,global_i,:]

         else:
            return_rain[local_j, local_i,:] = rainy_image[global_j,global_i,:]

   return return_rain, 1-return_mask, small_window_size

def compute_similarity(rainy_image, ldgp_image, j, i):

   # initial
   neighbor, probability = list(), list()
   Height, Width = rainy_image.shape[0], rainy_image.shape[1]
   
   # compute center valid pixel and convert to vector
   center_rain_grid, mask_grid, size = get_mask_and_size(rainy_image, ldgp_image, j, i)
   center_rain_grid = np.transpose(center_rain_grid,(2, 0, 1)) # H,W,C > C, H, W
   center_rain_grid = (center_rain_grid*mask_grid)
   center_rain_grid = np.transpose(center_rain_grid,(1, 2, 0))  # C,H,W > H,W,C
   center_rain_vector = center_rain_grid.flatten()
   
   # determin big window size (K)
   big_window_size = opt.K # default K
   big_window_size = max(big_window_size, 3)
   big_padding_size = big_window_size//2
   
   # compute neighbor grid
   for n_j in range(max(j-big_padding_size,0), min(j+big_padding_size+1, Height)):
      for n_i in range(max(i-big_padding_size,0), min(i+big_padding_size+1, Width)):
         if ldgp_image[n_j, n_i] == 0: # if is non-rain > candidate
            
            neighbor.append((n_j, n_i))
            # compute neighbor valid pixel and convert to vector
            
            neighbor_rain_grid, _, _ = get_mask_and_size(rainy_image, ldgp_image, n_j, n_i, small_window_size=size, center=False)
            
            neighbor_rain_grid = np.transpose(neighbor_rain_grid,(2, 0, 1)) # H,W,C > C, H, W
            neighbor_rain_grid = (neighbor_rain_grid*mask_grid)
            neighbor_rain_grid = np.transpose(neighbor_rain_grid,(1, 2, 0))  # C,H,W > H,W,C
            neighbor_rain_vector = (neighbor_rain_grid).flatten()
            
            # convert simialrity to probability (l1 loss)
            probability.append(1/(max(np.sum(np.abs(center_rain_vector - neighbor_rain_vector))**(opt.pow), 1e-4)))
            # probability.append(1/(max(1, 1e-4)))

   # normalize probability   
   probability = np.array(probability).astype(np.float32)
   probability = probability/ probability.sum()

   return neighbor, probability

make_folder(target_path2)

rainy_folder = os.listdir(input_path)                       # the input folder
already_folder = os.listdir(target_path2)                   # the result folder
rainy_folder = list(set(rainy_folder)-set(already_folder))  # input - result 

total_time = 0

## SDR
for image_name in rainy_folder:
   make_folder(os.path.join(target_path, image_name[:-4]))
   
   rainy_image    = cv2.imread(os.path.join(input_path, image_name)) 
   ldgp_image     = cv2.imread(os.path.join(ldgp_path, image_name), cv2.IMREAD_GRAYSCALE)
   return_images  = np.zeros((sdr_num, rainy_image.shape[0], rainy_image.shape[1], rainy_image.shape[2]))
   return_images[:,:,:,:] = rainy_image[:,:,:]
   
   Height, Width = rainy_image.shape[0], rainy_image.shape[1]
   
   start = time.process_time()
   
   for j in range(Height):
      for i in range(Width):
         if ldgp_image[j,i] > ldgp_intensity_threshold:
            neighbor, probability = compute_similarity(rainy_image, ldgp_image.copy(), j, i)
            try:
               np.random.seed(0)
               sample = np.random.choice(len(neighbor), sdr_num, p = probability)
               for num in range(sdr_num):
                  pix = neighbor[sample[num]]
                  return_images[num,j,i,:] = rainy_image[pix[0],pix[1],:]
            except:
               pass

   stop = time.process_time()    
   
   # save sdr in folder
   for i in range(sdr_num):
      cv2.imwrite(os.path.join(target_path, image_name[:-4], str(i)+".png"), return_images[i])

   # save the fusion image of sdrs
   fuse = np.average(return_images, axis=0, keepdims=False)
   cv2.imwrite(os.path.join(target_path2, image_name), fuse)
   

   # calculte time
   duration = stop - start
   print("SDR: ", image_name, '/Time: ',duration ) 
   total_time += duration

# average time
print("Average Time: ", total_time/len(rainy_folder))