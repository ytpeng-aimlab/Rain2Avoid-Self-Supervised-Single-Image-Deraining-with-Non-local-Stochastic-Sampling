import torch
import numpy as np
from torchvision.transforms import functional as F




def all_rain(grid):
   return np.all(grid>0)

def get_mask_and_size(rainy_image, ldgp_image, j, i, small_window_size=3, center=True):
   
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
   big_window_size = 7 # default K
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
            probability.append(1/(max(np.sum(np.abs(center_rain_vector - neighbor_rain_vector))**(1), 1e-4)))
            # probability.append(1/(max(1, 1e-4)))

   # normalize probability   
   probability = np.array(probability).astype(np.float32)
   probability = probability/ probability.sum()

   return neighbor, probability


ldgp_intensity_threshold = 10

def generate_sdr(noise_images, mask, device, sdr_num):
   
   # print(noise_images)
   # tensor > numpy array
   rainy_image = np.clip(noise_images[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)*255
   ldgp_image = np.clip(mask[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)*255

   # print(rainy_image.shape)
   # print(ldgp_image.shape)

   

   # print("Start Generate SDR")

   # rainy_image    = cv2.imread(os.path.join(input_path, image_name)) 
   # ldgp_image     = cv2.imread(os.path.join(ldgp_path, image_name), cv2.IMREAD_GRAYSCALE)
   return_images  = np.zeros((sdr_num, rainy_image.shape[0], rainy_image.shape[1], rainy_image.shape[2]))
   return_images[:,:,:,:] = rainy_image[:,:,:]

   Height, Width = rainy_image.shape[0], rainy_image.shape[1]

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
            
   # exit()
   print("Finish Generate SDR")
   # print(return_images.shape)
   # numpy array > tensor
   r = list()

   for num in range(sdr_num):
      each_tensor = F.to_tensor(return_images[num]).float().to(device).unsqueeze(0)
      each_tensor = each_tensor/255
      # print(each_tensor.shape)
      # print(each_tensor)
      # each_tensor = each_tensor.permute(1,2,0).to(device)
      
      r.append(each_tensor)

   return r