import torch
from torch.optim import Adam
from torch.nn import MSELoss
import torchvision.transforms as T

import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from network import UNet
from data import test_all_dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../dataset/Rain100L/", help='path to input data')
parser.add_argument("--model_path", type=str, default="./result_all/Rain100L/model.pkl", help='path to model weight')
parser.add_argument("--img_path", type=str, default="./result_all/Rain100L/image/", help='path to save result data')

opt = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
if __name__ == "__main__":
   try:
      os.makedirs(opt.img_path)
   except:
      pass
   
   data_loader = test_all_dataloader(opt.data_path, batch_size=1)
   state_dict = torch.load(opt.model_path)
   model = UNet().to(device)
   model.load_state_dict(state_dict['model'])
   
   for batch in tqdm(data_loader):
      rainy_images, clean_images, ldgp_mask, sdr_images, name = batch
      rainy_images = rainy_images.to(device)
      derained_image = model(rainy_images).detach()
      
      derained_image = np.clip(derained_image[0].permute(1, 2, 0).detach().cpu().numpy(), 0, 1)
      plt.imsave(os.path.join(opt.img_path, name[0]), derained_image)