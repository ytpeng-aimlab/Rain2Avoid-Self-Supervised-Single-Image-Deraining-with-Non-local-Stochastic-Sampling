import torch
from torch.optim import Adam
from torch.nn import MSELoss
import torchvision.transforms as T

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from network import UNet
from data import train_all_dataloader
from torch.utils.tensorboard import SummaryWriter

loss_function = MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_all(opt):
   writer = SummaryWriter()
   random.seed(opt.seed)
   data_loader = train_all_dataloader(opt.data_path, batch_size=opt.batch_size)
   model = UNet().to(device)
   optimizer = Adam(model.parameters(), lr=0.001)
   scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_steps, opt.gamma)

   for epoch in tqdm(range(opt.epoch)):
      for itx, batch in enumerate(data_loader):
         rainy_images, clean_images, ldgp_mask, sdr_images, name = batch
         rainy_images = rainy_images.to(device)
         clean_images = clean_images.to(device)
         ldgp_mask    = ldgp_mask.to(device)
         sdr_images   = sdr_images.to(device)
         _, _, H, W = rainy_images.shape
         
         more_rainy_images = rainy_images

         net_output = model(more_rainy_images)
         loss = loss_function(net_output, sdr_images)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
   
      # if epoch%10==0:
      #    print("Epoch: ", epoch, "Epoch Loss: ", loss.item())
   
   model_name = os.path.join(opt.model_save_dir, 'model.pkl')
   torch.save({'model': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict()}, model_name)