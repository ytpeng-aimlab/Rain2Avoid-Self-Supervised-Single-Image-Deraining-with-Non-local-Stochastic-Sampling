import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

from PIL import Image as Image
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm

from utils import Timer
from network import UNet, ResNet, DnCNN
from restormer import Restormer
from data import SDR_dataloader, train_dataloader


torch.manual_seed(3)
parser = argparse.ArgumentParser()
dataset = 'DDN_SIRR_real_pad'
parser.add_argument("--rainy_data_path", type=str, default="./dataset/"+dataset+"/", help='Path to rainy data')
parser.add_argument("--sdr_data_path", type=str, default="./dataset/"+dataset+"/sdr_except/sdr_7_nlm_3/", help='Path to sdr data')
parser.add_argument("--result_path", type=str, default="./result/except(old)/result_7_NLM_3/"+dataset+"/test/", help='Path to save result')
parser.add_argument("--backbone", type=str, default="Unet", help= "select backbone to be used in SDRL")
parser.add_argument("--epoch", type=int, default=100)
opt = parser.parse_args()

data_path = opt.rainy_data_path
save_path = opt.result_path
sdr_path = opt.sdr_data_path
epochs = opt.epoch

loss_function = MSELoss()
data_loader = train_dataloader(data_path, batch_size=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    os.makedirs(save_path)
except:
    pass

epoch_timer = Timer('s') 
total_time = 0

for batch in data_loader:
    try:
        # train 
        rainy_images, clean_images, name = batch
        epoch_timer.tic()    
        
        if opt.backbone == "Unet":
            model = UNet()
        elif opt.backbone == "ResNet":
            model = ResNet()
        elif opt.backbone == "DnCNN":
            model = DnCNN()
        elif opt.backbone == "Restormer":
            model = Restormer()

        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=0.001)

        inner_batch_size = 1
        rainy_images = rainy_images.to(device)
        clean_images = clean_images.to(device)
        model.train()
        SDR_loader = SDR_dataloader(os.path.join(sdr_path, name[0][:-4]), batch_size=inner_batch_size)
        

        for j in tqdm(range(epochs)):
            for k, inner_batch in enumerate(SDR_loader):
                sdr_images = inner_batch
                sdr_images = sdr_images.to(device)
                images = torch.cat([rainy_images for _ in range(len(sdr_images))],0)
                net_output = model(images)
                loss = loss_function(net_output, sdr_images)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # inference

        model.eval()
        net_output = model(rainy_images)
        time = epoch_timer.toc()
        print("Time: ", time)
        total_time += time
        denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        plt.imsave(os.path.join(save_path,name[0]), denoised)
    
    except Exception as e:
        print("Exception occur: ", e)
        pass
    
print("Finish! Average Time:", total_time/len(data_loader))