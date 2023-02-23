import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from PIL import Image as Image
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import Timer, Adder
from models.unet import UNet
from data import SDR_dataloader, train_dataloader

torch.manual_seed(3)
parser = argparse.ArgumentParser()
parser.add_argument("--rainy_data_path", type=str, default="./dataset/Rain100L/test/", help='Path to rainy data')
parser.add_argument("--sdr_data_path", type=str, default="./dataset/Rain100L/test/sdr/", help='Path to sdr data')
parser.add_argument("--result_path", type=str, default="./result/Rain100L/test/", help='Path to save result')
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

for batch in tqdm(data_loader):
    rainy_images, clean_images, name = batch
    print("Processing:", name[0][:-4])
    
    epoch_timer.tic()
    model = UNet()
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    
    inner_batch_size = 1
    rainy_images = rainy_images.to(device)
    clean_images = clean_images.to(device)
    model.train()
    SDR_loader = SDR_dataloader(os.path.join(sdr_path, name[0][:-4]),batch_size=inner_batch_size)
    
    for j in range(epochs):
        for k, inner_batch in enumerate(SDR_loader):
            sdr_images = inner_batch
            sdr_images = sdr_images.to(device)
            images = torch.cat([rainy_images for _ in range(len(sdr_images))],0)
            net_output = model(images)
            loss = loss_function(net_output, sdr_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    net_output = model(rainy_images)
    print("Time: ", epoch_timer.toc())
    denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
    plt.imsave(os.path.join(save_path,name[0]), denoised)
    
print("Finish!")