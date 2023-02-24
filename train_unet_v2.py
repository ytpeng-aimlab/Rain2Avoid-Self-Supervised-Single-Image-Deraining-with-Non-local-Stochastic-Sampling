from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from PIL import Image as Image
import torch
from torch.nn import MSELoss
from torch.optim import Adam
import torch
from torch.utils.tensorboard import SummaryWriter
from models.unet import UNet
from data_v2 import train_dataloader
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import Timer, Adder
import argparse
from generateSDR_multiple import generate_sdr



parser = argparse.ArgumentParser()
parser.add_argument("--rainy_data_path", type=str, default="./dataset/Rain100L/test/", help='Path to rainy data')
parser.add_argument("--sdr_data_path", type=str, default="./dataset/Rain100L/test/sdr/", help='Path to sdr data')
parser.add_argument("--result_path", type=str, default="./result/Rain100L/test/", help='Path to save result')
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--gpu_id", type=str, default="9")
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
torch.manual_seed(3)
data_path = opt.rainy_data_path
save_path = opt.result_path
epochs = opt.epoch
batch_size = opt.batch_size

loss_function = MSELoss()
data_loader = train_dataloader(data_path, batch_size=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    os.makedirs(save_path)
except:
    pass

epoch_timer = Timer('s') 
for idx, batch in enumerate(data_loader):
    rainy_images, clean_images, ldgp_images, name = batch
    print("Processing:", name[0])
    
    epoch_timer.tic()
    model = UNet()
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    
    inner_batch_size = 1
    rainy_images = rainy_images.to(device)
    clean_images = clean_images.to(device)
    model.train()
    
    sdr_images = generate_sdr(rainy_images, ldgp_images, device, batch_size)
    sdr_images = sdr_images.to(device)
    # images = rainy_images.expand(batch_size, -1, -1, -1)                            
    
    for epoch in tqdm(range(epochs)):    
        net_output = model(rainy_images)
        loss = loss_function(net_output, sdr_images[epoch % batch_size])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    net_output = model(rainy_images)
    print("Time: ", epoch_timer.toc())
    denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
    plt.imsave(os.path.join(save_path,name[0]), denoised)
    print("-"*10)
print("Finish!")