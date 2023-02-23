from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from PIL import Image as Image
import torch
from torch.nn import MSELoss
from torch.optim import Adam
import torch
from torch.utils.tensorboard import SummaryWriter
from models.unet import UNet
from data import train_dataloader
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import Timer, Adder


torch.manual_seed(3)
dataset = 'Rain100L/train'
experiment = 'pgt-7'
data_path = './dataset/'+dataset+'/'
save_path = os.path.join('./results/', dataset, experiment)
epochs = 100
loss_function = MSELoss()
data_loader = train_dataloader(data_path, batch_size=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    os.makedirs(save_path)
except:
    pass

epoch_timer = Timer('s') 

for batch in tqdm(data_loader):
    epoch_timer.tic()
    model = UNet()
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    noisy_images, clean_images, name = batch
    inner_batch_size = 1
    noisy_images = noisy_images.to(device)
    clean_images = clean_images.to(device)
    model.train()
    pgt_loader = PGT_dataloader(os.path.join(pgt_path, name[0][:-4]),batch_size=inner_batch_size)
    
    for j in range(epochs):
        for k, inner_batch in enumerate(pgt_loader):
            pgt_images = inner_batch
            pgt_images = pgt_images.to(device)
            images = torch.cat([noisy_images for _ in range(len(pgt_images))],0)
            net_output = model(images)
            loss = loss_function(net_output, pgt_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    net_output = model(noisy_images)
    print("Time: ", epoch_timer.toc())
    denoised = np.clip(net_output[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
    plt.imsave(os.path.join(save_path,name[0]), denoised)
    
print("Finish!")