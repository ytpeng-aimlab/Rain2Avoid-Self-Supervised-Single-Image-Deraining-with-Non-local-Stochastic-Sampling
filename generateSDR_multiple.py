import torch
import numpy as np

ks= 9 # set kernel size (Randomly selet the pixle in such size of area.)
padding_num = ks//2

def generate_sdr(noise_images, mask, device, sdr_num):
    print("Start Generate SDR")
    return_images = torch.tensor(np.zeros((sdr_num, noise_images.shape[0], noise_images.shape[1], noise_images.shape[2], noise_images.shape[3])))
    return_images = return_images.to(device)
    
    for batch in range(noise_images.shape[0]):
        return_images[:,batch,:,:,:] = noise_images[batch,:,:,:]
        for j in range(padding_num, mask.shape[2]-padding_num):
            for i in range(padding_num, mask.shape[3]-padding_num):
                if mask[batch,0,j,i] > 0.4:
                    c_m = mask[batch, 0, j-padding_num:j+padding_num+1,i-padding_num:i+padding_num+1]
                    neighbor = []
                    for c_m_j in range(c_m.shape[0]):
                        for c_m_i in range(c_m.shape[1]):
                            if c_m[c_m_j,c_m_i] == 0:
                                neighbor.append(ks*c_m_j+c_m_i)
                    try:
                        sample = np.random.choice(neighbor, sdr_num)
                    except:
                        break
                    
                    for num in range(sdr_num):
                        pix = sample[num]
                        return_images[num, batch,:,j,i] = noise_images[batch,:,j+(pix//ks-padding_num),i+(pix%ks-padding_num)]
    print("Finish Generate SDR")      
    return return_images.float()