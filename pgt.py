import os 
import cv2
import numpy as np
import copy


ks= 7
padding_num = ks//2
random_num = 50

dataset = 'Rain100L/test'
input_path = './dataset/'+dataset+'/rainy/'
mask_path = './LDGP/'+dataset+'/mask/'
result_path = './pseudo_gt/'+dataset+'/pgt-'+str(ks)+'/'

try:
    os.mkdir(result_path)
except:
    pass

folder1 = os.listdir(input_path)
folder2 = os.listdir(mask_path)

for idx in range(len(folder1)):    
    print("pseudo ground truth: ", folder1[idx])
    # Create Folder to place filled img
    dir_path = os.path.join(result_path,folder1[idx][:-4])
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Read Image
    img = cv2.imread(os.path.join(input_path, folder1[idx]))
    mask = cv2.imread(os.path.join(mask_path, folder2[idx]), cv2.IMREAD_GRAYSCALE)
    mask = cv2.bitwise_not(mask)
    h = mask.shape[0]
    w = mask.shape[1]
    
    # create images w/ padding
    images = []
    for n in range(random_num):
        images.append( cv2.copyMakeBorder(copy.deepcopy(img),padding_num,padding_num,padding_num,padding_num, cv2.BORDER_CONSTANT, value=0))
    mask = cv2.copyMakeBorder(mask, padding_num,padding_num,padding_num,padding_num, cv2.BORDER_CONSTANT, value=0)
    img = cv2.copyMakeBorder(img, padding_num,padding_num,padding_num,padding_num, cv2.BORDER_CONSTANT, value=0)
    
    # start to fill
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
        cv2.imwrite(os.path.join(dir_path,folder1[idx][:-4]+"-"+str(l)+'.png'),images[l][padding_num:padding_num+h, padding_num:padding_num+w])            
    
    
    
