import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def hog(img, show= True, save="", cell_size = (8, 8), num_cells_per_block = (4, 4)): 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    block_size = (num_cells_per_block[0] * cell_size[0],
              num_cells_per_block[1] * cell_size[1])
    x_cells = gray_img.shape[1] // cell_size[0]
    y_cells = gray_img.shape[0] // cell_size[1]
    h_stride = 1
    v_stride = 1
    block_stride = (cell_size[0] * h_stride, cell_size[1] * v_stride)
    num_bins = 36
    win_size = (x_cells * cell_size[0] , y_cells * cell_size[1])
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    hog_descriptor = hog.compute(gray_img)
    
    tot_bx = np.uint32(((x_cells - num_cells_per_block[0]) / h_stride) + 1)
    tot_by = np.uint32(((y_cells - num_cells_per_block[1]) / v_stride) + 1)

    tot_els = (tot_bx) * (tot_by) * num_cells_per_block[0] * num_cells_per_block[1] * num_bins
    
    plt.rcParams['figure.figsize'] = [9.8, 9]

    hog_descriptor_reshaped = hog_descriptor.reshape(tot_bx,
                                                    tot_by,
                                                    num_cells_per_block[0],
                                                    num_cells_per_block[1],
                                                    num_bins).transpose((1, 0, 2, 3, 4))

    ave_grad = np.zeros((y_cells, x_cells, num_bins))
    hist_counter = np.zeros((y_cells, x_cells, 1))

    for i in range (num_cells_per_block[0]):
        for j in range(num_cells_per_block[1]):
            ave_grad[i:tot_by + i, j:tot_bx + j] += hog_descriptor_reshaped[:, :, i, j, :]
            
            hist_counter[i:tot_by + i, j:tot_bx + j] += 1

    ave_grad /= hist_counter
    sumofdegree = [0 for _ in range(num_bins)]
    
    for i in range(num_bins):
        for j in range(ave_grad.shape[0]):
            for k in range(ave_grad.shape[1]):
                sumofdegree[i]+=ave_grad[j][k][i]
    
    
    x = [(180/num_bins)*(i) for i in range(num_bins)]
    
    plt.bar(x,
        sumofdegree, 
        width=180/num_bins, 
        bottom=None, 
        align='center', 
        color=['lightsteelblue', 
            'cornflowerblue', 
            'royalblue', 
            'midnightblue', 
            'navy', 
            'darkblue', 
            'mediumblue'])
    plt.xticks(rotation='vertical')
    if save:
        plt.savefig(save)
    if show:
        plt.show()
   
    plt.cla()
    return sumofdegree.index(max(sumofdegree))


def most_frequent(List):
    return max(set(List), key = List.count)
    
def rotate(img, theta):
    rows, cols = img.shape[0], img.shape[1]
    image_center = (cols/2, rows/2)

    M = cv2.getRotationMatrix2D(image_center,theta,1)

    abs_cos = abs(M[0,0])
    abs_sin = abs(M[0,1])

    bound_w = int(rows * abs_sin + cols * abs_cos)
    bound_h = int(rows * abs_cos + cols * abs_sin)

    M[0, 2] += bound_w/2 - image_center[0]
    M[1, 2] += bound_h/2 - image_center[1]
    rotated = cv2.warpAffine(img,M,(bound_w,bound_h),borderValue=(255,255,255))
    return rotated

if __name__ == '__main__':
    dataset = 'Rain100L/test'
    input_path = './dataset/'+dataset+'/rainy/'
    mask_path = './LDGP/'+dataset+'/mask/'
    
    try:
        os.makedirs(mask_path)
    except:
        pass

    folder = os.listdir(input_path)
    for name in folder:
        print("LDGP:", name)
        image = cv2.imread(input_path+name)
        H = image.shape[0]
        W = image.shape[1]
        patch_size = 80
        H_PatchNum = int(H/patch_size)
        W_PatchNum = int(W/patch_size)
        degrees = []
        iiidx = 0
        for x in range(W_PatchNum):
            for y in range(H_PatchNum):
                image_patch = image[patch_size*y:patch_size*(y+1),patch_size*x:patch_size*(x+1)]
                tmp_degree = 5*hog(image_patch, show=False, save='')
                iiidx+=1
                degrees.append(tmp_degree)
        
        degree = most_frequent(degrees)
        if degree==90:
            degree=0
        if degree>90:
            degree-=180
        tmp = degree
        if (tmp<0):
            tmp = -tmp

        c_W =  H*math.cos(math.radians(tmp))*math.sin(math.radians(tmp))
        c_H =   W*math.sin(math.radians(tmp))*math.cos(math.radians(tmp))
        
        c_W = math.floor(c_W)
        c_H = math.floor(c_H)
        final = rotate(image, degree)
        
        kernel = np.array([[-1, 2, -1]])
        final = cv2.cvtColor(final,cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(final, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)
        
        vertical = np.copy(bw)
        rows = vertical.shape[0]
        verticalsize = rows // 30
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)

        dst = rotate(vertical, -degree)
        dst = dst[c_H:c_H+H,c_W:c_W+W]
        cv2.imwrite(mask_path+name, dst)