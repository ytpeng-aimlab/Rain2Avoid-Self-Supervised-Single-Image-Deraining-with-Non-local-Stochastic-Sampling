from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
import cv2

class im():
    def __init__(self,name,score,score2,score3,score4,score5):
        self.name = name
        self.score = score
        self.score2 = score2
        self.score3 = score3
        self.score4 = score4
        self.score5 = score5

def avg(score):
    return sum(score)/len(score)


if __name__ == "__main__":
    gt_path = './dataset/train/gt/'
    input_path = './dataset/train/input/'
    
    # derain_path = './out(convolution,square_sum,ks=3)/'     # red
    # derain_path2 = './out(convolution,square_sum,ks=5)/'    # green
    # derain_path3 = './out(convolution,square_sum,ks=7)/'    # purple
    
    derain_path = './out(convolution,square_sum,ks=3)/'       # red
    derain_path2 = './out(convolution,square_sum,ks=5)/'      # green
    derain_path3 = './out(convolution,square_sum,ks=7)/'      # purple
    derain_path4 = './out(convolution,penta,ks=3,new_Hog)/'      # yellow

    gt_folder = os.listdir(gt_path)
    
    ims = []

    for i in range(len(gt_folder)):
        gt=cv2.imread(gt_path+gt_folder[i])
        input =cv2.imread(input_path+gt_folder[i])
        
        derain = cv2.imread(derain_path+gt_folder[i])
        derain2 = cv2.imread(derain_path2+gt_folder[i])
        derain3 = cv2.imread(derain_path3+gt_folder[i])
        derain4 = cv2.imread(derain_path4+gt_folder[i])

        b_psnr = compare_psnr(gt,input)
        b_sim = ssim(gt,input, multichannel=True)
        
        a_psnr = compare_psnr(gt,derain)
        a_sim = ssim(gt,derain, multichannel=True)

        a_psnr2 = compare_psnr(gt,derain2)
        a_sim2 = ssim(gt,derain2, multichannel=True)
        
        a_psnr3 = compare_psnr(gt,derain3)
        a_sim3 = ssim(gt,derain3, multichannel=True)
        
        a_psnr4 = compare_psnr(gt,derain4)
        a_sim4 = ssim(gt,derain4, multichannel=True)

        im_ob = im(gt_folder[i],b_psnr,a_psnr,a_psnr2,a_psnr3, a_psnr4)

        ims.append(im_ob)
        # print(gt_folder[i],"Before:",b_psnr, b_sim," After:", a_psnr, a_sim)

    print("done")
    scores = []
    scores2 = []
    scores3 = []
    scores4 = []
    scores5 = []
    
    newlist = sorted(ims, key=lambda x: x.score, reverse=False)
    for i in newlist:
        scores.append(i.score)
        scores2.append(i.score2)
        scores3.append(i.score3)
        scores4.append(i.score4)
        scores5.append(i.score5)
    

    print(avg(scores[:25]),avg(scores2[:25]),avg(scores3[:25]),avg(scores4[:25]),avg(scores5[:25]))

    print(avg(scores[25:51]),avg(scores2[25:51]),avg(scores3[25:51]),avg(scores4[25:51]),avg(scores5[25:51]))
        
    print(avg(scores[51:76]),avg(scores2[51:76]),avg(scores3[51:76]),avg(scores4[51:76]),avg(scores5[51:76]))
    
    print(avg(scores[76:]),avg(scores2[76:]),avg(scores3[76:]),avg(scores4[76:]),avg(scores5[76:]))
    

    plt.plot(scores, color='blue', marker='o',mfc='blue' ) 
    # plt.plot(scores2, color='red', marker='o',mfc='red') 
    # plt.plot(scores3, color='green', marker='o',mfc='green') 
    # plt.plot(scores4, color='purple', marker='o',mfc='purple') 
    plt.plot(scores5, color='yellow', marker='o',mfc='yellow') 
    

    plt.ylabel('score') #set the label for y axis
    plt.xlabel('index') #set the label for x-axis
    plt.title("Rain100L") #set the title of the graph
    plt.show() #display the graph


