import os
import argparse
from train_all import train_all

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./dataset/Rain100L/", help='path to input data')
parser.add_argument("--save_path", type=str, default="./result/Rain100L/", help='path to save result')

parser.add_argument("--seed", type=int, default=10, help='random seed')
parser.add_argument("--psd_num", type=int, default=50, help='the num of pseudo gt')
parser.add_argument("--mode", type=str, default="train_all", choices=['train_single', 'train_all', 'train_single_pyramid'],help='training mode')

# train_single parameter
parser.add_argument("--single_epoch", type=int, default=1, help='training epoch')
parser.add_argument("--f1", type=int, default=1, help='training epoch')
parser.add_argument("--f2", type=int, default=1, help='training epoch')

# train_all parameter
parser.add_argument("--epoch", type=int, default=2000, help='training epoch')
parser.add_argument("--batch_size", type=int, default=64, help='training batch size')
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lr_steps', type=list, default=[(x+1) * 100 for x in range(1000//100)])
parser.add_argument('--model_save_dir', type=str, default="./result_all/Rain100L/")

opt = parser.parse_args()

if opt.mode=='train_all':
    try:
        os.makedirs(opt.model_save_dir)
    except:
        pass


if __name__ == "__main__":
    
    train_all(opt)