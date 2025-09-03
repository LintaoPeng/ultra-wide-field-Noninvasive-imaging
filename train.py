import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion
from torchvision.utils import save_image
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.utils import *
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt



dtype = 'float32'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.set_default_tensor_type(torch.FloatTensor)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Wavelet-Based Diffusion Model')
    parser.add_argument("--config", default='speckle.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--image_folder", default='results/', type=str,
                        help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=230, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    #print("=> using dataset '{}'".format(config.data.train_dataset))
    #DATASET = datasets.__dict__[config.data.type](config)
    training_x = []
    path = './data/train/speckle/'  # 路径
    path_list = os.listdir(path)
    path_list.sort(key=lambda x: int(x.split('.')[0]))  # 根据文件名排序

    for item in path_list:
        impath = path + item
        #print("开始处理" + impath)
        img = sio.loadmat(impath)["data"]
        imgx = (img - img.min()) / (img.max() - img.min())  # 归一化

        # 切割为28x28的块，并整合成32个通道
        blocks = []
        for i in range(0, 256, 32):
            for j in range(0, 256, 32):
                block = imgx[i:i + 32, j:j + 32]
                if block.shape == (32, 32):  # 确保是完整的28x28块
                    blocks.append(block)
                if len(blocks) == 64:  # 每张图片生成32个28x28块后停止
                    break # 跳出内层循环
            if len(blocks) == 64:
                training_x.append(blocks)
                break  # 跳出外层循环

                

    # 转换为PyTorch张量
    X_train = np.array(training_x)  # 转换为numpy数组，形状为(N, 32, 28, 28)
    X_train = X_train.astype(np.float32)  # 设置数据类型
    X_train = torch.from_numpy(X_train)  # 转换为PyTorch张量
    #X_train = X_train.unsqueeze(1) # 归一化到[0, 1]

    print("input shape:", X_train.shape)  # 输出张量形状，应该为(N, 32, 28, 28)

    training_y=[]
    path='./data/train/GT/'#要改
    path_list = os.listdir(path)
    path_list.sort(key=lambda x:int(x.split('.')[0]))
    for item in path_list:
        impath=path+item
        #print("开始处理"+impath)
        img= cv2.imread(path+item,0)
        imgx=(img-img.min())/(img.max()-img.min())
        training_y.append(imgx)


    y_train = []
    for features in training_y:
        y_train.append(features)

    y_train = np.array(y_train)
    y_train=y_train.astype(dtype)
    y_train= torch.from_numpy(y_train)
    y_train=y_train.unsqueeze(1)
    y_train=y_train
    print("output shape:",y_train.shape)


    test_x=[]
    path='./data/test/speckle/'#要改
    path_list = os.listdir(path)
    path_list.sort(key=lambda x:int(x.split('.')[0]))
    for item in path_list:
        impath=path+item
        #print("开始处理"+impath)
        img=sio.loadmat(impath)["data"]
        imgx=(img-img.min())/(img.max()-img.min())

        # 切割为28x28的块，并整合成32个通道
        blocks = []
        for i in range(0, 256, 32):
            for j in range(0, 256, 32):
                block = imgx[i:i + 32, j:j + 32]
                if block.shape == (32, 32):  # 确保是完整的28x28块
                    blocks.append(block)
                if len(blocks) == 64:  # 每张图片生成32个28x28块后停止
                    break # 跳出内层循环
            if len(blocks) == 64:
                test_x.append(blocks)
                break  # 跳出外层循环



    x_test = np.array(test_x)
    x_test=x_test.astype(dtype)
    x_test= torch.from_numpy(x_test)
    #x_test=x_test.unsqueeze(1)
    #x_test=x_test.unsqueeze(1)
    print("test input shape:",x_test.shape)


    test_Y=[]
    path='./data/test/GT/'#要改
    path_list = os.listdir(path)
    path_list.sort(key=lambda x:int(x.split('.')[0]))
    for item in path_list:
        impath=path+item
        #print("开始处理"+impath)
        img= cv2.imread(path+item,0)
        imgx=(img-img.min())/(img.max()-img.min())
        test_Y.append(imgx)


    Y_test = []
    for features in test_Y:
        Y_test.append(features)

    Y_test = np.array(Y_test)
    Y_test=Y_test.astype(dtype)
    Y_test= torch.from_numpy(Y_test)
    Y_test=Y_test.unsqueeze(1)
    Y_test=Y_test
    print("test output shape:",Y_test.shape)

    
    train_loader = DataLoader(TensorDataset(X_train,y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, Y_test), batch_size=16)



    # create model
    print("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args, config)
    diffusion.train(train_loader,test_loader)


if __name__ == "__main__":
    main()
