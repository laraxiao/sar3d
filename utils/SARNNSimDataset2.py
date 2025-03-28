import os
from os import listdir
import random
from PIL import Image
import torch
import torch.nn.parallel
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


# from skimage import io, transform
# from ImgRegi import imgregist



# 几何信息图像网格大小
sim_size = 138 #SLICY 54*2 T72 138
real_size = 138

imgsize = 54 #SLICY 54*2 T72 138
# ==========================================================
# 由于真实数据过大，内存和显存无法一次性全部读取，
# 所以只存储标签名，在每个epoch中再读取数据
# 以下的数据读取 以及通过网络后的回波图像转换 只能在每个epoch里顺序进行

# 回波网络的计算 是网络本身的一部分

# 输入：真值的文件夹目录 仿真数据的文件夹目录
class SARimgelevazimDataset(Dataset):

    def _getfilenames(self, ):
        filenames = listdir(self.realpath)
        filenum = len(filenames)

        fileazims = np.empty([filenum])
        for ii in range(filenum):
            fileazims[ii] = float(filenames[ii].split('-')[2])
        fileindex = np.argsort(fileazims)
        filenamessort = [None] * filenum
        for ii in range(filenum):
            filenamessort[ii] = filenames[fileindex[ii]]
        # filelist = []
        # for ii in range(filenum):
        #     filelist.append( [ filenames[ii],  float(filenames[ii].split('-')[2]),float(filenames[ii].split('-')[3]) ] )
        # filelist.sort(key = lambda x : (x[1], x[2]))

        return filenamessort
    #----------------------------------------------------------------------------------


    #----------------------------------------------------------------------------------
    def __init__(self, realpath, imgsize, device, transform=None):

        self.realpath = realpath
        self.realfilenames = self._getfilenames()
        self.imgsize = imgsize
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.realfilenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        realfilename = self.realfilenames[idx]

        # ---------------------------------------------------------------------------------------
        realimgpath = os.path.join(self.realpath,  realfilename)
        # realimg = pd.read_csv(realimgpath, header=None, index_col=False)
        # realimg = realimg.fillna(0).values
        # realimg = realimg[np.newaxis, 0:self.imgsize, 0:self.imgsize] # 对真实图像进行裁切
        # realimg = torch.as_tensor(realimg, dtype=torch.float32, device=self.device).view(1, self.imgsize, self.imgsize)
        realimg = 0

        azim = float(realimgpath.split('-')[2])
        azim = torch.tensor(azim, dtype=torch.float32, device=self.device)#.view(1)
        elev = float(realimgpath.split('-')[3])
        elev = torch.tensor(elev, dtype=torch.float32, device=self.device)#.view(1)
        sample = {'samplename': realfilename ,'realimg': realimg, 'elev': elev, 'azim': azim}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample


