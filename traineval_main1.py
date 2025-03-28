# 单个网络训练

import os
from os import listdir
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from tqdm import tqdm
from utils.dataloader1 import dataloader

import warnings
warnings.filterwarnings("ignore")

# from EPcode.SARNNSimEP1 import SARNNM as SARNNMEP1
# from EPcode.SARNNSimEP2 import SARNNM as SARNNMEP2
# from EPcode.SARNNSimEP3 import SARNNM as SARNNMEP3
# from EPcode.SARNNSimEP4 import SARNNM as SARNNMEP4
# from EPcode.SARNNSimEP5 import SARNNM as SARNNMEP5
# from EPcode.SARNNSimEP6 import SARNNM as SARNNMEP6
# from EPcode.SARNNSimEP7 import SARNNM as SARNNMEP7
# from EPcode.SARNNSimEP8 import SARNNM as SARNNMEP8
# from EPcode.SARNNSimEP9 import SARNNM as SARNNMEP9
# from EPcode.SARNNSimEP10 import SARNNM as SARNNMEP10
# from EPcode.SARNNSimEP11 import SARNNM as SARNNMEP11
# from EPcode.SARNNSimEP12 import SARNNM as SARNNMEP12
from EPcode.SARNNSim2 import SARNNM # as SARNNMEP13
# from EPcode.SARNNSim2 import SARNNM # as SARNNMEP13

def set_seed(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #cuDNN在使用deterministic模式时（下面两行），可能会造成性能下降（取决于model）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # ==============================================
    ## 随机种子初始化
    set_seed()

    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        # Print out CUDA device information
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(device)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA device index: {torch.cuda.current_device()}")
        print(f"Total memory allocated on CUDA device {torch.cuda.current_device()}: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Total memory cached on CUDA device {torch.cuda.current_device()}: {torch.cuda.memory_cached() / 1024**3:.2f} GB")


    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")


    # Set paths
    DATA_DIR = "./data"
    # obj_filename = os.path.join(DATA_DIR, "SLICYP2.obj")
    # obj_filename = os.path.join(DATA_DIR, "T72WP.obj")
    obj_filename = os.path.join(DATA_DIR, "T72W.obj")
    if os.path.exists(obj_filename):
     print(f"File found: {obj_filename}")
    else:
     print(f"File not found: {obj_filename}")

    # realpath1 = r"D:/Project/Data/MSTAR/SLICYCSVuni/30_DEG"
    # realpath2 = r"D:/Project/Data/MSTAR/SLICYCSVuni/15_DEG"
    # realpath1 =  r"/mnt/d/Project/Data/MSTAR/SLICYSim/CSV/30_DEG"
    # realpath2 =  r"/mnt/d/Project/Data/MSTAR/SLICYSim/CSV/15_DEG"
    # realpath1 =  r"/mnt/d/Project/Data/MSTAR/SLICYCSV/30_DEG"
    # realpath2 =  r"/mnt/d/Project/Data/MSTAR/SLICYCSV/15_DEG"
    # realpath1 =  r"/mnt/d/Project/Data/MSTAR/T72WSim/CSV/15A05"
    # realpath2 =  r"/mnt/d/Project/Data/MSTAR/T72WSim/CSV/17A05"

    # original path
    # realpath1 =  r"/mnt/d/Project/Data/MSTAR/T72WSim/CSV32"
    # realpath2 =  r"/mnt/d/Project/Data/MSTAR/T72WSim/CSV32"

    realpath1 =  r"/scratch/xiaojingying/csv_T72_azim_elev"
    realpath2 =  r"/scratch/xiaojingying/csv_T72_azim_elev"


    savepath=r'./logs/Resultnew/EP'
    # ==============================================
    # 载入数据集
    batch_size = 8
    img_size = 124 #SLICY 54*2 T72 138
    train_loader, val_loader = dataloader(realpath1, realpath2, img_size, batch_size, device)


    # ==============================================
    distance = 90  # distance from camera to the object
    SlantPixSpacing = 0.2
    scalefactor = 4 # 4 T72W 1/276.5*9.53
    imagesize = img_size
    simsetting = {'distance': distance ,'SlantPixSpacing': SlantPixSpacing, 'imagesize': imagesize, 
                            'scalefactor': scalefactor}

    # ==============================================
    # ==============================================

    # 建立模型
    EPNum = 39
    model1 = SARNNM(obj_filename=obj_filename, simsetting = simsetting, EPNum = EPNum, savepath=savepath, device=device)

    EPNum = 0
    model2 = SARNNM(obj_filename=obj_filename, simsetting = simsetting, EPNum = EPNum, savepath=savepath, device=device)


    paramsets1 = [  # {'params': model1.netPix.parameters()},
        {'params': model1.shininess},
        # {'params': model1.ambient_color},
        {'params': model1.diffuse_color},
        {'params': model1.specular_color},
        {'params': model1.deform_verts},
        # {'params':model1.verts_rgb}
    ]
    model1.optimizer = torch.optim.Adam(params = paramsets1, lr=0.01)
    # model1.optimizer = torch.optim.SGD(params = paramsets1, lr=1.0, momentum=0.9)


    paramsets2 = [  # {'params': model2.netPix.parameters()},
        {'params': model2.shininess},
        # {'params': model2.ambient_color},
        {'params': model2.diffuse_color},
        {'params': model2.specular_color},
        # {'params': model2.deform_verts},
        {'params':model2.verts_rgb}
    ]
    model2.optimizer = torch.optim.Adam(params = paramsets2, lr=0.01)
    

    # 载入训练好的模型
    model1.mloadmodel()
    model2.mloadmodel()
    # model.mloadmodel(r'./logs/Result/EP3/models')

    num_epoch = 200
    val_loss_old1 = 1
    val_loss_old2 = 1

    loop = tqdm(range(model1.epoch ,num_epoch))
    for epoch in loop:


        # **************************************************************
        # training-----------------------------------
        train_loss1 = 0
        train_loss2 = 0
        for trainindex, sample in enumerate(train_loader):  #### each time load 8 samples ,while batch size is 8, do the loading 3 times, that's why trainindex = 2
            train_loss1 += model1.mtrain(sample, epoch + 1)
            # train_loss2 += model2.mtrain(sample, epoch + 1)
            # break #######
        model1.writer.add_scalar('train loss', train_loss1 / (trainindex+1.0), global_step=epoch + 1)
        model2.writer.add_scalar('train loss', train_loss2 / (trainindex+1.0), global_step=epoch + 1)

        
        with torch.no_grad():
        # evaluation--------------------------------
            val_loss1 = 0
            val_loss2 = 0
            for valindex, sample in enumerate(val_loader):
                val_loss1 += model1.mtest(sample, epoch + 1)
                # val_loss2 += model2.mtest(sample, epoch + 1)
                break ########
            model1.writer.add_scalar('val loss', val_loss1 / (valindex+1.0), global_step=epoch + 1)
            #model2.writer.add_scalar('val loss', val_loss2 / (valindex+1.0), global_step=epoch + 1)
        
        

        if (epoch > 10) and (val_loss_old1 > (train_loss1 / (trainindex+1.0))):
            model1.msave(epoch + 1)   # overwrite the one before and get the final and best model, save to models/SARNNM.pth
            val_loss_old1 = train_loss1 / (trainindex+1.0)
        # if (epoch > 10) and (val_loss_old2 > (val_loss2 / len(val_loader))):
        #     model2.msave(epoch + 1)
        #     val_loss_old2 = val_loss2 / len(val_loader)

        if (epoch %10 ) == 0:
            model1.meshvisual(epoch = epoch)
            # model2.meshvisual(epoch = epoch)
        
        '''
        if epoch == 40:
                paramsets = [   {'params': model1.netPix.parameters()},
                    {'params': model1.shininess},
                    # {'params': model1.ambient_color},
                    {'params': model1.diffuse_color},
                    {'params': model1.specular_color},
                    {'params': model1.deform_verts},
                    # {'params':model1.verts_rgb}
                ]
                model1.optimizer = torch.optim.Adam(paramsets)

                paramsets = [   {'params': model2.netPix.parameters()},
                    {'params': model2.shininess},
                    # {'params': model2.ambient_color},
                    {'params': model2.diffuse_color},
                    {'params': model2.specular_color},
                    # {'params': model2.deform_verts},
                    {'params':model2.verts_rgb}
                ]
                model2.optimizer = torch.optim.Adam(paramsets)
        '''

        # desc = f"val_loss: {val_loss1 /  (valindex+1.0)}"
        desc = f"val_loss: {train_loss1 /  (trainindex+1.0)}"
        loop.set_description(desc)

        # break #######

    model1.meshsave(epoch=epoch)
    model2.meshsave(epoch=epoch)

    model1.writer.close()
    model2.writer.close()
    print('训练已完成！')