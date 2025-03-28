# 单个网络训练

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from tqdm import tqdm
from utils.dataloader1 import dataloader


from pytorch3d.utils import ico_sphere
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
from EPcode.SARNNSim import SARNNM # as SARNNMEP13

def set_seed(seed=2021): # seed的数值可以随意设置，本人不清楚有没有推荐数值
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
    else:
        device = torch.device("cpu")

    # Set paths
    DATA_DIR = ".\\data"
    obj_filename = os.path.join(DATA_DIR, "ico_sphereP.obj")
    realimgpath = r"D:\Project\Data\MSTAR\SLICYCSVuni\30_DEG"

    # ==============================================
    # 载入数据集
    batch_size = 64
    img_size = 138 #SLICY 54*2 T72 138
    train_loader, val_loader = dataloader(realimgpath, img_size, batch_size, device)

    # ==============================================
    distance = 15  # distance from camera to the object
    SlantPixSpacing = 0.2 / 2
    imagesize = img_size * 2
    simsetting = {'distance': distance ,'SlantPixSpacing': SlantPixSpacing, 'imagesize': imagesize}

    # ==============================================
    # ==============================================

    # 建立模型
    EPNum = str(1)
    model1 = SARNNM(obj_filename=obj_filename, simsetting = simsetting ,savepath=r'.\logs\Result\EP'+EPNum, device=device)
    writer1 = SummaryWriter(r'.\logs\runs\EP'+EPNum)

    EPNum = str(2)
    model2 = SARNNM(obj_filename=obj_filename, simsetting = simsetting ,savepath=r'.\logs\Result\EP'+EPNum, device=device)
    writer2 = SummaryWriter(r'.\logs\runs\EP'+EPNum)

    paramsets = [  # {'params': model1.netPix.parameters()},
        {'params': model1.shininess},
        # {'params': model1.ambient_color},
        {'params': model1.diffuse_color},
        {'params': model1.specular_color},
        {'params': model1.deform_verts},
        # {'params':model1.verts_rgb}
    ]
    model1.optimizer = torch.optim.Adam(paramsets)

    paramsets = [  # {'params': model2.netPix.parameters()},
        {'params': model2.shininess},
        # {'params': model2.ambient_color},
        {'params': model2.diffuse_color},
        {'params': model2.specular_color},
        # {'params': model2.deform_verts},
        {'params':model2.verts_rgb}
    ]
    model2.optimizer = torch.optim.Adam(paramsets)


    # 载入训练好的模型
    # model.mloadmodel()
    # model.mloadmodel(r'.\logs\Result\EP3\models')

    num_epoch = 1000
    val_loss_old1 = 1
    val_loss_old2 = 1

    loop = tqdm(range(model1.epoch ,num_epoch))
    for epoch in loop:


        # **************************************************************
        # training-----------------------------------
        train_loss1 = 0
        train_loss2 = 0
        for trainindex, sample in enumerate(train_loader):  ####
            train_loss1 += model1.mtrain(sample, epoch + 1)
            train_loss2 += model2.mtrain(sample, epoch + 1)
            # break #######
        writer1.add_scalar('train loss', train_loss1 / len(train_loader), global_step=epoch + 1)
        writer2.add_scalar('train loss', train_loss2 / len(train_loader), global_step=epoch + 1)

        # evaluation--------------------------------
        val_loss1 = 0
        val_loss2 = 0
        for valindex, sample in enumerate(val_loader):
            val_loss1 += model1.mtest(sample, epoch + 1)
            val_loss2 += model2.mtest(sample, epoch + 1)
            # break ########
        writer1.add_scalar('val loss', val_loss1 / len(val_loader), global_step=epoch + 1)
        writer2.add_scalar('val loss', val_loss2 / len(val_loader), global_step=epoch + 1)

        if (epoch > 10) and (val_loss_old1 > (val_loss1 / len(val_loader))):
            model1.msave(epoch + 1)
            val_loss_old1 = val_loss1 / len(val_loader)
        if (epoch > 10) and (val_loss_old2 > (val_loss2 / len(val_loader))):
            model2.msave(epoch + 1)
            val_loss_old2 = val_loss2 / len(val_loader)

        if (epoch %10 ) == 0:
            model1.meshvisual(epoch = epoch)
            model2.meshvisual(epoch = epoch)


        desc = f"val_loss: {val_loss1 / len(val_loader)}"
        loop.set_description(desc)

        # break #######

    model1.meshsave(epoch=epoch)
    model2.meshsave(epoch=epoch)

    writer1.close()
    writer2.close()
    print('训练已完成！')