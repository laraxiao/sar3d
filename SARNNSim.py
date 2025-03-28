import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from pytorch3d.io import load_obj,load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    RasterizationSettings,
    MeshRasterizer,MeshRenderer,
    FoVOrthographicCameras,
    PointLights,TexturesVertex,Materials,
    camera_position_from_spherical_angles,
    DirectionalLights,
)
from pytorch3d.renderer import SoftPhongShader as PhongShader

from utils.plot_image_grid import image_grid
from utils.ImgRegi import imgregist


# utils

def imgplot(img, title=''):
    # silhouette = True if img.shape[-1]==1 else False
    # inds = 0 if silhouette else range(3)
    plt.figure(figsize=(10, 10))
    plt.imshow(img.cpu().detach().numpy())
    plt.title(title)
    plt.grid("off")
    plt.axis("off")





# ==========================================================

class SimPix(nn.Module):
    def __init__(self, device):
        super(SimPix, self).__init__()
        self.device = device
        # 前面三层，完成一个像素内的信号累加的建模，也便于之后直接与真实图像进行比较。
        # 之后的图像用于验证冲击相应等因素的影响。
        # 各个像素之间的影响关系，由网络的后续部分进行建模。这部分可以插入注意力机制
        # input is (nc) x 423 x 423
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # 这一层为SAME, 特征图的大小不变 感受野 3
        # self.Convq = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False) # 调节信号强度的
        # self.Acq = nn.ReLU()

        self.Conv1 = nn.Conv2d(1, 1, kernel_size=107, stride=1, padding=53, bias=False) # 调节卷积核
        self.Ac1 = nn.ReLU()

        self.Conv2 = nn.Conv2d(1, 4, kernel_size=9, stride=1, padding=4, bias=True) # 相干斑噪声
        self.Ac2 = nn.ReLU()

        self.Conv3 = nn.Conv2d(6, 1, kernel_size=5, stride=1, padding=2, bias=False) # 取值组合
        self.Ac3 = nn.Tanh()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 这一层相当于池化层，大小变为原来的一半 H/2 W/2 感受野 4
        # self.Pool1 = nn.Conv2d(npf*2, 1, 2, 2, 0, bias=True)

        # 读取卷积核数据 numpy array
        # imppathfile = 'IMP.csv'
        # Kimp = pd.read_csv(imppathfile, header=None, index_col=False)  # 由于每行末尾有个逗号，通过这种方式去除影响
        # Kimp = Kimp.fillna(0).values  # sim_size sim_size
        # # Kimp = Kimp/np.max(Kimp)
        # self.Kimp = torch.as_tensor(Kimp, dtype=torch.float32, device=device).view(1, 1, Kimp.shape[0], Kimp.shape[1])

        imppathfile = 'Impt108.csv'
        Kimp = pd.read_csv(imppathfile, header=None, index_col=False)  # 由于每行末尾有个逗号，通过这种方式去除影响
        Kimp = Kimp.fillna(0).values  # sim_size sim_size
        Kimpt = torch.as_tensor(Kimp, dtype=torch.float32, device=self.device).view(1, 1, Kimp.shape[0], Kimp.shape[1])

        self.Conv1.weight.data = Kimpt


    def forward(self, input):

        # outputimp = self.Convq(input) # 固定卷积核之前的调整散射点强度
        # outputimp = self.Acq(outputimp)
        # outputimp = F.conv2d(outputimp, self.Kimp, stride=1, padding=53)
        outputimp = self.Conv1(input) # 仅使用网络
        outputimp = self.Ac1(outputimp)

        output = self.Conv2(input)  # 1 64 sim_size sim_size
        output = self.Ac2(output)

        output = torch.cat((input, outputimp, output), dim=1)

        output = self.Conv3(output)
        output = self.Ac3(output)

        output = self.maxpool(output)  # 最佳位置有待讨论
        # output = F.interpolate(output, size=[54, 54], mode='bilinear')



        return output

# ==========================================================


class SARNNM:

    def NISUM(self, I, xx, yy):
        # I 矩阵根据位置索引 xx yy 加到 N 矩阵相应的位置上
        # N 1 sim_size sim_size
        # I  1 sim_size sim_size
        # xx  sim_size sim_size
        # yy  sim_size sim_size
        N = torch.zeros_like(I)
        colnum = I.shape[-1]  # 列的数目
        posindex = xx + yy * colnum  # 图像点坐标问题
        posindex = posindex.view(-1).long()
        N = N.view(-1, 1)
        I = I.view(-1, 1)
        N.index_add_(0, posindex, I)
        N = N.view(-1, colnum)
        N = N.flip([0])  # y 周方向不一致 进行调整
        return N

    def depth2xxyy(self, depth, distcent, SlantPixSpacing=0.2):
        # 方位向不变 仅在距离向进行映射
        # zbuf 深度图
        # SlantPixSpacing 距离向分辨率 0.2
        H, W = depth.shape[-2:]  # H 高度 行数 W 宽度 列数
        range_length = depth - distcent
        yy = torch.floor(range_length / SlantPixSpacing + 0.5 * H - 2)
        xx = torch.ones(H, 1) * torch.arange(W)

        xx = xx.long().cuda()
        yy = yy.long().cuda()
        return xx, yy

    def optimgs2sarimgs(self, sarimgs, optimgs, depths, distance, SlantPixSpacing=0.2):
        batch_size = optimgs.shape[0]
        H, W = optimgs.shape[-2:]
        for batchnum in range(batch_size):
            optimg = optimgs[batchnum, ...]
            tmask = torch.ones_like(optimg)
            depth = depths[batchnum, ...]
            xx, yy = self.depth2xxyy(depth, distance, SlantPixSpacing)

            # optimg[yy > H - 1] = 0
            # yy[yy > H - 1] = 0
            #
            # optimg[yy < 0] = 0
            # yy[yy < 0] = 0
            tmask[yy > H - 1] = 0
            tmask[yy < 0] = 0

            yy = yy*tmask
            optimg = optimg*tmask

            sarimg = self.NISUM(optimg, xx, yy)
            sarimgs[batchnum, :, :, :] += sarimg

        return sarimgs

    def imgregcrop(self, Pixout, Target):

        # ------------------------------------------------------------------------------------
        with torch.no_grad():
            xlen, ylen, xm1b, ym1b, xm2b, ym2b, mnum, nnum = imgregist(Target, Pixout)
            # xm1b ym1b xm2b ym2b xlen ylen
            batchnum = Target.size(0)
            refreg = torch.empty((batchnum, 1, mnum, nnum))
            imgreg = torch.empty((batchnum, 1, mnum, nnum))

        for indbatch in range(batchnum):
            with torch.no_grad():
                xm1 = xm1b[indbatch]
                ym1 = ym1b[indbatch]
                xs1 = torch.ceil(xm1 - xlen).int()
                xe1 = torch.floor(xm1 + xlen - 0.1).int()
                ys1 = torch.ceil(ym1 - ylen).int()
                ye1 = torch.floor(ym1 + ylen - 0.1).int()

                xm2 = xm2b[indbatch]
                ym2 = ym2b[indbatch]
                xs2 = torch.ceil(xm2 - xlen).int()
                xe2 = torch.floor(xm2 + xlen - 0.1).int()
                ys2 = torch.ceil(ym2 - ylen).int()
                ye2 = torch.floor(ym2 + ylen - 0.1).int()

            refreg[indbatch, :, :, :] = Target[indbatch, :, xs1:(xe1 + 1), ys1:(ye1 + 1)]
            imgreg[indbatch, :, :, :] = Pixout[indbatch, :, xs2:(xe2 + 1), ys2:(ye2 + 1)]
        return imgreg,refreg

    def __init__(self, obj_filename, simsetting, device, savepath= ".\\results" ):

        self.device = device
        self.savepath = savepath
        self.traincount = 0
        self.epoch = 0

        self.deg = np.pi / 180.0
        self.distance = simsetting['distance'] # distance from camera to the object
        self.SlantPixSpacing = simsetting['SlantPixSpacing']
        self.imagesize = simsetting['imagesize']

        # Load obj file
        self.SLICY_mesh = load_objs_as_meshes([obj_filename], device=device)

        verts_shape = self.SLICY_mesh.verts_packed().shape
        self.deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)

        # also learn per vertex colors for our sphere mesh that define texture of the mesh
        self.verts_rgb = torch.full([1, verts_shape[0], 3], 0.7, device=device, requires_grad=True)

        # Add per vertex colors to texture the mesh
        self.SLICY_mesh.textures = TexturesVertex(verts_features=self.verts_rgb)
        self.shininess = torch.tensor(500.0, device=device, requires_grad=True)
        # ==============================================================
        self.modelsavepath = self.savepath + "\\models"
        self.modelpath = self.modelsavepath+ "\\SARNNM.pth"
        if not os.path.exists(self.modelsavepath):
            os.makedirs(self.modelsavepath)

        self.trainimagesavepath = self.savepath + "\\simimage\\train"
        if not os.path.exists(self.trainimagesavepath):
            os.makedirs(self.trainimagesavepath)

        self.testimagesavepath = self.savepath + "\\simimage\\test"
        if not os.path.exists(self.testimagesavepath):
            os.makedirs(self.testimagesavepath)
        #==============================================================
        ## 模型
        def get_parameter_number(net):
            total_num = sum(p.numel() for p in net.parameters())
            trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
            return {'Total': total_num, 'Trainable': trainable_num}

        # Create the SimPix
        self.netPix = SimPix(device).to(device)
        # Handle multi-gpu if desired
        # if (self.device.type == 'cuda') and (self.ngpu > 1):
        #     self.netPix = nn.DataParallel(self.netPix, list(range(ngpu)))
        # Apply the weights_init function to randomly initialize all weights
        # netD.apply(weights_init)
        print(self.netPix)

        paramnum = get_parameter_number(self.netPix)
        print('Total: ', paramnum['Total'], 'Trainable: ', paramnum['Trainable'])

        # ==============================================================
        self.criterionL2 = nn.MSELoss()
        self.criterionMAE = nn.L1Loss()
        # optimizer = optim.SGD( [ {'params': netEM.parameters()},  {'params': netPix.parameters()}, {'params': netRes.parameters()}] ,   lr=0.001, momentum=0.9)
        self.optimizer = torch.optim.Adam([{'params': self.netPix.parameters()}])

    def mloadmodel(self):
        if(os.path.exists(self.modelpath)):
            checkpoint = torch.load(self.modelpath)
            self.netPix.load_state_dict(checkpoint['netPix_state_dict'])
            self.deform_verts = checkpoint['deform_verts']
            self.verts_rgb = checkpoint['verts_rgb']
            self.shininess = checkpoint['shininess']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.traincount = checkpoint['traincount']
            self.epoch = checkpoint['epoch']
            return 1
        return 0

    def msave(self, epoch=0):
        torch.save({
            'netPix_state_dict': self.netPix.state_dict(),
            'deform_verts': self.deform_verts,
            'verts_rgb': self.verts_rgb,
            'shininess': self.shininess,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'traincount': self.traincount,
            'epoch': epoch
        }, self.modelpath)  # .pth

    def sarimgnnsim(self,sample, meshes):
        device = self.device
        distance = self.distance
        deg = self.deg
        imagesize = self.imagesize
        SlantPixSpacing = self.SlantPixSpacing

        elev = sample['elev']  # angle of elevation in degrees
        azim = sample['azim'] + 90.0  # angle of azimuth in degrees
        batch_size = azim.shape[0]
        distanceb = torch.tensor(distance, dtype=torch.float32, device=device).expand(batch_size)
        R, T = look_at_view_transform(distanceb, elev, azim, device=device)

        # cameras = OpenGLPerspectiveCameras(device=device, R=R_goal, T=T_goal)
        xsize = self.SlantPixSpacing * self.imagesize / 2
        xsize = torch.tensor(xsize, dtype=torch.float32, device=device).expand(batch_size)
        ysize = self.SlantPixSpacing * self.imagesize / 2 * torch.tan(elev * deg) * 1.4
        # ysize = xsize
        cameras = FoVOrthographicCameras(
            znear=1.0, zfar=distance + 40.0,
            max_y=ysize, min_y=-ysize,
            max_x=xsize, min_x=-xsize,
            R=R, T=T,
            device=device, )

        # Get the position of the camera based on the spherical angles
        camerapos = camera_position_from_spherical_angles(distance=distanceb, elevation=elev, azimuth=azim,
                                                          device=device)
        lightsdirection = camerapos
        lights = DirectionalLights(direction=lightsdirection, device=device)

        raster_settings = RasterizationSettings(
            image_size=imagesize,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        rasterizer = MeshRasterizer(raster_settings=raster_settings)
        shader = PhongShader(device=device)
        phong_renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=shader
        )

        materials = Materials(shininess=self.shininess, device=device)
        meshes = meshes.extend(batch_size)

        optimgs, fragments = phong_renderer(meshes_world=meshes, cameras=cameras, lights=lights,
                                            materials=materials)

        optimgs = optimgs[..., 0]  # 灰度图像，取第一个通道。保留通道位置，后处理进行卷积
        depths = fragments.zbuf[..., 0]  # 深度信息，取第一个通道

        H, W = optimgs.shape[-2:]
        sarimgs = torch.zeros(batch_size, 1, H, W, device=device)

        sarimgs = self.optimgs2sarimgs(sarimgs, optimgs, depths, distance, SlantPixSpacing)

        return sarimgs

    def coreprocesses(self, sample):
        samplenames = sample['samplename']
        Target = sample['realimg']
        src_mesh = self.SLICY_mesh
        new_src_mesh = src_mesh.offset_verts(self.deform_verts)
        sarimgs = self.sarimgnnsim(sample,new_src_mesh)
        sarimgs = self.netPix(sarimgs)
        Pixout = sarimgs
        # ===============================================
        # 配准裁切 imgreg,refreg
        imgreg,refreg = self.imgregcrop(Pixout, Target)

        return imgreg,refreg

    def mtrain(self, sample, epoch = 0):

        imgreg, refreg = self.coreprocesses(sample)
        # 计算损失函数值
        loss1 = self.criterionL2(imgreg,refreg)
        loss = loss1

        self.optimizer.zero_grad()  # 内存不变
        loss.backward()  # 释放网络运行占用的内存
        self.optimizer.step()  # 不占内存

        self.traincount += 1

        return loss.item()

    #==============================================================
    def mtest( self, sample, epoch = 0):

        with torch.no_grad():
            self.netPix.eval()
            imgreg, refreg = self.coreprocesses(sample)

            # 计算损失函数值
            loss1 = self.criterionL2(imgreg,refreg)

            loss = loss1

        if (epoch % 10 == 9):
            for batchnum in range(1):#range(len(samplenames)):
                samplename = samplenames[batchnum]
                imagename ='ep' + str(epoch) + '-' + samplename[:-4] +  '-sarimg' + ".png"
                imagepathname = os.path.join(self.testimagesavepath , imagename)
                sarimg = sarimgs[batchnum, :, :, :]
                vutils.save_image(sarimg, fp=imagepathname, normalize=False) # 也可以一个batch存为一个图片 就是命名有点麻烦
                imagename ='ep' + str(epoch) + '-' + samplename[:-4] +  '-sarimgn' + ".png"
                imagepathname = os.path.join(self.testimagesavepath , imagename)
                vutils.save_image(sarimg, fp=imagepathname, normalize=True)

                imagename = 'ep' + str(epoch) + '-' +samplename[:-4] +  '-Pixout' + ".png"
                imagepathname = os.path.join(self.testimagesavepath , imagename)
                Pixoutimg = Pixout[batchnum, :, :, :]
                vutils.save_image(Pixoutimg, fp=imagepathname, normalize=False)


        return loss.item()








#
#
# img = img.cpu().numpy()
# plt.figure(figsize=(10, 10))
# plt.imshow(img)  # only plot the alpha channel of the RGBA image
# plt.grid(False)
# # plt.colorbar()
# # plt.show()
#
#
# optim = torch.optim.Adam([azim], lr=0.5)
# azims = []
# losses = []
#
# loop = tqdm(range(80))
# for i in loop:
#     optim.zero_grad()
#     desc = f"Azim: {azim.detach().numpy()}"
#     loop.set_description(desc)
#     # Update camera extrinsics with the current azimuth angle
#     R, T = look_at_view_transform( dist=distance, elev=elev, azim=azim)
#
#     fragments = rasterizer(car_mesh, R=R.cuda(), T=T.cuda())
#
#     depth = fragments.zbuf[0, :, :, 0]
#
#     # NOTE: Strange behaviour:
#     # Using a negative losses optimizes the azimuth correctly
#     # loss = (((depth_goal - depth)**2).mean())
#
#
#     # The typical l2 loss however, does not
#     # loss = (depth_goal - depth).pow(2).sum().sqrt()
#     diffdepth = ((depth - depth_goal) ** 2)
#     depthmask = depth_goal != -1
#     loss = diffdepth[depthmask].mean()
#
#
#     loss.backward()
#     optim.step()
#
#     azims.append(float(azim))
#     losses.append(loss)
#
# plt.figure()
# plt.subplot(121)
# plt.plot(losses)
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
#
# plt.subplot(122)
#
# plt.plot(azims)
# plt.xlabel("Iteration")
# plt.ylabel("Azimuth")
#
# plt.tight_layout()
# plt.show()