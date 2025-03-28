import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib

from pytorch3d.io import load_objs_as_meshes, IO, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    RasterizationSettings,
    MeshRasterizer,MeshRenderer,MeshRendererWithFragments,
    FoVOrthographicCameras,FoVPerspectiveCameras,
    PointLights,TexturesVertex,Materials,
    camera_position_from_spherical_angles,
    DirectionalLights,
    SARDirectionalLights,
)
from pytorch3d.renderer import SoftPhongShader, HardFlatShader, SARSoftPhongShader
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.utils import ico_sphere


from utils.plot_image_grid import image_grid
from utils.ImgRegi import imgregist
from utils.imgplot import imgplot



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
        # self.Ac1 = nn.ReLU()
        #
        # self.Conv2 = nn.Conv2d(1, 4, kernel_size=9, stride=1, padding=4, bias=True) # 相干斑噪声
        # self.Ac2 = nn.ReLU()
        #
        # self.Conv3 = nn.Conv2d(6, 1, kernel_size=5, stride=1, padding=2, bias=False) # 取值组合
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

        imppathfile = 'Impt108.csv' #54*2
        Kimp = pd.read_csv(imppathfile, header=None, index_col=False)  # 由于每行末尾有个逗号，通过这种方式去除影响
        Kimp = Kimp.fillna(0).values  # sim_size sim_size
        Kimpt = torch.as_tensor(Kimp, dtype=torch.float32, device=self.device).view(1, 1, Kimp.shape[0], Kimp.shape[1])

        self.Conv1.weight.data = Kimpt
        # normweight = torch.sqrt(torch.sum(self.Conv1.weight.data.pow(2))) 
        # self.Conv1.weight.data = self.Conv1.weight.data/normweight

    def forward(self, input):

        # outputimp = self.Convq(input) # 固定卷积核之前的调整散射点强度
        # outputimp = self.Acq(outputimp)
        # outputimp = F.conv2d(outputimp, self.Kimp, stride=1, padding=53)
        

        output = self.Conv1(input) # 仅使用卷积核
        # outputimp = self.Ac1(outputimp)

        # output = self.Conv2(input)  # 1 64 sim_size sim_size
        # output = self.Ac2(output)
        #
        # output = torch.cat((input, outputimp, output), dim=1)
        #
        # output = self.Conv3(output)
        # output = self.Ac3(output)

        output = self.maxpool(output)  # 最佳位置有待讨论
        # output = F.interpolate(output, size=[54, 54], mode='bilinear')


        return output

# ==========================================================




# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh, loss):
    # and (b) the edge length of the predicted mesh
    loss["edge"] = mesh_edge_loss(mesh)

    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)

    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")

def IoU(inputs, targets, smooth = 0):
    #intersection is equivalent to True Positive count
    #union is the mutually inclusive area of all labels & predictions 
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
    IoU = (intersection + smooth)/(union + smooth)
    return IoU



class SARNNM:


    '''
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

            # imgt = xx
            # imgplot(imgt)
            #
            # imgt = yy
            # imgplot(imgt)
            #
            # plt.show()
            a = 1

        return sarimgs
    '''

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

    def __init__(self, obj_filename, simsetting, device,
                 EPNum = 0, savepath= r"./results", logpath = r'./logsnew/runs/EP' ):
        self.losses = {"rgb": {"weight": 0.0, "values": []},
                "silhouette": {"weight": 1.0, "values": []},
                "edge": {"weight": 0.0, "values": []},
                "normal": {"weight": 0.01, "values": []},
                "laplacian": {"weight": 1.0, "values": []},
                }

        self.device = device
        self.savepath = savepath + str(EPNum)
        self.traincount = 0
        self.epoch = 0
        self.writer = SummaryWriter(logpath+str(EPNum))

        self.deg = np.pi / 180.0
        self.distance = simsetting['distance'] # distance from camera to the object
        self.SlantPixSpacing = simsetting['SlantPixSpacing']
        self.imagesize = simsetting['imagesize']

        # Load obj file
        src_mesh = load_objs_as_meshes([obj_filename], device=device)
        #src_mesh = ico_sphere(5, device)
        # generates a subdivided icosahedron, 5 is the subdivided level, originally 12 vertices and 20 triangular faces
        src_mesh.offset_verts_(-torch.tensor([0, -1.0, 0])) # shifted downwards for 1 

        src_mesh.scale_verts_(simsetting['scalefactor']) #sphere expand or shrink
        self.src_mesh = src_mesh
        #plot_scene({
        #"Mesh": {
        #       "mesh": src_mesh,
        #       }
        #})
        groundmask = self.src_mesh.verts_packed()
        groundmask = groundmask[:,1] != 0 # slect only vertices that are not on the Y=0 plane
        groundmask = groundmask.view(-1,1)*torch.ones(1,3)
        groundmask.requires_grad = False
        self.ground_mask = groundmask

        verts_shape = self.src_mesh.verts_packed().shape # retrievws all vertices of the mesh as a (v,3) tensor, V is the number of vertices
        self.deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)# true stands that the network can leran to adjust vertex positions during training, allow learnale vertex displacement, meaning the model can modify the shape of the mesh dynamically
        # also learn per vertex colors for our sphere mesh that define texture of the mesh
        self.verts_rgb = torch.full([1, verts_shape[0], 3], 0.665, device=device, requires_grad=True)
        # Add per vertex colors to texture the mesh
        self.src_mesh.textures = TexturesVertex(verts_features=self.verts_rgb)# assign the vertex colors to the mesh

        # 材质参数
        self.ambient_color = torch.tensor(((0.0, 0.0, 0.0),), device=device, requires_grad=True) # true: model can learn and adjust these parameters, no ambient color
        self.shininess = torch.tensor(64.0, device=device, requires_grad=True) # control how sharp the highlights are 
        self.diffuse_color = torch.tensor(((1.0, 1.0, 1.0),), device=device, requires_grad=True) # full light reflection
        self.specular_color = torch.tensor(((1.0, 1.0, 1.0),), device=device, requires_grad=True) # define the reflective highlight color
        self.materials = Materials(shininess=self.shininess, ambient_color=self.ambient_color, 
            diffuse_color=self.diffuse_color, specular_color=self.specular_color, device=device)


        # ==============================================================
        self.modelsavepath = self.savepath + r"/models"
        if not os.path.exists(self.modelsavepath):
            os.makedirs(self.modelsavepath)
        self.modelpath = self.modelsavepath+ r"/SARNNM.pth"

        self.trainimagesavepath = self.savepath + r"/simimage/train"
        if not os.path.exists(self.trainimagesavepath):
            os.makedirs(self.trainimagesavepath)

        self.testimagesavepath = self.savepath + r"/simimage/test"
        if not os.path.exists(self.testimagesavepath):
            os.makedirs(self.testimagesavepath)

        self.meshsavepath = self.savepath + r"/meshs"
        if not os.path.exists(self.meshsavepath):
            os.makedirs(self.meshsavepath)
        #==============================================================
        ## 后处理模型
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
        # print(self.netPix)
        # paramnum = get_parameter_number(self.netPix)
        # print('Total: ', paramnum['Total'], 'Trainable: ', paramnum['Trainable'])

        # ==============================================================
        self.criterionL2 = nn.MSELoss()
        self.criterionMAE = nn.L1Loss()
        # optimizer = optim.SGD( [ {'params': netEM.parameters()},  {'params': netPix.parameters()}, {'params': netRes.parameters()}] ,   lr=0.001, momentum=0.9)
        # paramsets = [{'params': self.netPix.parameters()},{'params': self.shininess},
        #                                    {'params': self.deform_verts}, {'params':self.verts_rgb}]
        '''
        paramsets = [#{'params': self.netPix.parameters()},
                     {'params': self.shininess},
                     # {'params': self.ambient_color},
                     {'params': self.diffuse_color},
                     {'params': self.specular_color},
                     {'params': self.deform_verts},
                     # {'params':self.verts_rgb}
        ]
        self.optimizer =  torch.optim.SGD(paramsets, lr=1.0, momentum=0.9)choose which parameters to learn, with which learning rate and with which smooths updates
        '''

    def mloadmodel(self, modelsavepath=None):
        if modelsavepath == None:
            if(os.path.exists(self.modelpath)):
                checkpoint = torch.load(self.modelpath)
                self.netPix.load_state_dict(checkpoint['netPix_state_dict'])
                self.deform_verts = checkpoint['deform_verts']
                self.verts_rgb = checkpoint['verts_rgb']
                self.src_mesh.textures = TexturesVertex(verts_features=self.verts_rgb)
                self.shininess = checkpoint['shininess']
                self.ambient_color = checkpoint['ambient_color']
                self.diffuse_color = checkpoint['diffuse_color']
                self.specular_color = checkpoint['specular_color']
                self.materials = Materials(shininess=self.shininess, ambient_color=self.ambient_color, 
                    diffuse_color=self.diffuse_color, specular_color=self.specular_color, device=self.device)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.traincount = checkpoint['traincount']
                self.epoch = checkpoint['epoch']
                return 1
            return 0
        else:
            modelpath = modelsavepath + r"/SARNNM.pth"
            if(os.path.exists(modelpath)):
                checkpoint = torch.load(modelpath)
                self.netPix.load_state_dict(checkpoint['netPix_state_dict'])
                self.deform_verts = checkpoint['deform_verts']
                self.verts_rgb = checkpoint['verts_rgb']
                self.src_mesh.textures = TexturesVertex(verts_features=self.verts_rgb)
                self.shininess = checkpoint['shininess']
                self.ambient_color = checkpoint['ambient_color']
                self.diffuse_color = checkpoint['diffuse_color']
                self.specular_color = checkpoint['specular_color']
                self.materials = Materials(shininess=self.shininess, ambient_color=self.ambient_color, 
                    diffuse_color=self.diffuse_color, specular_color=self.specular_color, device=self.device)
                # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # self.traincount = checkpoint['traincount']
                # self.epoch = checkpoint['epoch']
                return 1
            return 0


    def msave(self, epoch=0):
        torch.save({
            'netPix_state_dict': self.netPix.state_dict(),# neural network weights
            'deform_verts': self.deform_verts,
            'verts_rgb': self.verts_rgb,
            'shininess': self.shininess,
            'ambient_color': self.ambient_color,
            'diffuse_color': self.diffuse_color,
            'specular_color': self.specular_color,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'traincount': self.traincount,
            'epoch': epoch
        }, self.modelpath)  # .pth

    def sarimgnnsim(self, sample, meshes):  # sample is the ground truth table( a sar image from specific view angle, and meshes is from the new 3D model, now have to generate the 2d figure of this new 3D model from the same angle as the as the goundtruth looking at angle, so that it can be compared and do loss calculation)
        device = self.device
        elev = sample['elev']  # angle of elevation in degrees
        azim = sample['azim'] # angle of azimuth in degrees SLICY
        batch_size = azim.shape[0]
        distance = self.distance #camera to object
        distanceb = torch.tensor(distance, dtype=torch.float32, device=device).expand(batch_size)
        deg = self.deg
        
        meshes = meshes.extend(batch_size) #duplicates the mesh across batch_size, ensuring each sample has its own copy
        #=========================================
        # here can set the view angle, possible place to do the optimization of angle
        R, T = look_at_view_transform(distanceb, elev, azim, device=device)#computes camera rotation and traslation, ensures that camera looks at the object from the given elevation and azimuth
        xsize = self.SlantPixSpacing * self.imagesize / 2
        xsize = torch.tensor(xsize, device=device).expand(batch_size)
        xsize = torch.torch.ceil(xsize * 1.1) #round up for safety margins
        ysize = self.SlantPixSpacing * self.imagesize / 2 * torch.tan(elev * deg)
        ysize = torch.torch.ceil(ysize * 1.1) # xsize and ysize define the rectangular projection limits
        # cameras1 
        lights_cameras = FoVOrthographicCameras(
            znear=0.5, zfar=distance + 40.0,# creates an orthographic camera with znear and zfar defining its depth range
            max_y=ysize, min_y=-ysize,
            max_x=xsize, min_x=-xsize,
            R=R, T=T,
            device=device, )

        # Get the position of the camera based on the spherical angles
        camerapos = camera_position_from_spherical_angles(distance=distanceb, elevation=elev, azimuth=azim,
                                                          device=device)
        lightsdirection = camerapos
        lights = DirectionalLights(ambient_color=((0.0, 0.0, 0.0),),
                                   diffuse_color=((1.0, 1.0, 1.0),),
                                   specular_color=((1.0, 1.0, 1.0),),
                                   direction=lightsdirection, device=device) # 排除光照影响 完全由材质和纹理决定
        # with torch.no_grad():
        #create a soft rasterizer as follows:
        imgsamfactor = 2 #resolution
        image_size = self.imagesize* imgsamfactor
        # Renderer for Image-based 3D Reasoning', ICCV 2019
        sigma = 1e-4
        raster_settings_soft = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,  # to smooth rendering
            faces_per_pixel=2,
            bin_size = 0,
        )
        # MeshRenderer is a differentiable renderer in PyTorch3D that takes a 3D mesh and converts it into a 2D image. 
        # It consists of two main components: 1. meshrasterizer that converts the 3D mesh into a set of 2D pixels by projecting it into the image plane.
        # 2. Shader: determins the final appearance of the image, including lighting, shading, and color blending.
        # Differentiable soft renderer using per vertex RGB colors for texture
        # different shader setting allow us to generate: shadow mao(depth based visibility of objects), Silhouettes(binary masks of objects), Textured renders(full RGB images with shading and lighting effect)
        renderer_textured = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings_soft
            ),
            shader=SoftPhongShader(device=device)
        )

        _, fragments = renderer_textured(meshes_world=meshes, cameras=lights_cameras, lights=lights,
                                            materials=self.materials)# extracts the depth buffer, which stores how far each pixel is from the camera,

        shadowmaps = fragments.zbuf[..., 0]  # 深度信息，取第一个通道 computes shadow maps to detect occluded areas
        # the shadow map helps simulate SAR reflections by pass
        # 1. occlusion information: which parts of the object block light
        # 2.depth-based visibility: how SAR waves interact with different surfaces
        # 3. better material response simulation(light reflection depend on depth) 

        #=========================================
        #silimgs1
        R, T = look_at_view_transform(distanceb, elev, azim, device=device)
        xsize = self.SlantPixSpacing * self.imagesize / 2
        xsize = torch.tensor(xsize, device=device).expand(batch_size)
        xsize = torch.ceil(xsize) #torch.ceil(xsize * 1.1)
        ysize = self.SlantPixSpacing * self.imagesize / 2 * torch.tan(elev * deg)
        ysize = torch.ceil(ysize) #torch.ceil(ysize * 3)
        # cameras1 
        SAR_cameras1 = FoVOrthographicCameras(
            znear=0.5, zfar=distance + 40.0,
            max_y=ysize, min_y=-ysize,
            max_x=xsize, min_x=-xsize,
            R=R, T=T,
            device=device, )

        # Get the position of the camera based on the spherical angles
        camerapos = camera_position_from_spherical_angles(distance=distanceb, elevation=elev, azimuth=azim,
                                                          device=device)
        lightsdirection = camerapos

        # 影响镜面反射计算, define lighting conditions
        SARlights = SARDirectionalLights(ambient_color=((0.0, 0.0, 0.0),),
                                   diffuse_color=((1.0, 1.0, 1.0),),
                                   specular_color=((1.0, 1.0, 1.0),),
                                   direction=lightsdirection, device=device) 


        # with torch.no_grad():

        imgsamfactor = 1   #resolution
        image_size = self.imagesize* imgsamfactor  
        # Renderer for Image-based 3D Reasoning', ICCV 2019
        sigma = 1e-4
        raster_settings_soft = RasterizationSettings(  #defines how meshs will be converted to pixels
            image_size=image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,  #controls blurring of the object edges for differentiability, ensures gradients flow smoothly during backpropagation
            faces_per_pixel=20, #up to 20 faces contribute to a single pixel
            bin_size = 0,
        )
        # Differentiable soft renderer using per vertex RGB colors for texture
        blend_params = BlendParams(gamma=1e-2, background_color = (0.0, 0.0, 0.0)) # gamma controls intensity blending in the rendeer, bgc ensures objects are rendered ona black background
        
        renderer_textured1 = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings_soft
            ),
            shader=SoftPhongShader(device=device, blend_params = blend_params) # uses blending settings to emphasize object contours
        )  # converts 3D geometry to 2D, uses the phong shading model with blending parameters

        simimgs1 = renderer_textured1(meshes_world=meshes, cameras=SAR_cameras1, lights=SARlights,
                                            materials=self.materials)

        silimgs1 = simimgs1[..., 3] # the original simimgs1(at right side) is a RGBA image(red, green, blue, alpha), [...,3]extracts the Alpha(opacity) channel
        # the result is a silhouette mask, which highlights the shape of the object against the background
        # shadow map provides depth information, which helps model how SAR signals interact with different surfaces
        # shadow map model occlusions and SAR wave interactions
        # silhouette map isolates the shape of the object, removing all textures and sahding for clean shpe representation, silhouettes help with edge detection and target recognition        


        # 正交相机位置调整
        elev = 90.0 - elev  # angle of elevation in degrees,align with a top-down SAR view
        azim = azim+ 180.0  # angle of azimuth in degrees, reveese the perspective
        R, T = look_at_view_transform(distanceb, elev, azim, up= ((0, -1, 0),), device=device)
        xsize = self.SlantPixSpacing * self.imagesize / 2
        xsize = torch.tensor(xsize, device=device).expand(batch_size)
        xsize = torch.ceil(xsize)
        ysize = self.SlantPixSpacing * self.imagesize / 2  # 此角度下即为斜距向大小
        ysize = torch.tensor(ysize, device=device).expand(batch_size)
        ysize = torch.ceil(ysize) # the projection area that the camera covers
        # cameras2 计算斜距投影
        SAR_cameras2 = FoVOrthographicCameras(
            znear=0.5, zfar=distance + 40.0,
            max_y=ysize, min_y=-ysize,
            max_x=xsize, min_x=-xsize,
            R=R, T=T,
            device=device, )


        imgsamfactor = 1
        image_size = self.imagesize* imgsamfactor
        sigma = 1e-4
        raster_settings_soft = RasterizationSettings(
            image_size= image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            faces_per_pixel=20,
            bin_size = 0,
        )

        # Differentiable soft renderer using per vertex RGB colors for texture
        renderer_textured2 = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings_soft
            ),
            shader=SARSoftPhongShader(device=device, blend_params = blend_params)
        )
       

        simimgs2, shadowmask = renderer_textured2(meshes_world = meshes, cameras = SAR_cameras2, 
                                                lights = SARlights, materials = self.materials,
                                                shadowmaps = shadowmaps, lights_cameras = lights_cameras)

        # shadowmask: occlusion mask
        sarimgs = simimgs2[..., 0]  # 灰度图像，取第一个通道。保留通道位置，后处理进行卷积, SAR gryscale image which stores radar intensity(brightness of reflections)
        # storng reflection: metallic surfaces, weaker reflection: rough/soft materials, no return: shadowed areas
        silimgs2 = simimgs2[..., 3] # Silhouette image, represents object contours,outer boundary of the object

        # shadowmask = torch.prod(shadowmask,dim=-1)
        shadowmask = shadowmask.sum(dim=-1,keepdim=False) # summarizes shadowmask(has multiple layers due to depth contribution) to a single channel
        #determines which regoins are occluded from the raddar sensor's view, used to remove incorrect reflections caused by multiple bounces
        # silimgs = (silimgs1 + silimgs2)*(sarimgs == 0)*(shadowmask==0)
        # silimgs = (silimgs1 + silimgs2)*(shadowmask==0)
        silimgs = (silimgs1 + silimgs2)*(sarimgs==0)   # represents the object's outline in the SAR projection

        sarimgs = torch.unsqueeze(sarimgs, 1)  # simulated radar intensity ar each pixel, high: stronger radar reflection, low: weak or no return
        silimgs = torch.unsqueeze(silimgs, 1)


        #imgt = SARimgs[0]
        #imgplot(imgt)
        #plt.show()

        # H, W = optimgs.shape[-2:]
        # sarimgs = torch.zeros(batch_size, 1, H, W, device=device)
        # sarimgs = self.optimgs2sarimgs(sarimgs, optimgs, depths, distance, SlantPixSpacing)

        # imgt = sarimgs[0, 0, ...]
        # imgplot(imgt)
        # plt.show()

        # H, W = silimgs.shape[-2:]
        # sarsilimgs = torch.zeros(batch_size, 1, H, W, device=device)
        # sarsilimgs = self.optimgs2sarimgs(sarsilimgs, silimgs, depths, distance, SlantPixSpacing)

        # imgt = sarsilimgs[0, 0, ...]
        # imgplot(imgt)
        # plt.show()
        # a = imgt.detach().cpu().numpy()

        return sarimgs, silimgs

    def coreprocesses(self, sample):

        src_mesh = self.src_mesh
        # new_src_mesh = src_mesh.offset_verts(self.deform_verts*self.ground_mask) 
        new_src_mesh = src_mesh.offset_verts(self.deform_verts) 
        new_src_mesh.textures = TexturesVertex(verts_features=self.verts_rgb) 
        plot_scene({
          "Mesh": {
               "mesh": new_src_mesh,
                   }
          })
        loss = {k: torch.tensor(0.0, device=self.device) for k in self.losses} #initialize loss
        update_mesh_shape_prior_losses(new_src_mesh, loss) #之后可以开启进行几何约束


        sarimgs,sarsilimgs = self.sarimgnnsim(sample,new_src_mesh)# render from 3D(new model) TO 2D, outputs will be compared to real SAR images for loss computation
        
        # Pixout = self.netPix(sarimgs)
        # ===============================================
        # 计算损失函数值
        Targetimg = sample['realimg']  # the real SAR image, ground truth table
        Targetsilimg = (Targetimg<0.01)*1.0  #the real SAR silhouette mask(where the pixel intensity is very low)

        # sarimgs = self.netPix(sarimgs)
        # sarsilimgs = self.netPix(sarsilimgs)
        # sarimgs = F.max_pool2d(sarimgs,kernel_size=2, stride=2, padding=0)
        # sarsilimgs = F.max_pool2d(sarsilimgs, kernel_size=2, stride=2, padding=0)
        
        # 配准裁切 imgreg,refreg
        if False: # self.epoch>=50:
            imgreg,refreg = self.imgregcrop(sarimgs, Target)  # alogn the SAR image with the real target before computing the loss
        else:
            imgreg = sarimgs
            refreg = Targetimg
        
        self.sarimgs = sarimgs

        loss_rgb = self.criterionL2(sarimgs,Targetimg)
        loss["rgb"] = loss_rgb
        # loss_sil = (1.0- IoU(Targetsilimg,sarsilimgs*(sarsilimgs>0.01)))*0.1
        loss_sil = (1.0- IoU(Targetsilimg,sarsilimgs))*0.1
        loss_sil += self.criterionL2(Targetsilimg,sarsilimgs)  # treat both factor IOU nad L2 loss as loss part, note the 0.1 for balance here
        loss["silhouette"] = loss_sil




        # a = sarsilimgs.detach().cpu().numpy()

        # loss_silhouette = self.criterionL2(torch.tanh(sarsilimgs), torch.tanh(Target))
        # loss_silhouette = torch.tensor(0.0, device=self.device) 
        # loss["silhouette"] += loss_silhouette


        # Weighted sum of the losses
        sum_loss = torch.tensor(0.0, device=self.device)
        for k, l in loss.items():
            sum_loss += l * self.losses[k]["weight"] # for each item, give different weights
            self.losses[k]["values"].append(l) # add original loss to these losses part
 

        return loss,sum_loss

    def mtrain(self, sample, epoch = 0):
        self.optimizer.zero_grad()  # 内存不变, clears previous gradients before backpropagation to prevent accumulation

        loss, sum_loss= self.coreprocesses(sample)

        
        sum_loss.backward()  # 释放网络运行占用的内存 # computes the gradients of sum_loss with respect to model parameters
        self.optimizer.step()  # 不占内存 #updates model parameters based on comluted gradients

        self.traincount += 1

        loss = sum_loss
        return loss.item()

    #==============================================================
    def mtest( self, sample, epoch = 0):

        with torch.no_grad(): # disabling gradient computation, saves memory by disabling gradient tracking
            self.netPix.eval() # switches the model to evaluation mode, disabling dropout and batch norm updates
            samplenames = sample['samplename']
            loss,sum_loss= self.coreprocesses(sample)
            # sarimgs = F.max_pool2d(sarimgs, kernel_size=2, stride=2, padding=0)
            # 计算损失函数值

            loss = sum_loss

            if (epoch % 10 == 9):
                for batchnum in range(1):#range(len(samplenames)): #一个batch寸一张图
                    sarimg = self.sarimgs[batchnum, :, :, :]
                    
                    samplename = samplenames[batchnum]
                    imagename ='ep' + str(epoch) + '-' + samplename[:-4] +  '-sarimg' + ".png"
                    imagepathname = os.path.join(self.testimagesavepath , imagename)
                    
                    vutils.save_image(sarimg, fp=imagepathname, normalize=False) # 也可以一个batch存为一个图片 就是命名有点麻烦
                    #imagename ='ep' + str(epoch) + '-' + samplename[:-4] +  '-sarimgn' + ".png"
                    #imagepathname = os.path.join(self.testimagesavepath , imagename)
                    #vutils.save_image(sarimg, fp=imagepathname, normalize=True)

                    # imagename = 'ep' + str(epoch) + '-' +samplename[:-4] +  '-Pixout' + ".png"
                    # imagepathname = os.path.join(self.testimagesavepath , imagename)
                    # Pixoutimg = self.Pixout[batchnum, :, :, :]
                    # vutils.save_image(Pixoutimg, fp=imagepathname, normalize=False)

        return loss.item()

    def meshvisual(self, epoch = 0, meshname = 'SLICY'):
        with torch.no_grad():
            # 目标模型调整后可视化
            # 1.三维模型光学渲染
            src_mesh = self.src_mesh
            new_src_mesh = src_mesh.offset_verts(self.deform_verts)
            # new_src_mesh.textures = TexturesVertex(verts_features=self.verts_rgb)
            meshes = new_src_mesh

            device = self.device
            distance = 15
            deg = self.deg
            imagesize = 400 #self.imagesize
            # SlantPixSpacing = self.SlantPixSpacing

            elev = torch.tensor(30.0, dtype=torch.float32, device=device) #sample['elev']  # angle of elevation in degrees 15.0
            azim = torch.tensor(45.0, dtype=torch.float32, device=device)  # angle of azimuth in degrees 135.0
            batch_size = 1 #azim.shape[0]
            distanceb = torch.tensor(distance, dtype=torch.float32, device=device).expand(batch_size)
            R, T = look_at_view_transform(distanceb, elev, azim, device=device)


            # cameras = OpenGLPerspectiveCameras(device=device, R=R_goal, T=T_goal)
            # xsize = self.SlantPixSpacing * self.imagesize / 2
            # xsize = torch.tensor(xsize, dtype=torch.float32, device=device).expand(batch_size)
            # ysize = self.SlantPixSpacing * self.imagesize / 2 * torch.tan(elev * deg) * 1.4
            # ysize = xsize
            cameras = FoVPerspectiveCameras(
                znear=1.0, zfar=distance + 40.0,
                R=R, T=T,
                device=device, )

            # Get the position of the camera based on the spherical angles
            camerapos = camera_position_from_spherical_angles(distance=distanceb, elevation=elev, azimuth=azim,
                                                              device=device)
            lightsdirection = camerapos
            # lights = DirectionalLights(ambient_color=((0.8, 0.8, 0.8),),
            #                            diffuse_color=((0.5, 0.5, 0.5),),
            #                            specular_color=((0.2, 0.2, 0.2),),
            #                            direction=lightsdirection, device=device)

            lights = DirectionalLights(ambient_color=((0.0, 0.0, 0.0),),
                                       diffuse_color=((1.0, 1.0, 1.0),),
                                       specular_color=((1.0, 1.0, 1.0),),
                                       direction=lightsdirection, device=device)

            raster_settings = RasterizationSettings(
                image_size=imagesize,
                blur_radius=0.0,
                faces_per_pixel=3,
            )
            rasterizer = MeshRasterizer(raster_settings=raster_settings)
            shader = HardFlatShader(device=device)
            phong_renderer = MeshRenderer(
                rasterizer=rasterizer,
                shader=shader
            )


            meshes = meshes.extend(batch_size)
            materials = Materials(shininess=self.shininess, ambient_color=self.ambient_color,
                                  diffuse_color=self.diffuse_color, specular_color=self.specular_color, device=device)

            optimgs = phong_renderer(meshes_world=meshes, materials=materials,
                                        cameras=cameras, lights=lights)


            # optimgs = optimgs[..., 0]
            # imgt = optimgs[0]
            # imgplot(imgt)
            #
            # plt.show()

            optimg  = optimgs[..., 0] # 4个通道取第一个通道
            # 命名先大类 再小类 变量放在最后
            imagename =meshname+ '-elev-' + str(int(elev.cpu().numpy())) + '-azim-' + str(int(azim.cpu().numpy()-90))   + '-ep' + str(epoch) + '.png'
            imagepathname = os.path.join(self.meshsavepath, imagename)
            vutils.save_image(optimg, fp=imagepathname, normalize=False)





    def meshsave(self, epoch = 0, meshname ="SLICY"):
        # 目标模型调整后可视化
        # 1.三维模型存储 PyTorch3D 0.3版本 仅能保存obj三维模型，无法保存材质及纹理

        src_mesh = self.src_mesh
        new_src_mesh = src_mesh.offset_verts(self.deform_verts)



        # Store the predicted mesh using save_obj 命名从大到小
        # meshname = meshname + '-ep' + str(epoch) +  '.obj'
        # final_obj = os.path.join(self.meshsavepath, meshname)

        # Fetch the verts and faces of the final predicted mesh
        # final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
        # save_obj(final_obj, final_verts, final_faces)

        # Store the predicted mesh using save_obj 命名从大到小
        meshname = meshname + '-ep' + str(epoch) +  '.ply'
        final_ply = os.path.join(self.meshsavepath, meshname)
        IO().save_mesh(new_src_mesh, final_ply, colors_as_uint8=True)  # save vertices, faces, and vertex colors


        a = 1

