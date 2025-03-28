import os
import random
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



# Decide which device we want to run on
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
if ngpu>0:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def set_seed(seed=2021): # seed的数值可以随意设置，本人不清楚有没有推荐数值
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #根据文档，torch.manual_seed(seed)应该已经为所有设备设置seed
    #但是torch.cuda.manual_seed(seed)在没有gpu时也可调用，这样写没什么坏处
    torch.cuda.manual_seed(seed)
    #cuDNN在使用deterministic模式时（下面两行），可能会造成性能下降（取决于model）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
def imshow(img):

    npimg = img.squeeze().cpu().numpy()  # 将torch.FloatTensor 转换为numpy
    # plt.axis("off")  # 不显示坐标尺寸
    plt.figure()
    plt.imshow(npimg)
    plt.show()


# def batch_fftshift2d(x):
#     real, imag = torch.unbind(x, -1)
#     for dim in range(1, len(real.size())):
#         n_shift = real.size(dim)//2
#         if real.size(dim) % 2 != 0:
#             n_shift += 1  # for odd-sized images
#         real = roll_n(real, axis=dim, n=n_shift)
#         imag = roll_n(imag, axis=dim, n=n_shift)
#     return torch.stack((real, imag), -1)  # last dim=2 (real&imag)
#
# def batch_ifftshift2d(x):
#     real, imag = torch.unbind(x, -1)
#     for dim in range(len(real.size()) - 1, 0, -1):
#         real = roll_n(real, axis=dim, n=real.size(dim)//2)
#         imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
#     return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def ifftshift2d(x):
    dims = (x.ndim-2, x.ndim-1)
    shifts = (  x.size(dims[0]) //2,  x.size(dims[1]) //2 )
    x = torch.roll(x, dims=dims, shifts=shifts)
    return x  # last dim=2 (real&imag)

# def fftshift(real, imag):
#     for dim in range(1, len(real.size())):
#         real = fft.roll_n(real, axis=dim, n=real.size(dim)//2)
#         imag = fft.roll_n(imag, axis=dim, n=imag.size(dim)//2)
#     return real, imag
#
# def ifftshift(real, imag):
#     for dim in range(len(real.size()) - 1, 0, -1):
#         real = fft.roll_n(real, axis=dim, n=real.size(dim)//2)
#         imag = fft.roll_n(imag, axis=dim, n=imag.size(dim)//2)
#     return real, imag




def myabs(x):
    realindices = torch.tensor([0])
    imindices = torch.tensor([1])
    rx = torch.index_select(x, dim=-1, index=realindices)
    ix = torch.index_select(x, dim=-1, index=imindices)
    # absx = torch.sqrt(rx*rx + ix*ix) # 如果只是求峰值的位置 sqrt 可以不用
    absx = rx * rx + ix * ix
    absx = absx.squeeze(dim=-1)
    return absx

def mycompmulconj(A, B):
    # (a + bi)(c + di) = (ac - bd) + (bc + ad)i
    # realindices = torch.tensor([0])
    # imindices = torch.tensor([1])
    # a = torch.index_select(A, dim=-1, index=realindices)
    # b = torch.index_select(A, dim=-1, index=imindices)
    # c = torch.index_select(B, dim=-1, index=realindices)
    # d = torch.index_select(B, dim=-1, index=imindices)
    a = A.real # torch.index_select(A, dim=-1, index=realindices)
    b = A.imag # torch.index_select(A, dim=-1, index=imindices)
    c = B.real # torch.index_select(B, dim=-1, index=realindices)
    d = B.imag # torch.index_select(B, dim=-1, index=imindices)
    d = -d #B矩阵取共轭
    cr = a*c - b*d
    ci = b*c + a*d
    C =  torch.cat((cr, ci), dim=-1)
    return C

# ---------------------------------------------------------------------------------------
def imgregist(ref, img):
    # 输入为2个batch的图片，# batch_size 1 54 54
    # 输出为相应配准信息
    data1 = ref
    data2 = img

    A = torch.fft.fft2(data1) # 2D FFT
    B = torch.fft.fft2(data2) # 2D FFT
    C = A*B.conj() #/ torch.sqrt(  A.abs().pow(2)+ B.abs().pow(2) )

    data3 = torch.abs(torch.fft.ifft2(C)).squeeze()
    # data3 = torch.abs(C).squeeze()
    data3 = torch.fft.fftshift(data3, dim=(-2,-1))

    mdata, __ = torch.max(data3,-1) #  每一行的最大值
    # __, ind2 = torch.max(mdata,-1) # 行号
    ind1 = torch.max(mdata,-1)[1] # 行号
    mdata, __ = torch.max(data3,-2) # 每一列的最大值
    ind2 = torch.max(mdata,-1)[1] # 列号

    batchnum = data3.size(0)

    # a = data3[indbatch,ind1,ind2 ]
    dm = data3.size(-2) # 行数
    dn = data3.size(-1)# 列数
    dx = ind1-(dm*torch.ones(batchnum))//2
    dy = ind2-(dn*torch.ones(batchnum))//2
    # 求最小重合面积 x 为行号 y为列号
    dxmax = dx.abs().max().int().item()
    dymax = dy.abs().max().int().item()

    xlen = (dm-dxmax)/2
    ylen = (dn-dymax)/2

    xm = (dm-1)/2 # 图像中心坐标
    ym = (dn-1)/2
    xm1b = xm + dx/2
    ym1b = ym + dy/2
    xm2b = xm - dx/2
    ym2b = ym - dy/2
    mnum = dm - dxmax
    nnum = dn - dymax

    return xlen, ylen, xm1b, ym1b, xm2b, ym2b,mnum, nnum
# ---------------------------------------------------------------------------------------
if __name__ == '__main__' :
    # set_seed()

    realimgpath = "slicey.csv"
    realimg = pd.read_csv(realimgpath, header=None, index_col=False)
    realimg = realimg.fillna(0).values
    realimg = realimg[np.newaxis, :, :]
    realimg = torch.as_tensor(realimg).view(1, 54, 54)
    # realimg = realimg[:, :53, :53 ].view(1, 53, 53)

    imshow(realimg)

    # batch_size 1 54 54
    img1 = realimg[:, 0:48, :].view(1, 1, 48, 54)
    img2 = realimg[:, 2:50, :].view(1, 1, 48, 54)

    imshow(img1)
    imshow(img2)


    ref = torch.cat((img1, img1, img2), dim=0)
    img = torch.cat((img1, img2, img1), dim=0)
    # ref = img1
    # img = img2
#------------------------------------------------------------------------------------
    xlen, ylen, xm1b, ym1b, xm2b, ym2b,mnum, nnum = imgregist(ref,img)
    #xm1b ym1b xm2b ym2b xlen ylen
    batchnum = ref.size(0)
    refreg = torch.empty((batchnum, 1, mnum, nnum))
    imgreg = torch.empty((batchnum, 1, mnum, nnum))
    for indbatch in range(batchnum):
        xm1 = xm1b[indbatch]
        ym1 = ym1b[indbatch]
        xs1 =  torch.ceil(xm1-xlen).int()
        xe1 = torch.floor(xm1+xlen-0.1).int()
        ys1 = torch.ceil(ym1-ylen).int()
        ye1 = torch.floor(ym1+ylen-0.1).int()

        xm2 = xm2b[indbatch]
        ym2 = ym2b[indbatch]
        xs2 =  torch.ceil(xm2-xlen).int()
        xe2 = torch.floor(xm2+xlen-0.1).int()
        ys2 = torch.ceil(ym2-ylen).int()
        ye2 = torch.floor(ym2+ylen-0.1).int()


        refreg[indbatch, :, :, :] = ref[indbatch,:, xs1:(xe1+1), ys1:(ye1+1)]
        imgreg[indbatch, :, :, :] = img[indbatch,:, xs2:(xe2+1), ys2:(ye2+1)]

    print((refreg-imgreg).max())

    imshow(refreg[0, :, :, :]-imgreg[0, :, :, :])
    imshow(refreg[1, :, :, :]-imgreg[1, :, :, :])
    imshow(refreg[2, :, :, :]-imgreg[2, :, :, :])
    # data1 = data1[indbatch,:, torch.ceil(xm1-xlen):torch.floor(xm1+xlen),  ]



    a = 0


