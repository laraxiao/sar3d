
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from utils.SARNNSimDataset2 import SARimgelevazimDataset




def dataloader(realpath1, realpath2, img_size,batch_size, device):

    Dataset1 = SARimgelevazimDataset(realpath1, img_size, device=device)
    Dataset2 = SARimgelevazimDataset(realpath2, img_size, device=device)

    # -----------------------------------------------------------------------------------
    # 对上面2个数据集进行合并

    train_dataset = ConcatDataset((Dataset1, Dataset2))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    return train_loader
