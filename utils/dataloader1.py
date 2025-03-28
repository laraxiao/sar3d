
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from utils.SARNNSimDataset import SARimgelevazimDataset




def dataloader(realpath1, realpath2, img_size,batch_size, device):

    Dataset1 = SARimgelevazimDataset(realpath1, img_size, device=device)
    Dataset2 = SARimgelevazimDataset(realpath2, img_size, device=device)

    # train_size1 = int(0.8 * len(Dataset1))
    # val_size1 = len(Dataset1) - train_size1
    # train_size2 = int(0.2 * len(Dataset2))
    # val_size2 = len(Dataset2) - train_size2

    train_size1 = int(1 * len(Dataset1))
    val_size1 = len(Dataset1) - train_size1
    train_size2 = int(0* len(Dataset2))
    val_size2 = len(Dataset2) - train_size2

    train_dataset1, val_dataset1 = torch.utils.data.random_split(Dataset1, [train_size1, val_size1], generator=torch.Generator(device=device))
    train_dataset2, val_dataset2 = torch.utils.data.random_split(Dataset2, [train_size2, val_size2], generator=torch.Generator(device=device))
    # -----------------------------------------------------------------------------------
    # 对上面2个数据集进行合并

    train_dataset = ConcatDataset((train_dataset1, train_dataset2))
    val_dataset = ConcatDataset((val_dataset1, val_dataset2))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))

    return train_loader, val_loader
