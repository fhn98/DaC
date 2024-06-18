import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import torchvision.transforms as transforms


class Dominoes (Dataset):
    def __init__(self, split, path, mask_path = None, get_mask = False, get_names = False, transform=None):
        self.get_mask = get_mask
        self.get_names = get_names
        self.split = split

        self.X = torch.tensor(np.load(os.path.join(path,f'X_{split}.npy')))
        self.y = F.one_hot(torch.tensor(np.load(os.path.join(path, f'y_{split}.npy'))).type(torch.LongTensor), 2).type(torch.FloatTensor)
        self.envs = F.one_hot(torch.tensor(np.load(os.path.join(path, f'env_{split}.npy'))).type(torch.LongTensor), 4)

        self.get_mask = get_mask
        self.get_names = get_names
        self.mask_path = mask_path
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if not self.transform==None:
            x = self.transform(self.X[idx])
        else:
            x = self.X[idx]
        ret = [x, self.y[idx], self.envs[idx]]

        if self.get_mask:
            mask = np.load(os.path.join(self.mask_path, f'{idx}.npy'))
            ret.append(mask)

        if self.get_names:
            ret.append(str(idx))

        return tuple(ret)


def get_domino_loaders(path, batch_size = 32, mask_path = None, get_mask = False, get_names = False):
    trainset = Dominoes('train', path = path, mask_path=mask_path, get_mask = get_mask, get_names=get_names)
    valset = Dominoes('val', path=path, mask_path=mask_path, get_mask=get_mask, get_names=get_names)
    testset = Dominoes('test', path=path, mask_path=mask_path, get_mask=get_mask, get_names=get_names)


    trainloader = DataLoader(trainset, shuffle=True, num_workers=4,
                           batch_size=batch_size)
    valloader = DataLoader(valset, shuffle=False, num_workers=4,
                         drop_last=False, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=False, num_workers=4,
                          drop_last=False, batch_size=batch_size)

    return trainloader, valloader, testloader
