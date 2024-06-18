import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import pandas as pd


class WaterbirdDataset(Dataset):
    def __init__(self, split, path, transform, mask_path = None, get_mask=False, get_names=False):
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        self.env_dict = {
            (0, 0): torch.Tensor(np.array([1,0,0,0])),
            (0, 1): torch.Tensor(np.array([0,1,0,0])),
            (1, 0): torch.Tensor(np.array([0,0,1,0])),
            (1, 1): torch.Tensor(np.array([0,0,0,1]))
        }
        self.split = split
        self.dataset_dir= path
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'metadata.csv'))
        self.metadata_df = self.metadata_df[self.metadata_df['split']==self.split_dict[self.split]]

        y_array = torch.Tensor(np.array(self.metadata_df['y'].values)).type(torch.LongTensor)
        print(y_array.shape)
        self.y_array = self.metadata_df['y'].values

        self.place_array = self.metadata_df['place'].values
        self.filename_array = self.metadata_df['img_filename'].values
        self.transform = transform
        self.get_mask = get_mask
        self.get_names = get_names
        self.mask_path = mask_path

        self.y_one_hot = nn.functional.one_hot(y_array, num_classes=2).type(torch.FloatTensor)
        print(self.y_one_hot.shape)
    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        place = self.place_array[idx]
        img_filename = os.path.join(
            self.dataset_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)

        label = self.y_one_hot[idx]

        ret = [img, label, self.env_dict[(y, place)]]

        if self.get_mask:
            mask_filename = os.path.join(
                self.mask_path,
                self.filename_array[idx].replace('/','_') )+ '.npy'
            mask = np.load(mask_filename)
            ret.append(mask)

        if self.get_names:
            name = self.filename_array[idx].replace('/','_')
            ret.append(name)

        return tuple(ret)

    def get_raw_image(self,idx):
      scale = 256.0/224.0
      target_resolution = [224, 224]
      img_filename = os.path.join(
            self.dataset_dir,
            self.filename_array[idx])
      img = Image.open(img_filename).convert('RGB')
      transform = transforms.Compose([
          transforms.Resize(
              (int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
          transforms.CenterCrop(target_resolution),
          transforms.ToTensor(),
      ])
      return transform(img)




def get_waterbird_dataloader(split, transform, path, batch_size, mask_path = None, get_mask = False, get_names = False):
    kwargs = {'pin_memory': True, 'num_workers': 2, 'drop_last': False}
    dataset = WaterbirdDataset( split=split, path = path, mask_path=mask_path, transform = transform, get_names = get_names, get_mask=get_mask)
    if not split == 'train':
      print (split)
      dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, **kwargs)
    else:
      dataloader = DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True, **kwargs)
    return dataloader


def get_waterbird_loaders(path, batch_size, mask_path=None, get_mask = False, get_names = False):
    t_train = get_transform_cub(True)
    t_tst = get_transform_cub(False)
    trainloader = get_waterbird_dataloader('train', t_tst, path, batch_size, mask_path = mask_path, get_mask = get_mask, get_names = get_names)
    valloader = get_waterbird_dataloader('val', t_tst, path, batch_size, mask_path = mask_path, get_mask = get_mask, get_names = get_names)
    testloader = get_waterbird_dataloader('test', t_tst, path, batch_size, mask_path = mask_path, get_mask = get_mask, get_names = get_names)

    return trainloader, valloader, testloader


def get_waterbird_dataset(split, path, transform, mask_path= None, get_mask = False, get_names = False):
    dataset = WaterbirdDataset(split=split, path = path, mask_path=mask_path, transform = transform, get_names = get_names, get_mask=get_mask)
    return dataset


def get_transform_cub(train):
    scale = 256.0/224.0
    target_resolution = [224, 224]
    assert target_resolution is not None

    if (not train):

      transform = transforms.Compose([
                transforms.Resize(
                    (int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                0.229, 0.224, 0.225]),
            ])

    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                            0.229, 0.224, 0.225]),
        ])

    return transform

