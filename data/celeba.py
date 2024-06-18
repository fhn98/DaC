import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
import random



class celebADataset(Dataset):
    def __init__(self, phase, root_dir, mask_path, transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), get_mask = False, get_names = False):
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        # (y, gender)
        self.env_dict = {
            (0, 0): torch.Tensor([1, 0, 0, 0]).type(torch.LongTensor),   # nonblond hair, female
            (0, 1): torch.Tensor([0, 1, 0, 0]).type(torch.LongTensor),   # nonblond hair, male
            (1, 0): torch.Tensor([0, 0, 1, 0]).type(torch.LongTensor),   # blond hair, female
            (1, 1): torch.Tensor([0, 0, 0, 1]).type(torch.LongTensor)    # blond hair, male
        }
        self.dataset_dir = root_dir
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'celeba_split.csv'))
        self.metadata_df = self.metadata_df[self.metadata_df['split']==self.split_dict[phase]]
        self.get_mask = get_mask
        self.get_names = get_names
        self.mask_path = mask_path

        if phase == 'train':
            label0 = self.metadata_df.index[self.metadata_df['Blond_Hair'] == 0].tolist()
            label1 = self.metadata_df.index[self.metadata_df['Blond_Hair'] == 1].tolist()
            l = int(min(len(label0), len(label1)))
            label0 = [label0[i]for i in random.sample(range(0, len(label0)), l)]
            label1 = [label1[i] for i in random.sample(range(0, len(label1)), l)]

            self.metadata_df = self.metadata_df.iloc[label0+label1]


        self.y_array = self.metadata_df['Blond_Hair'].values
        y_array = torch.Tensor(np.array(self.metadata_df['Blond_Hair'].values)).type(torch.LongTensor)
        self.y_one_hot = nn.functional.one_hot(y_array, num_classes=2).type(torch.FloatTensor)
        self.gender_array = self.metadata_df['Male'].values
        self.filename_array = self.metadata_df['image_id'].values
        self.transform = transform
        self.labels = torch.nn.functional.one_hot(torch.Tensor(self.y_array).type(torch.LongTensor)).type(torch.FloatTensor)

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        gender = self.gender_array[idx]
        img_filename = os.path.join(
            self.dataset_dir,
            'img_align_celeba',
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        env = self.env_dict[(y, gender)]

        ret = [img, label, env]

        if self.get_mask:
            mask_filename = os.path.join(
                self.mask_path,
                self.filename_array[idx] + '.npy')
            mask = np.load(mask_filename).astype(int)
            ret.append(np.squeeze(mask, 1))

        if self.get_names:
            ret.append(self.filename_array[idx])

        return tuple(ret)

    def get_images(self):
        print('Generating CelebA...')
        mean = (0, 0, 0)
        std = (1, 1, 1)
        trans = transforms.Compose([
                # transforms.Resize((64, 64)),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ])
        images = []
        for idx in tqdm(range(len(self.filename_array))):
            img_filename = os.path.join(
                self.dataset_dir,
                'img_align_celeba',
                self.filename_array[idx])
            img = Image.open(img_filename).convert('RGB')
            images.append(torch.unsqueeze(trans(img), 0))
        return torch.cat(images, dim=0)

    def get_envs(self):
        envs = []
        for idx in tqdm(range(len(self.filename_array))):
            y = self.y_array[idx]
            gender = self.gender_array[idx]
            env = self.env_dict[(y, gender)]
            envs.append(torch.unsqueeze(env, 0))
        return torch.cat(envs, dim=0)

    def get_raw_image(self,idx):
      scale = 256.0/224.0
      target_resolution = [224, 224]
      img_filename = os.path.join(
            self.dataset_dir,
            'img_align_celeba',
            self.filename_array[idx])
      img = Image.open(img_filename).convert('RGB')
      transform = transforms.Compose([
          transforms.Resize(
              (int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
          transforms.CenterCrop(target_resolution),
          transforms.CenterCrop(target_resolution),
          transforms.ToTensor(),
      ])
      return transform(img)



def get_dataset (phase, root_dir, mask_path, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))]), get_mask=False, get_names=False):
    dataset = celebADataset(phase=phase, root_dir=root_dir, mask_path = mask_path, transform=transform, get_mask=get_mask, get_names=get_names)
    print (len(dataset))
    return dataset

def get_loader (root_dir,mask_path, phase, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))]), batch_size = 32, get_mask = False, get_names = False):
    dataset = get_dataset(phase, root_dir, mask_path, transform, get_mask=get_mask, get_names=get_names)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)



def get_transform_celeba(train):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    if not train:
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return transform

def get_celeba_loaders(path, mask_path, batch_size, get_mask=False, get_names=False):
    transforms_train = get_transform_celeba(True)
    transforms_test = get_transform_celeba(False)

    trainloader = get_loader(path, mask_path, 'train', transform=transforms_train, batch_size=batch_size, get_mask=get_mask, get_names=get_names)
    valloader = get_loader(path, mask_path, 'val', transform=transforms_test, batch_size=batch_size, get_mask=get_mask, get_names=get_names)
    testloader = get_loader(path, mask_path, 'test', transform=transforms_test, batch_size=batch_size, get_mask=get_mask, get_names=get_names)

    return trainloader, valloader, testloader
