import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import pdb

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import os
import pdb
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms





class Subset(torch.utils.data.Dataset):
    """
    Subsets a dataset while preserving original indexing.
    NOTE: torch.utils.dataset.Subset loses original indexing.
    Args:
      dataset (Dataset): Dataset to extract the subset
      indices (np.array): Indices to select

    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        self.group_array = self.get_group_array(re_evaluate=True)
        self.label_array = self.get_label_array(re_evaluate=True)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def get_group_array(self, re_evaluate=True):
        """Return an array [g_x1, g_x2, ...]"""
        # setting re_evaluate=False helps us over-write the group array if necessary (2-group DRO)
        if re_evaluate:
            group_array = self.dataset.get_group_array()[self.indices]
            assert len(group_array) == len(
                self.indices), f"length of self.group_array:{len(group_array)}, length of indices:{len(self.indices)}"
            assert len(self.indices) == len(self)
            assert len(group_array) == len(self)
            return group_array
        else:
            return self.group_array

    def get_label_array(self, re_evaluate=True):
        if re_evaluate:
            label_array = self.dataset.get_label_array()[self.indices]
            assert len(label_array) == len(self)
            return label_array
        else:
            return self.label_array



def get_transform_cub(model_type, train, augment_data):
    scale = 256.0/224.0
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    target_resolution = (224,224)
    assert target_resolution is not None
    if (not train) or (not augment_data):
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0] * scale),
                               int(target_resolution[1] * scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            normalize
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
            normalize
        ])
    return transform


class ConfounderDataset(Dataset):
    """
    General confounder dataset where confounders are made explicit

    Args:
        root_dir (str): Root dir that stores raw data
        target_name (str): Data label
        confounder_names (list): A list of confounders
        model_type (str, optional): Type of model on the dataset, see models.py
        augment_data (bool, optional): Whether to use data augmentation, e.g., RandomCrop

    """
    def __init__(self, root_dir,
                 target_name, confounder_names,
                 model_type=None, augment_data=None):
        raise NotImplementedError

    def __len__(self):
        return len(self.group_array)

    def __getitem__(self, idx):
        g = self.group_array[idx]
        y = self.y_array[idx]

        if self.precomputed:
            x = self.features_mat[idx]
            if not self.pretransformed:
                if self.split_array[idx] == 0:
                    x = self.train_transform(x)
                else:
                    x = self.eval_transform(x)
            assert not isinstance(x, list)
        else:
            if not self.mix_array[idx]:
                x = self.get_image(idx)
            else:
                idx_1, idx_2 = self.mix_idx_array[idx]
                x1, x2 = self.get_image(idx_1), self.get_image(idx_2)
                l = self.mix_weight_array[idx]
                x = l * x1 + (1-l) * x2

        if self.mix_up:
            y_onehot = self.y_array_onehot[idx]
            try:
                true_g = self.domains[idx]
            except:
                true_g = None
            if true_g is None:
                return x, y, g, y_onehot, idx
            else:
                return x, y, true_g, y_onehot, idx
        else:
            return x, y, g, idx

    def refine_dataset(self):
        for name, split_id in self.split_dict.items():
            idxes = np.where(self.split_array == split_id)
            group_counts = (torch.arange(self.n_groups).unsqueeze(1)==torch.tensor(self.group_array[idxes])).sum(1).float()
            unique_group_id = torch.where(group_counts > 0)[0]
            group_dict = {id: new_id for new_id, id in enumerate(unique_group_id.tolist())}
            self.group_array[idxes] = np.array([group_dict[id] for id in self.group_array[idxes]])

    def get_image(self, idx):
        img_filename = self.filename_array[idx]
        img = Image.open(img_filename)
        if self.RGB:
            img = img.convert("RGB")
        # Figure out split and transform accordingly
        if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
            img = self.train_transform(img)
        elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and
              self.eval_transform):
            img = self.eval_transform(img)


        return img

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            mask = self.split_array == self.split_dict[split]
            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if train_frac<1 and split == 'train':
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, indices)
        return subsets

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes)
        c = group_idx % (self.n_groups//self.n_classes)
        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name

class DRODataset(Dataset):
    """
    General DRO dataset which enables reweighting groups

    Args:
        dataset (torch.utils.data)
        process_item_fn (func or `None`): Preprocess each image
        n_groups (int): Number of groups
        n_classes (int): Number of classes
        group_str_fn (str): Defines the group str given group id

    """
    def __init__(self, dataset, process_item_fn, n_groups, n_classes, group_str_fn):
        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.group_str = group_str_fn

        self._group_array = torch.LongTensor(dataset.get_group_array())
        self._y_array = torch.Tensor(dataset.get_label_array())
        self._group_counts = (torch.arange(self.n_groups).unsqueeze(1)==self._group_array).sum(1).float()
        self._group_counts = self._group_counts[np.where(self._group_counts > 0)]
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(1)==self._y_array).sum(1).float()
        self.group_indices = {loc.item():torch.nonzero(self._group_array == loc).squeeze(-1)
                               for loc in self._group_array.unique()}
        self.distinct_groups = np.unique(self._group_array)

        assert len(self._group_array) == len(self.dataset)

    def get_sample(self, g, idx, cross=False):
        g = g.item()
        if cross:
            g = np.random.choice(np.setdiff1d(self.distinct_groups, [g]))
        new_idx = np.random.choice(self.group_indices[g].numpy())
        return self.dataset[new_idx]

    def __getitem__(self, idx):
        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

    def get_loader(self, train, reweight_groups=None, **kwargs):
        if not train: # Validation or testing
            assert reweight_groups is None
            shuffle = False
            sampler = None
        elif not reweight_groups: # Training but not reweighting
            shuffle = True
            sampler = None
        else:
            # Training and reweighting
            # When the --robust flag is not set, reweighting changes the loss function
            # from the normal ERM (average loss over each training example)
            # to a reweighted ERM (weighted average where each (y,c) group has equal weight) .
            # When the --robust flag is set, reweighting does not change the loss function
            # since the minibatch is only used for mean gradient estimation for each group separately
            group_weights = len(self)/self._group_counts
            weights = group_weights[self._group_array]
            assert not np.isnan(weights).any()

            # Replacement needs to be set to True, otherwise we'll run out of minority samples
            sampler = WeightedRandomSampler(weights, len(self), replacement=True)
            shuffle = False

        loader = DataLoader(self, shuffle=shuffle, sampler=sampler, **kwargs)
        return loader

    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for sample in self:
            x = sample[0]
            return x.size()

    def get_group_array(self):
        if self.process_item is None:
            return self.dataset.get_group_array()
        else:
            raise NotImplementedError

    def get_label_array(self):
        if self.process_item is None:
            return self.dataset.get_label_array()
        else:
            raise NotImplementedError

class MetaDatasetCatDog(ConfounderDataset):
    """
    MetaShift data.
    `cat` is correlated with (`sofa`, `bed`), and `dog` is correlated with (`bench`, `bike`);
    In testing set, the backgrounds of both classes are `shelf`.

    Args:
        args : Arguments, see run_expt.py
        root_dir (str): Arguments, see run_expt.py
        target_name (str): Data label
        confounder_names (list): A list of confounders
        model_type (str, optional): Type of model on the dataset, see models.py
        augment_data (bool, optional): Whether to use data augmentation, e.g., RandomCrop
        mix_up (bool, optional): Whether to use mixup
        mix_alpha, mix_unit, mix_type, mix_freq, mix_extent: Variables in LISA implemenation
        group_id (int, optional): Select a subset of dataset with the group id

    """
    def __init__(self, root_dir,
                 target_name, confounder_names,
                 model_type=None,
                 augment_data=False,
                 mix_up=False,
                 mix_alpha=2,
                 mix_unit='group',
                 mix_type=1,
                 mix_freq='batch',
                 mix_extent=None,
                 group_id=None):
        self.mix_up = mix_up
        self.mix_alpha = mix_alpha
        self.mix_unit = mix_unit
        self.mix_type = mix_type
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data
        self.RGB = True
        self.n_confounders = 1

        self.train_data_dir = os.path.join(self.root_dir, "train")
        self.test_data_dir = os.path.join(self.root_dir, 'test')

        # Set training and testing environments
        self.n_classes = 2
        self.n_groups = 4
        cat_dict = {0: ["sofa"], 1: ["bed"]}
        dog_dict = {0: ['bench'], 1: ['bike']}
        self.test_groups = { "cat": ["shelf"], "dog": ["shelf"]}
        self.train_groups = {"cat": cat_dict, "dog": dog_dict}
        self.train_filename_array, self.train_group_array, self.train_y_array = self.get_data(self.train_groups,
                                                                                              is_training=True)
        self.test_filename_array, self.test_group_array, self.test_y_array = self.get_data(self.test_groups,
                                                                                           is_training=False)

        # split test and validation set
        np.random.seed(100)
        test_idxes = np.arange(len(self.test_group_array))
        val_idxes, _ = train_test_split(np.arange(len(test_idxes)), test_size=0.85, random_state=0)
        test_idxes = np.setdiff1d(test_idxes, val_idxes)

        # define the split array
        self.train_split_array = np.zeros(len(self.train_group_array))
        self.test_split_array = 2 * np.ones(len(self.test_group_array))
        self.test_split_array[val_idxes] = 1

        self.filename_array = np.concatenate([self.train_filename_array, self.test_filename_array])
        self.group_array = np.concatenate([self.train_group_array, self.test_group_array])
        self.split_array = np.concatenate([self.train_split_array, self.test_split_array])
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        self.y_array = np.concatenate([self.train_y_array, self.test_y_array])
        self.y_array_onehot = torch.zeros(len(self.y_array), self.n_classes)
        self.y_array_onehot = self.y_array_onehot.scatter_(1, torch.tensor(self.y_array).unsqueeze(1), 1).numpy()
        self.mix_array = [False] * len(self.y_array)

        if group_id is not None:
            idxes = np.where(self.group_array == group_id)
            self.filename_array = self.filename_array[idxes]
            self.group_array = self.group_array[idxes]
            self.split_array = self.split_array[idxes]
            self.y_array = self.y_array[idxes]
            self.y_array_onehot = self.y_array_onehot[idxes]

        self.precomputed = False
        self.train_transform = get_transform_cub(
            self.model_type,
            train=True,
            augment_data=augment_data)
        self.eval_transform = get_transform_cub(
            self.model_type,
            train=False,
            augment_data=augment_data)

        self.domains = self.group_array
        self.n_groups = len(np.unique(self.group_array))

    def get_group_array(self):
        return self.group_array

    def get_label_array(self):
        return self.y_array

    def group_str(self, group_idx, train=False):
        if not train:
            if group_idx < len(self.test_groups['cat']):
                group_name = f'y = cat'
                group_name += f", attr = {self.test_groups['cat'][group_idx]}"
            else:
                group_name = f"y = dog"
                group_name += f", attr = {self.test_groups['dog'][group_idx - len(self.test_groups['cat'])]}"
        else:
            if group_idx < len(self.train_groups['cat']):
                group_name = f'y = cat'
                group_name += f", attr = {self.train_groups['cat'][group_idx][0]}"
            else:
                group_name = f"y = dog"
                group_name += f", attr = {self.train_groups['dog'][group_idx - len(self.train_groups['cat'])][0]}"
        return group_name

    def get_data(self, groups, is_training):
        filenames = []
        group_ids = []
        ys = []
        id_count = 0
        animal_count = 0
        for animal in groups.keys():
            if is_training:
                for _, group_animal_data in groups[animal].items():
                    for group in group_animal_data:
                        for file in os.listdir(f"{self.train_data_dir}/{animal}/{animal}({group})"):
                            filenames.append(os.path.join(f"{self.train_data_dir}/{animal}/{animal}({group})", file))
                            group_ids.append(id_count)
                            ys.append(animal_count)
                    id_count += 1
            else:
                for group in groups[animal]:
                    for file in os.listdir(f"{self.test_data_dir}/{animal}/{animal}({group})"):
                        filenames.append(os.path.join(f"{self.test_data_dir}/{animal}/{animal}({group})", file))
                        group_ids.append(id_count)
                        ys.append(animal_count)
                    id_count += 1
            animal_count += 1
        return filenames, np.array(group_ids), np.array(ys)

def prepare_confounder_data(train, path, return_full_dataset=False):
    """
    Data preparation
    """
    full_dataset = MetaDatasetCatDog(
        root_dir=path,
        target_name='cat',
        confounder_names='background',
        model_type='resnet50',
        augment_data=False,
        mix_up= False
        )

    splits = ['train', 'val', 'test'] if train else ['test']
    subsets = full_dataset.get_splits(splits, 1.0)
    dro_subsets = []
    for split in splits:
        n_groups = full_dataset.n_groups
        dro_subsets.append(
            DRODataset(
                subsets[split],
                process_item_fn=None,
                n_groups=n_groups,
                n_classes=full_dataset.n_classes,
                group_str_fn=full_dataset.group_str
                ))
    return dro_subsets


def get_metashift_loaders(batch_size, path):
    train_data, val_data, test_data = prepare_confounder_data(train=True, path = path)

    loader_kwargs = {'batch_size': batch_size, 'num_workers': 4, 'pin_memory': False}
    train_loader = train_data.get_loader(reweight_groups=False,
                                                 train=True, **loader_kwargs)

    test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    val_loader = val_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)

    return train_loader, val_loader, test_loader