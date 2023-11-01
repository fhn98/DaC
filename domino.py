import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split

class DominoeMnistCifarDataset(Dataset):
    def __init__(self, data_type, spuriousity):

        assert data_type in ['train', 'val', 'test'], print("Error! no data_type found")
        assert not data_type in ['val', 'test'] or spuriousity == 0.5, print("Error! val and test must have spuriousity=0.5")
        self.spuriousity = spuriousity
        self.data_type = data_type

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        mnist_train_raw = torchvision.datasets.MNIST('./data/mnist/', train=True, download=True, transform=transform)
        cifar_train_raw = torchvision.datasets.CIFAR10('./data/cifar10/', train=True, download=True, transform=transform)
        mnist_train, mnist_valid = random_split(mnist_train_raw, [0.80, 0.20], generator=torch.Generator().manual_seed(42))
        cifar_train, cifar_valid = random_split(cifar_train_raw, [0.80, 0.20], generator=torch.Generator().manual_seed(42))

        mnist_test = torchvision.datasets.MNIST('./data/mnist/', train=False, download=True, transform=transform)
        cifar_test = torchvision.datasets.CIFAR10('./data/FashionMNIST/', train=False, download=True, transform=transform)

        mnist_dataset = None
        cifar_dataset = None
        if data_type == 'train':
            mnist_dataset = mnist_train
            cifar_dataset = cifar_train
        elif data_type == 'val':
            mnist_dataset = mnist_valid
            cifar_dataset = cifar_valid
            spuriousity = 0.5
        elif data_type == 'test':
            mnist_dataset = mnist_test
            cifar_dataset = cifar_test


        x, y, g = make_spurious_dataset(mnist_dataset, cifar_dataset, spuriousity)

        self.x = x
        # self.y = y
        # self.g = g
        self.y = nn.functional.one_hot(y.long(), num_classes=2).type(torch.FloatTensor)
        self.g = nn.functional.one_hot(g.long())


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.g[idx]


def get_MCDomino_dataset(phase, spuriousity):
    dataset = DominoeMnistCifarDataset(data_type=phase, spuriousity=spuriousity)
    return dataset

def get_domino_loaders(path, spuriosity = 90, batch_size = 32):
  train_set = torch.load(path+'train_'+str(spuriosity)+'.pt')
  val_set = torch.load(path+'val_'+str(spuriosity)+'.pt')
  test_set = torch.load(path+'test_'+str(spuriosity)+'.pt')

  trainloader =  DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
  valloader =  DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last = False, num_workers=4)
  testloader =  DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last = False, num_workers=4)

  return trainloader, valloader, testloader
