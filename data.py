from celeba import get_celeba_loaders
from waterbirds import get_waterbird_loaders
from metashift import get_metashift_loaders
from domino import get_domino_loaders

def get_loader(dataset, path, batch_size= 32):
    if dataset == 'WaterBirds':
        return get_waterbird_loaders(path, batch_size)
    elif dataset == 'CelebA':
        return get_celeba_loaders(path, batch_size)
    elif dataset == 'MetaShift':
        return get_metashift_loaders(batch_size, path)
    elif dataset == 'Domino':
        return get_domino_loaders(path = path, spuriosity = 90, batch_size = batch_size)