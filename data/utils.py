from .celeba import get_celeba_loaders
from .waterbirds import get_waterbird_loaders
from .metashift import get_metashift_loaders
from .domino import get_domino_loaders

def get_loaders(dataset, path, batch_size= 32, mask_path = None, get_mask = False, get_names = False):
    if dataset == 'WaterBirds':
        return get_waterbird_loaders(path = path, batch_size = batch_size, mask_path = mask_path, get_mask = get_mask, get_names = get_names)
    elif dataset == 'CelebA':
        return get_celeba_loaders(path = path, batch_size = batch_size, get_mask = get_mask, mask_path = mask_path, get_names = get_names)
    elif dataset == 'MetaShift':
        return get_metashift_loaders(batch_size = batch_size, path = path, mask_path = mask_path, get_mask = get_mask, get_names = get_names)
    elif dataset == 'Domino':
        return get_domino_loaders(path = path, batch_size = batch_size, mask_path = mask_path, get_mask = get_mask, get_names = get_names)
