import torch
import numpy as np
import os
import random
import argparse
from models import ResNet50
from data.utils import get_loaders
from pytorch_grad_cam import XGradCAM
from tqdm import tqdm
import torch.nn as nn
from kneed import KneeLocator
from multiprocessing import Pool

def get_quantile_masks(heat_map, probs):
    masks = []
    quantiles = np.vsplit(np.quantile(heat_map, probs, axis=(1,2)), len(probs))
    for quantile in quantiles:
        quantile = quantile.reshape(quantile.shape[1], 1,1)
        masks.append(mask_heatmap_using_threshold(heat_map, quantile))
    return masks

def mask_heatmap_using_threshold(heat_maps, k):
  ret = heat_maps >= k
  return np.expand_dims(ret, 1)


def save_masks (losses, path, heat_maps, save_dir):
    range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    kneedle = KneeLocator(range, losses, online=True, S=1.0, curve="convex", direction="increasing")
    t = kneedle.elbow
    if t == None:
        t = 0.8
    final_mask = mask_heatmap_using_threshold(heat_maps, np.quantile(heat_maps, t))

    np.save(os.path.join(save_dir, str(path) + '.npy'), final_mask)


def main(args):
    device = torch.device('cuda')

    base_model = ResNet50().to(device)
    base_model.load_state_dict(torch.load(args.model_path))

    trainloader, valloader, testloader = get_loaders(dataset = args.dataset, path=args.data_path, batch_size=args.batch_size, get_mask = False, get_names = True)

    save_dir = os.path.join(args.data_path, f'masks_seed{args.seed}/')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    base_target_layers = [base_model.model.layer4[-1]]

    heat_map_generator = XGradCAM(
        model=base_model,
        target_layers=base_target_layers,
    )

    for loader in [trainloader, valloader, testloader]:
        for (batch, (image, label,_, path)) in enumerate(tqdm(loader)):
            pool = Pool()
            image = image.to(device)
            label = label.to(device)
            heat_maps = heat_map_generator(image)

            criterion = nn.CrossEntropyLoss(reduction='none')
            range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            masks = get_quantile_masks(heat_maps, range)

            losses = []

            for mask in masks:
                masked = image * torch.Tensor(mask).to(device)

                loss = criterion(base_model(masked), label)
                losses.append(loss.unsqueeze(1).detach().cpu().numpy())

            losses = np.concatenate(losses, axis = 1)

            losses = np.vsplit(losses, losses.shape[0])
            losses = [list(x.squeeze()) for x in losses]

            pool.starmap(save_masks, zip(losses, path, heat_maps, [save_dir]*len(path)))
            pool.close()


if __name__ == "__main__":
    seed = 30
    parser = argparse.ArgumentParser()
    default_data_path = '/home/f_hosseini/data/metashifts/metashifts/MetaDatasetCatDog'
    default_model_path = '/home/f_hosseini/dfr-ckpts/metashift/metashift_erm_run1.pt'
    parser.add_argument("--data_path", type=str, default=default_data_path, help="data path")
    parser.add_argument("--model_path", type=str, default=default_model_path, help="erm model path")
    parser.add_argument("--dataset", type=str, default='MetaShift')
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--seed", type=int, default=10)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True



    main(args)
