import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from tqdm import tqdm
import numpy as np
from data import *

def get_optimizer(parameters, learning_rate, optimizer_name, weight_decay=0):
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError("Invalid optimizer type. Supported options are 'Adam', 'AdamW', and 'SGD'.")

        return optimizer

def get_scheduler(optimizer, args):
    if args.scheduler == 'none':
        return None

    if args.scheduler == 'StepLr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    else:
        raise ValueError("Invalid scheduler type. Supported options are 'none', 'StepLr'.")

    return scheduler

def cal_sparsity(z):
    sparsity = torch.sum(torch.sum(torch.sum(z, dim=-1), -1))/(torch.numel(z[0]))
    return sparsity

def weight_init(m):
    """
    Initialize the weights of a given module.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)

def mask_heatmap_using_threshold_patched(heat_maps, k, patch_size, h, w, device = torch.device('cuda')):
  output_shape = (heat_maps.shape[0], h // patch_size, w // patch_size)
  heat_maps = heat_maps.reshape(output_shape[0], output_shape[1], patch_size, output_shape[2], patch_size)
  heat_maps = heat_maps.mean((2, 4))
  heat_maps = heat_maps.reshape([heat_maps.shape[0],-1])
  indices = torch.topk(heat_maps, k=k).indices
  mask = torch.zeros([heat_maps.shape[0], heat_maps.shape[1]]).to(device)
  mask.scatter_(1, indices, 1.)
  mask = rearrange(mask, 'b (h w c) -> b h w c', h=h//patch_size, c=1)

  temp = rearrange(torch.ones([heat_maps.shape[0], h, w]).to(device), 'b (h1 h2) (w1 w2) -> b h1 w1 (h2 w2)', w2=patch_size, h2=patch_size) ### w2 and h2 are patch_size

  selected = temp*mask

  rationale = rearrange(selected, 'b h1 w1 (h2 w2) -> b (h1 h2) (w1 w2) ', w2=patch_size, h2=patch_size) ### w2 and h2 are patch_size

  return rationale



def mask_heatmap_using_threshold(heat_maps, k, h, w, device = torch.device('cuda')):
  heat_maps = torch.tensor(heat_maps).to(device)
  heat_maps = heat_maps.reshape([heat_maps.shape[0],-1])
  indices = torch.topk(heat_maps, k=k).indices
  mask = torch.zeros([heat_maps.shape[0], heat_maps.shape[1]]).to(device)
  mask.scatter_(1, indices, 1.)
  ret = rearrange(mask, 'b (h w) -> b h w', w = w, h = h)
  return ret


def compute_loss_quantiles(dataset, model, quantile):
    """
    Conventional testing of a classifier.
    """
    avg_inv_acc = 0
    count = 0

    model.eval()

    criterion = nn.CrossEntropyLoss(reduction = 'none')

    device = torch.device('cuda')

    all_losses = []
    for (batch, (inputs, labels, _, _)) in enumerate(tqdm(dataset)):
        count+=1

        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs)


        losses = criterion(logits, labels)
        all_losses.append(losses.detach().cpu())


    all_losses = torch.cat(all_losses).numpy()
    return np.quantile(all_losses, quantile)

