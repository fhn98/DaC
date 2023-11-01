import torch
import torch.nn as nn
from einops import rearrange
def cal_sparsity(z):
    sparsity = torch.mean(torch.sum(z, dim=-1))
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
  heat_maps = torch.tensor(heat_maps).to(device)
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


