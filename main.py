import torch
import numpy as np
import os
import random
import argparse
import copy
from models import ResNet50
from data import *
from test import test
from train import train_masked_low_loss
from utils import weight_init, compute_loss_quantiles


def get_loaders(dataset, path, batch_size=32, mask_path=None, get_mask=False, get_names=False):
    if dataset == 'WaterBirds':
        return get_waterbird_loaders(path=path, batch_size=batch_size, mask_path=mask_path, get_mask=get_mask,
                                     get_names=get_names)
    elif dataset == 'CelebA':
        return get_celeba_loaders(path=path, batch_size=batch_size, get_mask=get_mask, mask_path=mask_path,
                                  get_names=get_names)
    elif dataset == 'MetaShift':
        return get_metashift_loaders(batch_size=batch_size, path=path, mask_path=mask_path, get_mask=get_mask,
                                     get_names=get_names)
    elif dataset == 'Domino':
        return get_domino_loaders(path=path, batch_size=batch_size, mask_path=mask_path, get_mask=get_mask,
                                  get_names=get_names)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Getting Dataloaders...')
    trainloader, valloader, testloader = get_loaders(args.dataset, path=args.data_path, mask_path=args.mask_path,
                                                     batch_size=args.batch_size, get_mask=True)

    print('Dataloaders prepared')
    model = ResNet50().to(device)
    model.load_state_dict(torch.load(args.model_path))

    base_model = ResNet50().to(device)
    base_model.load_state_dict(torch.load(args.model_path))

    model.eval()
    test(testloader, model, args)

    t = compute_loss_quantiles(trainloader, base_model, args.quantile)
    print('loss threshold', t)

    for n, p in model.named_parameters():
        p.requires_grad = False

    weight_init(model.model.fc)

    for p in model.model.fc.parameters():
        p.requires_grad = True

    trainable_parameters = model.model.fc.parameters()

    model_optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=args.step_size, gamma=args.gamma)

    ######################
    # learning
    ######################
    global_step = 0

    for p in base_model.parameters():
        p.requires_grad = True

    best_worst = 0
    best_cnn = None
    best_avg = 0

    for epoch in range(args.num_epochs):
        print("=========================")
        print("epoch:", epoch)
        print("=========================")
        model.train()
        base_model.eval()

        global_step = train_masked_low_loss(trainloader, model, base_model, model_optimizer,
                                            scheduler, global_step, t=t, args=args)

        # dev
        print('acc on val ....')
        avg_acc, envs_acc = test(valloader, model, args)
        if min(envs_acc) > best_worst:
            best_worst = min(envs_acc)
            best_cnn = copy.deepcopy(model)
            best_avg = avg_acc

        elif (min(envs_acc) == best_worst and avg_acc > best_avg):
            best_worst = min(envs_acc)
            best_cnn = copy.deepcopy(model)
            best_avg = avg_acc

        print('acc on test ....')
        test(testloader, model, args)

    model.load_state_dict(best_cnn.state_dict())
    model.eval()

    print('best model acc on val:')
    test(valloader, model, args)

    print('best model acc on test:')
    avg_acc, envs_acc = test(testloader, model, args)

    torch.save(model.state_dict(), args.save_path + f'alpha{args.alpha}_lt{args.quantile}_bs{args.batch_size}.model')


if __name__ == "__main__":
    seed = 10
    parser = argparse.ArgumentParser()
    default_data_path = '/home/f_hosseini/data/metashifts/metashifts/MetaDatasetCatDog'
    default_mask_path = '/home/f_hosseini/data/metashift_masks/'
    default_model_path = '/home/f_hosseini/dfr-ckpts/metashift/metashift_erm_run1.pt'
    default_save_path = '/home/f_hosseini/results/dominoes/'
    parser.add_argument("--data_path", type=str, default=default_data_path, help="data path")
    parser.add_argument("--mask_path", type=str, default=default_mask_path, help="data path")
    parser.add_argument("--model_path", type=str, default=default_model_path, help="pretrained model path")
    parser.add_argument("--save_path", type=str, default=default_save_path, help="path to save checkpoints")
    parser.add_argument("--dataset", type=str, default='MetaShift')
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_test_envs", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--step_size", type=float, default=5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=6)
    parser.add_argument("--quantile", type=float, default=0.8)
    parser.add_argument("--invert_mask", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=10)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    main(args)
