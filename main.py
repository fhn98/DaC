import torch
import numpy as np
import os
import random
import argparse
import copy
from models import ResNet50
from test import test
from train import train_masked_low_loss, train_erm
from utils import *
from data.utils import get_loaders


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Getting Dataloaders...')
    trainloader, valloader, testloader = get_loaders(args.dataset, path=args.data_path, mask_path=args.mask_path,
                                                     batch_size=args.batch_size, get_mask=True)

    print('Dataloaders prepared')
    model = ResNet50().to(device)

    if not args.experiment == 'ERM':
        model.load_state_dict(torch.load(args.model_path))

        model.eval()
        print('acc of the ERM model on the test set:')
        test(testloader, model, args)

        base_model = ResNet50().to(device)
        base_model.load_state_dict(torch.load(args.model_path))

        t = compute_loss_quantiles(trainloader, base_model, args.quantile)
        print('loss threshold', t)

        for n, p in model.named_parameters():
            p.requires_grad = False

        weight_init(model.model.fc)

        for p in model.model.fc.parameters():
            p.requires_grad = True

    if args.experiment == 'ERM':
        trainable_parameters = model.parameters()

    else:
        trainable_parameters = model.model.fc.parameters()

    model_optimizer = get_optimizer(trainable_parameters, args.lr, args.optimizer, args.weight_decay)
    scheduler = get_scheduler(model_optimizer, args)

    ######################
    # learning
    ######################
    global_step = 0

    if not args.experiment == 'ERM':
        for p in base_model.parameters():
            p.requires_grad = True

    best_worst = 0
    best_model = None
    best_avg = 0

    for epoch in range(args.num_epochs):
        print("=========================")
        print("epoch:", epoch)
        print("=========================")
        model.train()

        if not args.experiment == 'ERM':
            base_model.eval()

            global_step = train_masked_low_loss(trainloader, model, base_model, model_optimizer,
                                                scheduler, global_step, t=t, args=args)

        else:
            global_step = train_erm(trainloader, model, model_optimizer, scheduler, global_step)

        # dev
        print('acc on val ....')
        avg_acc, envs_acc = test(valloader, model, args)
        if min(envs_acc) > best_worst:
            best_worst = min(envs_acc)
            best_model = copy.deepcopy(model)
            best_avg = avg_acc

        elif (min(envs_acc) == best_worst and avg_acc > best_avg):
            best_worst = min(envs_acc)
            best_model = copy.deepcopy(model)
            best_avg = avg_acc

        print('acc on test ....')
        test(testloader, model, args)

    model.load_state_dict(best_model.state_dict())
    model.eval()

    print('best model acc on val:')
    test(valloader, model, args)

    print('best model acc on test:')
    test(testloader, model, args)

    torch.save(model.state_dict(), args.save_path + f'alpha{args.alpha}_lt{args.quantile}_bs{args.batch_size}.model')


if __name__ == "__main__":
    seed = 10
    parser = argparse.ArgumentParser()
    default_mask_path = './data/masks/'
    default_model_path = './ckpts/'
    default_save_path = './'
    parser.add_argument("--data_path", type=str, help="data path")
    parser.add_argument("--mask_path", type=str, default=default_mask_path, help="mask path")
    parser.add_argument("--model_path", type=str, default=default_model_path, help="pretrained model path")
    parser.add_argument("--save_path", type=str, default=default_save_path, help="path to save checkpoints")
    parser.add_argument("--experiment", type=str, default='DaC', help="The experiment to run", choices=['ERM', 'DaC'])
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_test_envs", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--optimizer", type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument("--scheduler", type=str, default='none', choices=['none', 'StepLr'])
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
