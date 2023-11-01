import torch
import numpy as np
import os
import random
import argparse
import copy
from model import ResNet50
from data import get_loader
from test import test
from train import train_masked_low_loss
from utils import weight_init


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    print('Getting Dataloaders...')
    trainloader, valloader, testloader = get_loader(args.dataset, path = args.data_path, batch_size = args.batch_size)
    print('Dataloaders prepared')

    ############################
    # making the rationalization model
    ############################
    cnn_image_encoder = ResNet50().to(device)

    cnn_image_encoder.load_state_dict(torch.load(args.model_path))


    base_model = ResNet50().to(device)

    base_model.load_state_dict(torch.load(args.model_path))

    cnn_image_encoder.eval()

    # test(testloader, cnn_image_encoder, args)

    for n, p in cnn_image_encoder.named_parameters():
        p.requires_grad = False

    weight_init(cnn_image_encoder.model.fc)

    for p in cnn_image_encoder.model.fc.parameters():
        p.requires_grad = True


    # trainable_parameters = masker.parameters()
    trainable_parameters = cnn_image_encoder.model.fc.parameters()

    model_optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=args.step_size, gamma=args.gamma)


    ######################
    # learning
    ######################
    global_step = 0

    for p in base_model.model.parameters():
        p.requires_grad = True


    best_worst = 0
    best_cnn = None

    for epoch in range(args.num_epochs):
        print("=========================")
        print("epoch:", epoch)
        print("=========================")
        cnn_image_encoder.train()

        # if epoch > 0:
        #   base_model.load_state_dict(cnn_image_encoder.state_dict())

        base_model.eval()
        global_step = train_masked_low_loss(trainloader, cnn_image_encoder, base_model, model_optimizer,
                                            scheduler, global_step, args = args)

        # dev
        print('acc on val ....')
        avg_acc, envs_acc = test(valloader, cnn_image_encoder, args)
        if min(envs_acc) >= best_worst:
            best_worst = min(envs_acc)
            best_cnn = copy.deepcopy(cnn_image_encoder)

        print('acc on test ....')
        test(testloader, cnn_image_encoder, args)

    cnn_image_encoder.load_state_dict(best_cnn.state_dict())
    cnn_image_encoder.eval()

    print('best model acc on val:')
    test(valloader, cnn_image_encoder, args)

    print('best model acc on test:')
    test(testloader, cnn_image_encoder, args)

    torch.save(cnn_image_encoder.state_dict(), args.checkpoint_path+f'ps{args.patch_size}_k{args.k}_sd{args.seed}_alpha{args.alpha}_beta{args.beta}_lt{args.loss_threshold}_bs{args.batch_size}.pt')




if __name__ == "__main__":
    seed = 30
    parser = argparse.ArgumentParser()
    default_data_path = '/home/user01/data/'
    default_model_path = '/home/user01/models/resnet50_celeba.model'
    default_ckpt_path = '/home/user01/results/celeba/'
    parser.add_argument("--data_path", type=str, default=default_data_path, help="data path")
    parser.add_argument("--model_path", type=str, default=default_model_path, help="pretrained model path")
    parser.add_argument("--checkpoint_path", type=str, default=default_ckpt_path, help="path to save checkpoints")
    parser.add_argument("--dataset", type=str, default='CelebA')
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_test_envs", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--step_size", type=float, default=5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--loss_threshold", type=float, default=0.01)
    parser.add_argument("--h", type=int, default=224)
    parser.add_argument("--w", type=int, default=224)
    parser.add_argument("--seed", type=int, default=10)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True



    main(args)
