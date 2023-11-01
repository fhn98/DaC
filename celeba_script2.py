import os
import subprocess
from itertools import product
from tqdm.auto import tqdm
import time


experiment_num = 0

num_of_patches = [15, 19]

seeds = [10, 30, 50]
alphas = [1, 2]
betas = [0, 1]
lrs = [0.001]
loss_thresholds = [0.00046035, 0.00143472, 0.00445709, 0.01214537]
for seed, num_patch, alpha, beta, loss_threshold, lr in tqdm(product(seeds, num_of_patches, alphas, betas, loss_thresholds, lrs)):
    path = f'logs/celeba/ps_32_k{num_patch}_alpha{alpha}_beta{beta}_lt{loss_threshold}_lr_{lr}_bs_64/seed_{seed}.txt'
    time.sleep(10)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/result.txt', 'w') as f:
        subprocess.run(['python3', 'main.py',\
                        f'--patch_size={32}',\
                        f'--k={num_patch}', \
                        f'--alpha={alpha}', \
                        f'--beta={beta}', \
                        f'--loss_threshold={loss_threshold}', \
                        f'--lr={lr}', \
                        f'--seed={seed}'],stdout=f)
    print(f"finish experiment {experiment_num} alhamdulillah")
    experiment_num += 1

