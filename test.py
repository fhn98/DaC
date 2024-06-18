import torch
from tqdm import tqdm

def test(dataset, model, args, device = torch.device('cuda')):
    """
    Conventional testing of a classifier.
    """

    count = 0

    corrects_envs = [0] * args.num_test_envs
    totals_envs = [0] * args.num_test_envs
    avg_acc_envs = [0] * args.num_test_envs

    model.eval()
    for (batch, (inputs, labels, envs, _)) in enumerate(tqdm(dataset)):
        count += 1

        inputs = inputs.to(device)
        labels = labels.to(device)
        envs = envs.to(device)

        logits = model(inputs)

        for env_num in range(args.num_test_envs):
            logits_env = logits[envs[:, env_num] == 1]
            labels_env = labels[envs[:, env_num] == 1]
            corrects_envs[env_num] += torch.sum(
                torch.argmax(logits_env, dim=1) == torch.argmax(labels_env, dim=1)).item()
            totals_envs[env_num] += len(logits_env)

    all_correct = 0
    all_totals = 0
    for env_num in range(args.num_test_envs):
        avg_acc_envs[env_num] = round(corrects_envs[env_num] / totals_envs[env_num], args.num_test_envs)
        print(f"env {env_num}, acc: {avg_acc_envs[env_num]}")
        all_correct += corrects_envs[env_num]
        all_totals += totals_envs[env_num]
    avg_inv_acc = round(all_correct / all_totals, 6)

    print(f"all envs mean acc: {avg_inv_acc}")

    return avg_inv_acc, avg_acc_envs