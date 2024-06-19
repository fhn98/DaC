import torch
import torch.nn as nn
from tqdm import tqdm
from utils import cal_sparsity
import numpy as np

def train_masked_low_loss(dataset, cnn_image_encoder, base_model, opt, scheduler, step, t, args, device = torch.device('cuda')):
    criterion = nn.CrossEntropyLoss()

    ### obtain the optimizer
    opt = opt

    ### average loss
    avg_inv_acc = 0
    avg_inv_loss = 0
    sparsity = 0
    count = 0

    selected_count = 0
    total_count = 0


    for (batch, (inputs, labels,_, mask)) in enumerate(tqdm(dataset)):
        count +=1

        inputs = inputs.to(device)
        labels = labels.to(device)
        mask = mask.permute(0,2,1,3).type(torch.IntTensor).to(device)

        if args.invert_mask:
            mask = 1-mask

        opt.zero_grad()

        batch_mean = torch.mean(inputs, dim = (0,2,3)).reshape([1,3,1,1])

        non_reduction_criterion = nn.CrossEntropyLoss(reduction = 'none')
        with torch.no_grad():
            tmp_logits = base_model (inputs)
            tmp_loss = non_reduction_criterion (tmp_logits, labels)

            selected_inputs = inputs[tmp_loss<=t]
            selected_labels = labels[tmp_loss<=t]
            selected_mask = mask[tmp_loss <=t]


        total_count += inputs.shape[0]
        selected_count += selected_inputs.shape[0]


        if selected_inputs.shape[0]>0:
            masked = selected_inputs*selected_mask+(1-selected_mask)*batch_mean
            splits = []
            label_splits = []
            splits_masks = []
            for e in range(2):
                splits.append(selected_inputs[selected_labels[:,e]==1].clone().detach().to(device))
                label_splits.append(selected_labels[selected_labels[:,e]==1].clone().detach().to(device))
                splits_masks.append(torch.clone(selected_mask[selected_labels[:,e]==1]).to(device))

            if splits[1].shape[0] > 0 and splits[0].shape[0] > 0:
                samples2 = np.random.choice(splits[1].shape[0], splits[0].shape[0])
                samples1 = np.random.choice (splits[0].shape[0], splits[1].shape[0])
                fusion1 = splits[0]*splits_masks[0] + (1-splits_masks[0])*(1-splits_masks[1][samples2].detach())*splits[1][samples2] + (1-splits_masks[0]) * splits_masks[1][samples2] * batch_mean
                fusion2 = splits[1]*splits_masks[1] + (1-splits_masks[1])*(1-splits_masks[0][samples1].detach())*splits[0][samples1] + (1-splits_masks[1]) * splits_masks[0][samples1] * batch_mean

                total_inputs = torch.cat([inputs, masked, fusion1, fusion2], dim=0)
                total_labels = torch.cat([labels, selected_labels,  label_splits[0],  label_splits[1]])



            else:
                p = torch.randperm(selected_inputs.shape[0])
                permuted_inputs = torch.clone(selected_inputs).detach()[p].to(device)
                permuted_mask = torch.clone(selected_mask).detach()[p].to(device)

                fusion = selected_inputs*selected_mask + (1-selected_mask)*(1-permuted_mask)*permuted_inputs + (1-selected_mask)*permuted_mask*batch_mean

                total_inputs = torch.cat((inputs, masked, fusion), dim=0)
                total_labels = torch.cat((labels, selected_labels, selected_labels), dim=0)



            shape = inputs.shape[0]
            masked_shape = selected_inputs.shape[0]
            logits = cnn_image_encoder (total_inputs)

            total_loss = criterion (logits[:shape], total_labels[:shape]) + args.alpha * criterion (logits[shape:shape+masked_shape], total_labels[shape:shape+masked_shape]) + args.alpha * criterion (logits[shape+masked_shape:], total_labels[shape+masked_shape:])

            total_loss.backward()

            sparsity += cal_sparsity(selected_mask)

        else:
            logits = cnn_image_encoder(inputs)
            total_labels = labels
            total_loss = criterion (logits, total_labels)
            total_loss.backward()


        opt.step()

        avg_inv_loss += total_loss

        avg_inv_acc += torch.sum(torch.argmax(logits, dim=1)==torch.argmax(total_labels, dim=1))

    # results
    avg_inv_acc = avg_inv_acc/(total_count+selected_count*2)
    avg_inv_loss = avg_inv_loss/count
    sparsity = sparsity/(selected_count+0.00001)

    print (selected_count/total_count)

    print("{:s}{:d}: {:s}{:.4f}, {:s}{:.4f}, {:s}{:4f}.".format(
        "----> [Train] Total iteration #", step, "inv acc: ",
        avg_inv_acc, "inv loss: ", avg_inv_loss, "sparsity: ", sparsity),
          flush=True)

    if not scheduler==None:
      scheduler.step()

    return step+1


def train_erm(dataloader, model, opt, scheduler, step, device=torch.device('cuda')):
    criterion =  nn.CrossEntropyLoss()

    ### average loss
    avg_acc = 0
    avg_loss = 0
    count = 0

    model.train()
    for (batch, (inputs, labels, _, _)) in enumerate(tqdm(dataloader)):
        count += inputs.shape[0]

        inputs = inputs.to(device)
        labels = labels.to(device)

        opt.zero_grad()
        logits = model(inputs)
        total_loss = criterion(logits, labels.float())
        total_loss.backward()
        opt.step()

        avg_loss += total_loss
        avg_acc += torch.sum(torch.argmax(logits, dim=1)==torch.argmax(labels, dim=1))

    # results
    avg_acc = avg_acc/(count)
    avg_loss = avg_loss/(count)

    if not scheduler==None:
        scheduler.step()

    print("{:s}{:d}: {:s}{:.4f}, {:s}{:.4f}.".format(
        "----> [Train] Total iteration #", step, "acc: ",
        avg_acc, "loss: ", avg_loss), flush=True)

    return step+1