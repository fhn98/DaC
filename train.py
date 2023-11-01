import torch
import torch.nn as nn
from tqdm import tqdm
from pytorch_grad_cam import XGradCAM
from utils import mask_heatmap_using_threshold_patched, cal_sparsity
import numpy as np

def train_masked_low_loss(dataset, cnn_image_encoder, base_model, opt, scheduler, step, args, device = torch.device('cuda')):
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

    for (batch, (inputs, labels, _)) in enumerate(tqdm(dataset)):
        count +=1

        inputs = inputs.to(device)
        labels = labels.to(device)

        opt.zero_grad()

        batch_mean = torch.mean(inputs, dim = (0,2,3)).reshape([1,3,1,1])

        non_reduction_criterion = nn.CrossEntropyLoss(reduction = 'none')
        with torch.no_grad():
            tmp_logits = base_model (inputs)
            tmp_loss = non_reduction_criterion (tmp_logits, labels)

            selected_inputs = inputs[tmp_loss<args.loss_threshold]
            selected_labels = labels[tmp_loss<args.loss_threshold]


            # low_loss_inputs = inputs[tmp_loss<args['low_loss_threshold']]
            # low_loss_labels = labels[tmp_loss<args['low_loss_threshold']]


            # high_loss_inputs = inputs[tmp_loss>args['high_loss_threshold']]
            # high_loss_labels = labels[tmp_loss>args['high_loss_threshold']]

            # inputs = inputs[tmp_loss<=args['high_loss_threshold']]
            # labels = labels[tmp_loss<=args['high_loss_threshold']]


        # selected_inputs = torch.cat((low_loss_inputs, high_loss_inputs), 0)
        # selected_labels = torch.cat((low_loss_labels, high_loss_labels), 0)


        total_count += inputs.shape[0]
        selected_count += selected_inputs.shape[0]


        if selected_inputs.shape[0]>0:
            heat_map_generator = XGradCAM(
            model=base_model,
            target_layers=[base_model.model.layer4[-1]],
            use_cuda=True,
            )

            heat_maps = heat_map_generator(selected_inputs)

            # if low_loss_inputs.shape[0]>0 and high_loss_inputs.shape[0]>0:
            #   low_loss_rationale = torch.unsqueeze(mask_heatmap_using_threshold_patched(heat_maps[:low_loss_inputs.shape[0]], args['k'], args['patch_size']), 1)
            #   high_loss_rationale = 1 - torch.unsqueeze(mask_heatmap_using_threshold_patched(heat_maps[low_loss_inputs.shape[0]:], args['k'], args['patch_size']), 1)


            #   selected_rationale = torch.cat((low_loss_rationale, high_loss_rationale), 0)

            # else:

            # all_rationales = []
            # select = []
            # losses = []
            # criterion_samplewise = nn.CrossEntropyLoss(reduction = 'none')
            # for i, k in enumerate([5 ,10, 15, 20, 25]):
            #   heat_maps = heat_map_generator(selected_inputs)
            #   rationale = torch.unsqueeze(mask_heatmap_using_threshold_patched(heat_maps, k, args['patch_size']), 1)
            #   masked = selected_inputs*rationale + (1-rationale)*batch_mean
            #   all_rationales.append(rationale)
            #   with torch.no_grad():
            #     pred = cnn_image_encoder(masked)
            #     losses.append(criterion_samplewise(pred, selected_labels).unsqueeze(1))
            #
            #
            # min_loss = torch.min(torch.cat(losses , -1), -1)[0].reshape(-1,1)
            #
            # for i, k in enumerate([5 ,10, 15, 20, 25]):
            #   select.append((losses[i]==min_loss).to(device).detach().reshape((-1,1,1,1)))
            #
            #
            # selected_rationale = select[0]*all_rationales[0] + select[1]*all_rationales[1] + select[2]*all_rationales[2] + select[3]*all_rationales[3] + select[4]*all_rationales[4]
            selected_rationale = torch.unsqueeze(mask_heatmap_using_threshold_patched(heat_maps, args.k, args.patch_size, h = args.h, w = args.w), 1)
            masked = selected_inputs*selected_rationale + (1-selected_rationale)*batch_mean

            splits = []
            label_splits = []
            splits_rationales = []
            for e in range(2):
                splits.append(selected_inputs[selected_labels[:,e]==1].clone().detach().to(device))
                label_splits.append(selected_labels[selected_labels[:,e]==1].clone().detach().to(device))
                splits_rationales.append(torch.clone(selected_rationale[selected_labels[:,e]==1]).to(device))

            if splits[1].shape[0] > 0 and splits[0].shape[0] > 0:
                samples2 = np.random.choice(splits[1].shape[0], splits[0].shape[0])
                samples1 = np.random.choice (splits[0].shape[0], splits[1].shape[0])
                fusion1 = splits[0]*splits_rationales[0] + (1-splits_rationales[0])*(1-splits_rationales[1][samples2].detach())*splits[1][samples2] + (1-splits_rationales[0]) * splits_rationales[1][samples2] * batch_mean
                fusion2 = splits[1]*splits_rationales[1] + (1-splits_rationales[1])*(1-splits_rationales[0][samples1].detach())*splits[0][samples1] + (1-splits_rationales[1]) * splits_rationales[0][samples1] * batch_mean

                total_inputs = torch.cat([inputs, masked, fusion1, fusion2], dim=0)
                total_labels = torch.cat([labels, selected_labels, label_splits[0], label_splits[1]])



            else:
                p = torch.randperm(selected_inputs.shape[0])
                permuted_inputs = torch.clone(selected_inputs).detach()[p].to(device)
                permuted_rationale = torch.clone(selected_rationale).detach()[p].to(device)

                fusion = selected_inputs*selected_rationale + (1-selected_rationale)*(1-permuted_rationale)*permuted_inputs + (1-selected_rationale)*permuted_rationale*batch_mean

                total_inputs = torch.cat((inputs, masked, fusion), dim=0)
                total_labels = torch.cat((labels, selected_labels, selected_labels), dim=0)



            shape = inputs.shape[0]
            masked_shape = selected_inputs.shape[0]
            logits = cnn_image_encoder (total_inputs)

            total_loss = criterion (logits[:shape], total_labels[:shape]) + args.alpha * criterion (logits[shape:shape+masked_shape], total_labels[shape:shape+masked_shape]) + args.beta * criterion (logits[shape+masked_shape:], total_labels[shape+masked_shape:])

            total_loss.backward()

            sparsity += cal_sparsity(selected_rationale)

        else:
            logits = cnn_image_encoder(inputs)
            total_labels = labels
            total_loss = criterion (logits, total_labels)
            total_loss.backward()


        opt.step()

        avg_inv_loss += total_loss

        avg_inv_acc += torch.sum(torch.argmax(logits, dim=1)==torch.argmax(total_labels, dim=1))

    # results
    avg_inv_acc = avg_inv_acc/total_count
    avg_inv_loss = avg_inv_loss/count
    sparsity = sparsity/count

    print (selected_count/total_count)

    print("{:s}{:d}: {:s}{:.4f}, {:s}{:.4f}, {:s}{:4f}.".format(
        "----> [Train] Total iteration #", step, "inv acc: ",
        avg_inv_acc, "inv loss: ", avg_inv_loss, "sparsity: ", sparsity),
          flush=True)

    if not scheduler==None:
      scheduler.step()

    return step+1
