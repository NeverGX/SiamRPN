import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np



def rpn_cross_entropy(input, target):
    r"""
    :param input: (15x15x5,2)
    :param target: (15x15x5,)
    :return:
    """
    mask_ignore = target == -1
    mask_calcu = 1 - mask_ignore
    loss = F.cross_entropy(input=input[mask_calcu], target=target[mask_calcu])
    return loss




def rpn_cross_entropy_balance(input, target, num_pos, num_neg):
    r"""
    :param input: (N,1445,1)
    :param target: (17x17x5,)
    :return:
    """
    # if ohem:
    #     final_loss = rpn_cross_entropy_balance_parallel(input, target, num_pos, num_neg, anchors, ohem=True,
    #                                                     num_threads=4)
    # else:
    loss_all = []
    for batch_id in range(target.shape[0]):
        min_pos = min(len(np.where(target[batch_id].cpu() == 1)[0]), num_pos)
        min_neg = int(min(len(np.where(target[batch_id].cpu() == 1)[0]) * num_neg / num_pos, num_neg))
        pos_index = np.where(target[batch_id].cpu() == 1)[0].tolist()
        neg_index = np.where(target[batch_id].cpu() == 0)[0].tolist()

        pos_index_random = random.sample(pos_index, min_pos)
        input = input.squeeze()
        if len(pos_index) > 0:
            pos_loss_bid_final = F.binary_cross_entropy_with_logits(input=input[batch_id][pos_index_random],
                                                                    target=target[batch_id][pos_index_random], reduction = 'none')
        else:
            pos_loss_bid_final = torch.FloatTensor([0]).cuda()

        if len(pos_index) > 0:
            neg_index_random = random.sample(np.where(target[batch_id].cpu() == 0)[0].tolist(), min_neg)
            neg_loss_bid_final = F.binary_cross_entropy_with_logits(input=input[batch_id][neg_index_random],
                                                 target=target[batch_id][neg_index_random], reduction='none')
        else:
            neg_index_random = random.sample(np.where(target[batch_id].cpu() == 0)[0].tolist(), num_neg)
            neg_loss_bid_final = F.binary_cross_entropy_with_logits(input=input[batch_id][neg_index_random],
                                                 target=target[batch_id][neg_index_random], reduction='none')

        loss_bid = (pos_loss_bid_final.mean() + neg_loss_bid_final.mean()) / 2
        loss_all.append(loss_bid)
    final_loss = torch.stack(loss_all).mean()
    return final_loss


def rpn_smoothL1(input, target, label, num_pos=16):
    r'''
    :param input: torch.Size([N, 1445, 4])
    :param target: torch.Size([N, 1445, 4])
            label: (torch.Size([N, 1445]) pos neg or ignore
    :return:
    '''
    loss_all = []
    for batch_id in range(target.shape[0]):
        min_pos = min(len(np.where(label[batch_id].cpu() == 1)[0]), num_pos)

        pos_index = np.where(label[batch_id].cpu() == 1)[0]
        pos_index = random.sample(pos_index.tolist(), min_pos)
        if len(pos_index) > 0:
            loss_bid = F.smooth_l1_loss(input[batch_id][pos_index], target[batch_id][pos_index])
        else:
            loss_bid = torch.FloatTensor([0]).cuda()[0]
        loss_all.append(loss_bid.mean())
    final_loss = torch.stack(loss_all).mean()
    return final_loss
