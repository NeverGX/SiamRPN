import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import argparse
import numpy as np
import os
import cv2
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config
from siamrpn import SiamRPN
from dataset import GOT_10KDataset
from custom_transforms import Normalize, ToTensor, RandomStretch,RandomCrop, CenterCrop, RandomBlur, ColorAug
from utils import generate_anchors
from loss import rpn_smoothL1, rpn_cross_entropy_balance
import sys
import math

torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir',type=str, default='/home/wangkh/Downloads/got-10k/crop_train_data', help='got_10k train dir')
parser.add_argument('--val_data_dir',type=str, default='/home/wangkh/Downloads/got-10k/crop_val_data', help='got_10k val dir')
parser.add_argument('--pretrain_model_dir',type=str, default='./models/pretrain/CIResNet22_PRETRAIN.model', help='got_10k val dir')
arg = parser.parse_args()
train_data_dir = arg.train_data_dir
val_data_dir = arg.val_data_dir
pretrain_model_dir = arg.pretrain_model_dir

def main():


    train_z_transforms = transforms.Compose([
        # RandomStretch(),
        # CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        # RandomStretch(),
        # RandomCrop((config.instance_size, config.instance_size),
        #            config.max_translate),
        # ColorAug(config.color_ratio),
        ToTensor()
    ])
    val_z_transforms = transforms.Compose([
        # CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor()
    ])
    val_x_transforms = transforms.Compose([
        ToTensor()
    ])

    score_size = int((config.instance_size - config.exemplar_size) / config.total_stride + 1)

    anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
                                    config.anchor_ratios,
                                    score_size)
    # create dataset
    train_dataset = GOT_10KDataset(train_data_dir, train_z_transforms, train_x_transforms, anchors)
    valid_dataset = GOT_10KDataset(val_data_dir, val_z_transforms, val_x_transforms, anchors)

    trainloader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                             shuffle=True, pin_memory=True, num_workers=config.train_num_workers, drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size=config.valid_batch_size,
                             shuffle=False, pin_memory=True, num_workers=config.valid_num_workers, drop_last=True)
    # create summary writer
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    summary_writer = SummaryWriter(config.log_dir)

    # start training
    with torch.cuda.device(config.gpu_id):
        model = SiamRPN()
        model.load_pretrain(pretrain_model_dir)
        model.freeze_layers()
        model = model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                    momentum=config.momentum, weight_decay=config.weight_decay)
        # schdeuler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

        scheduler = np.logspace(math.log10(config.lr), math.log10(config.end_lr), config.epoch)


        for epoch in range(config.epoch):
            train_loss = []
            model.train()
            curlr = scheduler[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = curlr
            for i, data in enumerate(tqdm(trainloader)):
                z, x, reg_label, cls_label = data
                z, x = Variable(z.cuda()), Variable(x.cuda())
                reg_label, cls_label = Variable(reg_label.cuda()), Variable(cls_label.cuda())
                pred_cls, pred_reg = model(z, x)
                optimizer.zero_grad()
                # permute
                pred_cls = pred_cls.reshape(-1, 1, config.anchor_num * score_size * score_size).permute(0,2,1)
                pred_reg = pred_reg.reshape(-1, 4, config.anchor_num * score_size * score_size).permute(0,2,1)
                cls_loss = rpn_cross_entropy_balance(pred_cls, cls_label, config.num_pos, config.num_neg)
                reg_loss = rpn_smoothL1(pred_reg, reg_label, cls_label, config.num_pos)
                loss = cls_loss + config.lamb * reg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
                optimizer.step()
                step = epoch * len(trainloader) + i
                summary_writer.add_scalar('train/loss', loss.data, step)
                train_loss.append(loss.data.cpu().numpy())
            train_loss = np.mean(train_loss)
            valid_loss = []
            model.eval()
            for i, data in enumerate(tqdm(validloader)):
                z, x, reg_label, cls_label = data
                z, x = Variable(z.cuda()), Variable(x.cuda())
                reg_label, cls_label = Variable(reg_label.cuda()), Variable(cls_label.cuda())
                pred_cls, pred_reg = model(z, x)
                pred_cls = pred_cls.reshape(-1, 1, config.anchor_num * score_size * score_size).permute(0, 2, 1)
                pred_reg = pred_reg.reshape(-1, 4, config.anchor_num * score_size * score_size).permute(0, 2, 1)
                cls_loss = rpn_cross_entropy_balance(pred_cls, cls_label, config.num_pos, config.num_neg)
                reg_loss = rpn_smoothL1(pred_reg, reg_label, cls_label, config.num_pos)
                loss = cls_loss + config.lamb * reg_loss
                valid_loss.append(loss.data.cpu().numpy())
            valid_loss = np.mean(valid_loss)
            print("EPOCH %d valid_loss: %.4f, train_loss: %.4f, learning_rate: %.4f" %
                  (epoch, valid_loss, train_loss, optimizer.param_groups[0]["lr"]))
            summary_writer.add_scalar('valid/loss',
                                      valid_loss, epoch + 1)
            torch.save(model.cpu().state_dict(),
                       "./models/siamrpn_{}.pth".format(epoch + 1))
            model.cuda()
            # schdeuler.step()

if __name__ == '__main__':
    sys.exit(main())


