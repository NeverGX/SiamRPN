import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from config import config

eps = 1e-5

def center_crop(x):
    """
    center crop layer. crop [1:-2] to eliminate padding influence.
    Crop 1 element around the tensor
    input x can be a Variable or Tensor, 4]
    """
    return x[:, :, 1:-1, 1:-1].contiguous()


def center_crop7(x):
    """
    Center crop layer for stage1 of resnet. (7*7)
    input x can be a Variable or Tensor
    """

    return x[:, :, 2:-2, 2:-2].contiguous()

class Bottleneck_CI(nn.Module):
    """
    Bottleneck with center crop layer, utilized in CVPR2019 model
    """
    expansion = 4

    def __init__(self, inplanes, planes, last_relu, stride=1, downsample=None, dilation=1):
        super(Bottleneck_CI, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        padding = 1
        if abs(dilation - 2) < eps: padding = 2
        if abs(dilation - 3) < eps: padding = 3

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.last_relu:         # remove relu for the last block
            out = self.relu(out)

        out = center_crop(out)     # in-residual crop

        return out
class ResNet(nn.Module):
    """
    ResNet with 22 layer utilized in CVPR2019 paper.
    Usage: ResNet(Bottleneck_CI, [3, 4], [True, False], [False, True], 64, [64, 128])
    """

    def __init__(self, block, layers, last_relus, s2p_flags, firstchannels=64, channels=[64, 128], dilation=1):
        self.inplanes = firstchannels
        self.stage_len = len(layers)
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, firstchannels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(firstchannels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # stage2
        if s2p_flags[0]:
            self.layer1 = self._make_layer(block, channels[0], layers[0], stride2pool=True, last_relu=last_relus[0])
        else:
            self.layer1 = self._make_layer(block, channels[0], layers[0], last_relu=last_relus[0])

        # stage3
        if s2p_flags[1]:
            self.layer2 = self._make_layer(block, channels[1], layers[1], stride2pool=True, last_relu=last_relus[1], dilation=dilation)
        else:
            self.layer2 = self._make_layer(block, channels[1], layers[1], last_relu=last_relus[1], dilation=dilation)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, last_relu, stride=1, stride2pool=False, dilation=1):
        """
        :param block:
        :param planes:
        :param blocks:
        :param stride:
        :param stride2pool: translate (3,2) conv to (3, 1)conv + (2, 2)pool
        :return:
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, last_relu=True, stride=stride, downsample=downsample, dilation=dilation))
        if stride2pool:
            layers.append(self.maxpool)
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == blocks - 1:
                layers.append(block(self.inplanes, planes, last_relu=last_relu, dilation=dilation))
            else:
                layers.append(block(self.inplanes, planes, last_relu=True, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)     # stride = 2
        x = self.bn1(x)
        x = self.relu(x)
        x = center_crop7(x)
        x = self.maxpool(x)   # stride = 4

        x = self.layer1(x)
        x = self.layer2(x)    # stride = 8

        return x

class SiamRPN(nn.Module):

    def __init__(self):
        super(SiamRPN, self).__init__()
        self.features = ResNet(Bottleneck_CI, [3, 4], [True, False], [False, True])
        self.anchor_nums = config.anchor_num
        self.template_cls = nn.Conv2d(512, 256 * self.anchor_nums * 1, kernel_size=3)
        self.template_reg = nn.Conv2d(512, 256 * self.anchor_nums * 4, kernel_size=3)
        self.search_cls = nn.Conv2d(512, 256, kernel_size=3)
        self.search_reg = nn.Conv2d(512, 256, kernel_size=3)
        self.adjust = nn.Conv2d(self.anchor_nums * 4, self.anchor_nums * 4, kernel_size=1)

    def forward(self, z, x):

        # Trainging
        if z is not None and x is not None:
            batch = z.shape[0]
            feature_z = self.features(z)
            feature_x = self.features(x)
            z_cls = self.template_cls(feature_z)
            z_reg = self.template_reg(feature_z)
            x_cls = self.search_cls(feature_x)
            x_reg = self.search_reg(feature_x)
            """
            z_cls: (batch,2*anchor_num*256,3,3)
            z_reg: (batch,4*anchor_num*256,3,3)
            x_cls: (batch,256,19,19)
            x_reg: (batch,256,19,19)
            """
            z_cls = z_cls.reshape(-1, 256, z_cls.shape[2], z_cls.shape[3])
            z_reg = z_reg.reshape(-1, 256, z_reg.shape[2], z_reg.shape[3])
            x_cls = x_cls.reshape(1, -1, x_cls.shape[2], x_cls.shape[3])
            x_reg = x_reg.reshape(1, -1, x_reg.shape[2], x_reg.shape[3])
            pred_cls = F.conv2d(x_cls, z_cls, groups=batch)
            pred_reg = F.conv2d(x_reg, z_reg, groups=batch)
            """
            pred_cls: (1,1*anchor_num*batch,17,17)
            pred_reg: (1,4*anchor_num*batch,17,17)
            """
            pred_cls = pred_cls.reshape(batch,-1,pred_cls.shape[2],pred_cls.shape[3])
            pred_reg = self.adjust(pred_reg.reshape(batch,-1,pred_reg.shape[2],pred_reg.shape[3]))
            """
            pred_cls: (batch,1*anchor_num,17,17)
            pred_reg: (batch,4*anchor_num,17,17) 
            """
            return pred_cls, pred_reg
        # Tracking
        elif z is None and x is not None:
            batch = x.shape[0]
            feature_x = self.features(x)
            x_cls = self.search_cls(feature_x)
            x_reg = self.search_reg(feature_x)
            x_cls = x_cls.reshape(1, -1, x_cls.shape[2], x_cls.shape[3])
            x_reg = x_reg.reshape(1, -1, x_reg.shape[2], x_reg.shape[3])
            pred_cls = F.conv2d(x_cls, self.z_cls, groups=batch)
            pred_reg = F.conv2d(x_reg, self.z_reg, groups=batch)
            pred_cls = pred_cls.reshape(batch, -1, pred_cls.shape[2], pred_cls.shape[3])
            pred_reg = self.adjust(pred_reg.reshape(batch, -1, pred_reg.shape[2], pred_reg.shape[3]))
            return pred_cls, pred_reg
        # Initing
        else:
            self.feature_z = self.features(z)
            z_cls = self.template_cls(self.feature_z)
            z_reg = self.template_reg(self.feature_z)
            self.z_cls = z_cls.reshape(-1, 256, z_cls.shape[2], z_cls.shape[3])
            self.z_reg = z_reg.reshape(-1, 256, z_reg.shape[2], z_reg.shape[3])

    def update_model(self, z):

        new_feature_z = self.features(z)
        new_z_cls = self.template_cls(new_feature_z)
        new_z_cls = new_z_cls.reshape(-1, 256, new_z_cls.shape[2], new_z_cls.shape[3])
        self.z_cls.data = (1-config.z_lr)*self.z_cls.data+config.z_lr*new_z_cls

    def load_pretrain(self, pretrained_path):

        print('load pretrained model from {}'.format(pretrained_path))
        pretrained_dict = torch.load(pretrained_path)
        pretrained_dict = {k.split('.', 1)[1]: v for k, v in pretrained_dict.items()}
        self.load_state_dict(pretrained_dict, strict=False)

    def load_model(self, model_path):

        print('load model from {}'.format(model_path))
        model_dict = torch.load(model_path)

        # for k, v in model_dict.items():
        #     print(k,v.shape)
        # for k, v in self.state_dict().items():
        #     print(k, v.shape)

        if model_path.split('/')[-1] == 'CIResNet22_RPN.pth':
            model_dict = {k.split('.', 1)[1]: v for k, v in model_dict.items()}
        self.load_state_dict(model_dict)

    def freeze_layers(self):

        if torch.cuda.device_count() > 1:
            for param in self.features.module.conv1.parameters():
                param.requires_grad = False
        else:
            for param in self.features.conv1.parameters():
                param.requires_grad = False
        print('conv1 weight frozen')









