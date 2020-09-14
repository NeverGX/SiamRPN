import os
import torch
import torchvision
from torch import nn
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from config import config
from networks import CLF
from torch.optim.lr_scheduler import StepLR
import json


num_epochs = 100
batch_size = 64
learning_rate = 1e-2
train_dir = '/home/wangkh/ae_responses_data'


model = CLF().cuda()
model.load_state_dict(torch.load('./models/clf_60.pth'))
e = torch.rand(1,17 * 17)
o = model(e.cuda())
o = torch.sigmoid(o)
print(o)




#
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-6)
# criterion = nn.BCEWithLogitsLoss()
# scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
#
# for epoch in range(num_epochs):
#
#     loss_train = []
#     for data in tqdm(trainloader):
#         img, label = data
#         img = Variable(img).cuda()
#         label = Variable(label).cuda()
#         # ===================forward=====================
#         output = model(img)
#         loss = criterion(output, label)
#         # ===================backward====================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # print(loss)
#         loss_train.append(loss.data.cpu().numpy())
#     scheduler.step()
#     # ===================log========================
#     print('epoch [{}/{}], loss:{:.4f}, lr:{}'.format(epoch + 1, num_epochs, np.mean(loss_train), scheduler.get_lr()))
#     if epoch % 10 == 0:
#         e = torch.rand(1,17 * 17)
#         o = model(e.cuda())
#         o = torch.sigmoid(o)
#         print(o)
#         torch.save(model.state_dict(), './models/oneclass_clf_{}.pth'.format(epoch))
