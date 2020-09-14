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
train_dir = '/home/wangkh/responses_data'


class dataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.videos = os.listdir(data_dir)
        self.num = 67166
    def __getitem__(self, item):
        index = item % len(self.videos)
        video = self.videos[index]
        video_path = os.path.join(self.data_dir, video)
        data_id = np.random.choice(len(os.listdir(video_path)))
        data_path = os.path.join(video_path,"{:0>8d}.json".format(data_id + 1))
        with open(data_path, 'r') as load_f:
            h_dict = json.load(load_f)
        h = np.asarray(h_dict["response_map"]).astype(np.float32)
        if h_dict["label"]==1:
            label = np.asarray([1]).astype(np.float32)
        else:
            label = np.asarray([0]).astype(np.float32)

        return h, label

    def __len__(self):

        return self.num

train_dataset = dataset(train_dir)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = CLF().cuda()
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-6)
criterion = nn.BCEWithLogitsLoss()
# scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
for epoch in range(num_epochs):

    loss_train = []
    for data in tqdm(trainloader):
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, label)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
        loss_train.append(loss.data.cpu().numpy())
    # scheduler.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}, lr:{}'.format(epoch + 1, num_epochs, np.mean(loss_train),optimizer.param_groups[0]['lr']))
    if epoch % 10 == 0:
        torch.save(model.state_dict(), './models/clf_{}.pth'.format(epoch))
