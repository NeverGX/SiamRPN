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
from networks import DAE


if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):

    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 17, 17)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3
train_dir = '/home/wangkh/responses_data'

class res_dataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.videos = os.listdir(data_dir)
        self.num = 64000   #64000
    def __getitem__(self, item):
        index = item % len(self.videos)
        video = self.videos[index]
        video_path = os.path.join(self.data_dir, video)
        h_id = np.random.choice(int(len(os.listdir(video_path))/2))
        h_path = os.path.join(video_path, "{:0>8d}.txt".format(h_id + 1))
        label_path = os.path.join(video_path, "{:0>8d}label.txt".format(h_id + 1))
        h = np.loadtxt(h_path, dtype=float, delimiter=',').astype(np.float32)
        label = np.loadtxt(label_path, dtype=float, delimiter=',').astype(np.float32)

        return h, label
    def __len__(self):
        return self.num


train_dataset = res_dataset(train_dir)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = DAE().cuda()
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
criterion = nn.MSELoss()


for epoch in range(num_epochs):

    loss_train = []
    for data in tqdm(trainloader):
        img, label = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        # ===================forward=====================
        output = model(img)
        output = torch.sigmoid(output)
        loss = F.smooth_l1_loss(output.view(-1, 1, 17, 17), label)
        # loss = criterion(output.view(-1, 1, 17, 17),label)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train.append(loss.data.cpu().numpy())
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, np.mean(loss_train)))
    if epoch % 10 == 0:

        # model.eval()
        # model.load_state_dict(torch.load('./models/autoencoder_20.pth'))
        e = torch.rand(1,17*17)
        o = model(e.cuda())
        o = torch.sigmoid(o)
        pic = to_img(o.cpu().data)
        save_image(pic, './mlp_img/image_ex_{}.png'.format(epoch))


        pic = to_img(output.cpu().data)
        save_image(pic, './mlp_img/image_{}.png'.format(epoch))
        pic = to_img(img.cpu().data)
        save_image(pic, './mlp_img/image_orgin{}.png'.format(epoch))

    torch.save(model.state_dict(), './models/autoencoder_{}.pth'.format(epoch))


