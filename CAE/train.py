__author__ = 'SKutukov'
import cv2
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import numpy as np

from autoencoder import autoencoder
from epoch_callback import on_epoch_end
from save_model import save_model
import json

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

learning_rate = 1e-3
size = 64
    
def cv2_loader(filename):
    im = cv2.imread(filename, 0)
    im = cv2.resize(im,(size,size),interpolation = cv2.INTER_AREA)
    im = im.reshape(size, size, 1)
    return im


if __name__ == '__main__':
        #geting params
    config_name = './config.json'
    with  open(config_name):
        data = json.load(open(config_name))
        train_directory = data["train_directory"]
        batch_size = data["batch_size"] 
        num_epochs = data["num_epochs"]

    img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(train_directory, loader=cv2_loader, transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    model = autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data
            img = Variable(img).cuda()
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        on_epoch_end(epoch, num_epochs, loss, output, size)        
        save_model(epoch, model)    
