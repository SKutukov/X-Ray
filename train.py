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
from save_model import save_checkpoint
import json

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

learning_rate = 1e-3
size = 512
   
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
        is_cuda = data["is_cuda"]
        check_point_period = data["check_point_period"]
        checkpoint_file = "checkpoint.pth.tar"



    img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(train_directory, loader=cv2_loader, transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    start_epoch = 0
    if(is_cuda):
        model = autoencoder().cuda()
    else:    
        model = autoencoder()
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

    #===========load from checkpoint ==============
    if os.path.isfile(checkpoint_file):
            print("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            start_epoch = int(checkpoint['epoch'])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_file, checkpoint['epoch']))
    else:
            print("=> no checkpoint found at '{}'".format(checkpoint_file))



    for epoch in range(start_epoch, num_epochs):
        for data in dataloader:
            img, _ = data
            if(is_cuda):
                img = Variable(img).cuda()
            else:    
                img = Variable(img)
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        on_epoch_end(epoch, num_epochs, loss, output, size)        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, check_point_period, checkpoint_file )    
