__author__ = 'SKutukov'
import cv2
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable

import json

import matplotlib.pyplot as plt

from model_128 import netD, netG
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import statistics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--resume', action='store_true', help='is resume previouse work')


    opt = parser.parse_args()
    print(opt)
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    size = int(opt.imageSize)
    nc = 1
    batch_size = int(opt.batchSize)

    def cv2_loader(filename):
        if(nc==1):
            im = cv2.imread(filename, 0)
        elif (nc==3):    
            im = cv2.imread(filename)
        im = cv2.resize(im,(size, size),interpolation = cv2.INTER_AREA)
        im = im.reshape(size, size, nc)
        return im

    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(opt.dataroot, loader=cv2_loader, transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG = netG(ngpu, nz, ngf, nc)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = netD(ngpu, nc, ndf)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))  
    print(netD)

    criterion = nn.BCELoss()

    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        criterion.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    fixed_noise = Variable(fixed_noise)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    
    epoch_begin = 1
    epochs = []
    Loss_D = []
    Loss_G = []
    D_x_list = []
    D_G_z1_list = [] 
    D_G_z2_list = []

    config_name = "dcgan/config.json"
    if opt.resume:
        from initLoadModel import loadFromCheckpoint
        Loss_D, Loss_G, D_x_list, D_G_z1_list, D_G_z2_list, epochs,  epoch_begin = loadFromCheckpoint(config_name, opt.outf)

    with  open(config_name):
        data = json.load(open(config_name))
        Loss_D_file = os.path.join(opt.outf,data["Loss_D"])
        Loss_G_file = os.path.join(opt.outf,data["Loss_G"])
        D_x_list_file = os.path.join(opt.outf,data["D_x_list"])
        D_G_z1_list_file = os.path.join(opt.outf,data["D_G_z1_list"])
        D_G_z2_list_file = os.path.join(opt.outf,data["D_G_z2_list"])

        
    for epoch in range(epoch_begin , opt.niter):
        Loss_D_stat = []
        Loss_G_stat = []
        D_x_stat = []
        D_G_z1_list_stat = []
        D_G_z2_list_stat = []

        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            if opt.cuda:
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            label.resize_(batch_size).fill_(real_label)
            inputv = Variable(input)
            labelv = Variable(label)

            output = netD(inputv)
            errD_real = criterion(output, labelv)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            labelv = Variable(label.fill_(fake_label))
            output = netD(fake.detach())
            errD_fake = criterion(output, labelv)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, labelv)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

            if i % 100 == 0:
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % opt.outf,
                        normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.data,
                        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)
            
            Loss_D_stat.append(errD.data[0])
            Loss_G_stat.append(errG.data[0])
            D_x_stat.append(D_x)
            D_G_z1_list_stat.append(D_G_z1)
            D_G_z2_list_stat.append(D_G_z2)

        print('[Mean / Dist] Loss_D : [%.4f / %.4f ] Loss_G : [%.4f / %.4f ] D(x): [%.4f / %.4f ] D(G(z)): [%.4f / %.4f ] / [%.4f / %.4f ]'
                % ( statistics.mean(Loss_D_stat), statistics.variance(Loss_D_stat),
                    statistics.mean(Loss_G_stat), statistics.variance(Loss_G_stat),
                    statistics.mean(D_x_stat), statistics.variance(D_x_stat),
                    statistics.mean(D_G_z1_list_stat), statistics.variance(D_G_z1_list_stat),
                    statistics.mean(D_G_z2_list_stat), statistics.variance(D_G_z2_list_stat)
                ))

        Loss_D.append(statistics.mean(Loss_D_stat))
        Loss_G.append(statistics.mean(Loss_G_stat))
        D_x_list.append(statistics.mean(D_x_stat))
        D_G_z1_list.append(statistics.mean(D_G_z1_list_stat))
        D_G_z2_list.append(statistics.mean(D_G_z2_list_stat))
        epochs.append(epoch)
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
       
        def write_list(filename, data):
            thefile = open(filename, 'a')
            thefile.write("%s\n" % data)

        write_list(Loss_D_file,statistics.mean(Loss_D_stat))
        write_list(Loss_G_file,statistics.mean(Loss_G_stat))
        write_list(D_x_list_file,statistics.mean(D_x_stat))
        write_list(D_G_z1_list_file, statistics.mean(D_G_z1_list_stat))
        write_list(D_G_z2_list_file, statistics.mean(D_G_z2_list_stat))
