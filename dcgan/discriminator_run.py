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



from model import netD, netG
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--outTXT', required=True, help='path to txt')


    opt = parser.parse_args()
    print(opt)

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

    netD = netD(ngpu, nc, ndf)
    
    netD.load_state_dict(torch.load(opt.netD))  
    # print(netD)

    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0
    batchSize = 1
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    label = torch.FloatTensor(batchSize)

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

    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    dataset = ImageFolder(opt.dataroot, loader=cv2_loader, transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    import statistics 
    D_x_list = []
    D_x_list_file = opt.outTXT
    from numpy import array
    for i, data in enumerate(dataloader, 0):
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        input.resize_as_(real_cpu).copy_(real_cpu)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        D_x = output.data.mean()

        print(D_x)
        D_x_list.append(D_x)
        def write_list(filename, data):
            thefile = open(filename, 'a')
            thefile.write("%s\n" % data)        
        write_list(D_x_list_file,D_x)

    print('[Mean / Dist] Loss_D : [%.4f / %.4f ]'
                % ( statistics.mean(D_x_list), statistics.variance(D_x_list)                   
                ))

                
 
       
    
    # def discr(filename):
    #     real_cpu = cv2_loader(filename)
    #     # print(real_cpu.shape)
    #     real_cpu = array(real_cpu).reshape(1, 1,opt.imageSize,opt.imageSize)
    #     real_cpu = torch.from_numpy(real_cpu)
    #     real_cpu = real_cpu.float()

    #     # print(real_cpu)

    #     batch_size = real_cpu.size(0)
    #     inputv = Variable(real_cpu)
    #     output = netD(inputv)
    
    #     D_x = output.data.mean()
    #     print("result ", D_x)

    # discr("/home/skutukov96/cloud/Splited_Normal_ChestXray/Ready/295/00025364_000.png")
    # discr("/home/skutukov96/cloud/Splited_Normal_ChestXray/Ready/295/00016278_000.png")
    # discr("/home/skutukov96/cloud/Splited_Normal_ChestXray/Ready/65/00022499_000.png")
    # discr("/home/skutukov96/cloud/Splited_Normal_ChestXray/Ready/208/00027089_001.png")

    # discr("/home/skutukov96/cloud/066112810103357.dcm.jpg")
    # discr("/home/skutukov96/cloud/034110909092422.dcm.jpg")
    # noise = torch.FloatTensor(batchSize, 1, opt.imageSize, opt.imageSize)


    # noise.resize_(batchSize, 1, opt.imageSize, opt.imageSize).normal_(0, 1)
    # noisev = Variable(noise)
    # output = netD(noisev)
    # D_x = output.data.mean()
    # print("result ", D_x)

            # if i % 100 == 0:
            #     vutils.save_image(real_cpu,
            #             '%s/real_samples.png' % opt.outf,
            #             normalize=True)
            #     fake = netG(fixed_noise)
            #     vutils.save_image(fake.data,
            #             '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
            #             normalize=True)
