import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt

import cv2
import os




from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


latent_dim = 100 * 4
size = 128
input_dim = size * size
size_2 = size // 2
size_4 = size // 4
dim1 = (size_2 * size_2)
dim2 = (size_4 * size_4)

nc = 1
epochs = 200
cuda = True

class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, dim1)
        self.linear2 = torch.nn.Linear(dim1, dim2)
        self.linear2 = torch.nn.Linear(dim2, input_dim)


    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_dim, dim1)
        self.linear2 = torch.nn.Linear(dim1, dim2)
        self.linear2 = torch.nn.Linear(dim2, input_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))

class VAE(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(dim2, latent_dim)
        self._enc_log_sigma = torch.nn.Linear(dim2, latent_dim)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batchSize', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
    parser.add_argument('--train_dataroot', required=True, help='path to train dataset')
    parser.add_argument('--test_dataroot', required=True, help='path to test dataset')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()


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

    train = args.train_dataroot
    dataset = ImageFolder(train, loader=cv2_loader, transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=args.batchSize, shuffle=True) 

    test = args.test_dataroot
    test_dataset = ImageFolder(test, loader=cv2_loader, transform=img_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batchSize, shuffle=True)


    encoder = Encoder()
    decoder = Decoder()
    vae = VAE(encoder, decoder)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    l = None


    if not os.path.exists("results"):
        os.mkdir("results")


    loss = []
    def write_list(filename, data):
        thefile = open(filename, 'a')
        thefile.write("%s\n" % data)

    for epoch in range(epochs):
        train_loss = 0
        for i, data in enumerate(dataloader, 0):
            inputs, classes = data
            inputs, classes = Variable(inputs.resize_(args.batchSize, input_dim)), Variable(classes)
            optimizer.zero_grad()
            dec = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + ll
            loss.backward()
            optimizer.step()
            l = loss.data[0]
            train_loss += l
            if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(dataloader.dataset),
                100. * i / len(dataloader),
                l / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        write_list("loss.txt",  train_loss / len(train_loader.dataset))

    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                recon_batch.view(args.batchSize, 1, size, size)[:n]])
            save_image(comparison.data.cpu(),
                    'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))