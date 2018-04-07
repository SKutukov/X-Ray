from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import cv2
import os

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batchSize', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train_dataroot', required=True, help='path to train dataset')
parser.add_argument('--test_dataroot', required=True, help='path to test dataset')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


size = 64
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists("results"):
    os.mkdir("results")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


nc = 1
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
input = torch.FloatTensor(args.batchSize, 1, size, size)
train_dataset = ImageFolder(train, loader=cv2_loader, transform=img_transform)
train_loader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True)

test = args.test_dataroot
test_dataset = ImageFolder(test, loader=cv2_loader, transform=img_transform)
test_loader = DataLoader(test_dataset, batch_size=args.batchSize, shuffle=True)


latent_dim = 400
size_2 = int(size/2) #64
# size_4 = int(size/4) #32

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(size * size, size_2 * size_2)
        self.fc21 = nn.Linear(size_2 * size_2, latent_dim)
        self.fc22 = nn.Linear(size_2 * size_2, latent_dim)
        self.fc3 = nn.Linear(latent_dim, size_2 * size_2)
        self.fc4 = nn.Linear(size_2 * size_2, size * size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, size * size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, size * size), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

loss = []
def write_list(filename, data):
    thefile = open(filename, 'a')
    thefile.write("%s\n" % data)

        
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    write_list("loss.txt",  train_loss / len(train_loader.dataset))
    


def test(epoch):
    model.eval()
    test_loss = 0
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


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    sample = Variable(torch.randn(64, latent_dim))
    if args.cuda:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(64, 1, size, size),
               'results/sample_' + str(epoch) + '.png')

    torch.save(model.state_dict(), '%s/vae_epoch_%d.pth' % ("results", epoch))