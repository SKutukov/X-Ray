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
size = 64
   
def cv2_loader(filename):
    im = cv2.imread(filename, 0)
    im = cv2.resize(im,(size,size),interpolation = cv2.INTER_AREA)
    im = im.reshape(size, size, 1)
    return im

def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph
    
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}
    print(param_map)
    
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    
    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot


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
    
    from graphviz import Digraph
    
    # inputs = torch.randn(1,1,1024,1024)
    # from autoencoder import encoder
    # dec = encoder()
    # y = dec(Variable(inputs))
    # # print(y)
    # g = make_dot(y, dec.state_dict())
    # g.view()

    # inputs = torch.randn(64,64,2,2)
    # from autoencoder import decoder

    # dec = decoder()
    # y = dec(Variable(inputs))
    # # print(y)
    # g = make_dot(y, dec.state_dict())
    # g.view()
    # exec(0)

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
