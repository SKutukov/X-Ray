from torchvision.utils import save_image
import torch
import os
import cv2

def to_img(x,size):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, size, size)
    return x

def save_cv2(epoch,path, im):
    epoch_dir = os.path.join(path, "epoch-" + str(epoch))

    if not os.path.exists(path):
        os.mkdir(path)

    if not os.path.exists(epoch_dir):
        os.mkdir(epoch_dir)

    path = os.path.join(epoch_dir, "epoch-{:03d}.png")
    im_path = path.format(epoch)
    cv2.imwrite(im_path,im)



def on_epoch_end(epoch, output, size, input):

    n = min(output.size(0), 8)
    comparison = torch.cat([input[:n],
                                  output[:n]])
    save_image(comparison.cpu().data, './result/dc_img/image_{}.png'.format(epoch),nrow=8)

