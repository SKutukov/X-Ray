from torchvision.utils import save_image
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


def on_epoch_end(epoch, num_epochs, loss, output, size):
    print('epoch [{}/{}], loss:{:.4f}'
        .format(epoch+1, num_epochs, loss.data[0]))
    data = output.cpu().data
    pic = to_img(data, size)
    save_image(pic, './dc_img/image_{}.png'.format(epoch),nrow=8)

