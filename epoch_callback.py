from torchvision.utils import save_image

def to_img(x,size):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, size, size)
    return x

def on_epoch_end(epoch, num_epochs, loss, output, size):
    print('epoch [{}/{}], loss:{:.4f}'
        .format(epoch+1, num_epochs, loss.data[0]))
    if epoch % 2 == 0:
        pic = to_img(output.cpu().data, size)
        save_image(pic, './dc_img/image_{}.png'.format(epoch),nrow=1)
