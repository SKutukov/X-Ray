from torch import save 

def save_model(epoch, model):
    if epoch % 10 == 0 and epoch != 0:
        print('save model on {} epoch'.format(epoch+1))
        save(model.state_dict(), './conv_autoencoder.pth')
