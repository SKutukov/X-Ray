from torch import save 

def save_checkpoint(state, check_point_perion = 1, filename='checkpoint.pth.tar'):
    epoch = state['epoch']
    if epoch % check_point_perion == 0 and epoch != 0:
        print('save state on {} epoch'.format(epoch))
        save(state, filename)

