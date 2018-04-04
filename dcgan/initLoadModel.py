import json
import os
def loadFromCheckpoint(config_name, outf):
    Loss_D = []
    Loss_G = []
    D_x_list = []
    D_G_z1_list = []
    D_G_z2_list = []
    epoch_begin = 1

    epochs = []
    
    with  open(config_name):
        data = json.load(open(config_name))

        with open(os.path.join(outf,data["Loss_D"]), 'r') as f:
            Loss_D = f.readlines()

        with open(os.path.join(outf,data["Loss_G"]), 'r') as f:
            Loss_G = f.readlines()        

        with open(os.path.join(outf,data["D_x_list"]), 'r') as f:
            D_x_list = f.readlines()        

        with open(os.path.join(outf,data["D_G_z1_list"]), 'r') as f:
            D_G_z1_list = f.readlines()        

        with open(os.path.join(outf,data["D_G_z2_list"]), 'r') as f:
            D_G_z2_list = f.readlines()        
       
        #init epochs for graph
        epoch_begin = int(data["epoch_begin"])
        for x in range(0, epoch_begin):
            epochs.append(x)
        
    return Loss_D, Loss_G, D_x_list, D_G_z1_list, D_G_z2_list, epochs,  epoch_begin