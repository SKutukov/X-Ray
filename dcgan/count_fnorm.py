import statistics 



def count(filename):
    D_x_list = []
    with open(filename, 'r') as f:
        D_x_list = f.readlines() 
    norm = 0
    anomaly = 0
    for  D_x in  D_x_list:
        if float(D_x) >= 0.5:
            norm +=1
        else:
            anomaly +=1    

    print("norm:", norm)
    print("anomaly:", anomaly)   

D_x_list_file = "/home/skutukov96/diplom/PyTorch_xray/Discriminator_test_anomaly.txt"
count(D_x_list_file)
D_x_list_file = "/home/skutukov96/diplom/PyTorch_xray/Discriminator_test_norm.txt"
count(D_x_list_file)