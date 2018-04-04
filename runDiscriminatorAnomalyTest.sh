#!bash
python3 dcgan/discriminator_run.py --dataroot /home/skutukov96/diplom/train --workers 4 \
--netD /home/skutukov96/diplom/PyTorch_xray/out64x64/netD_epoch_98.pth --batchSize 1 --cuda \
--outTXT Discriminator_norm_test_norm.txt
