#!bash
python3 dcgan/main.py --niter 500 \
--outf out \
--dataroot /home/skutukov96/diplom/train --workers 4 \
 --resume \
 --cuda \
# --imageSize 128 \
# --nz 500 \
# --ngf 128 \
# --ndf 128 \
# --batchSize -128 \
#  --epoch_start 4 \
  --netG /home/skutukov96/diplom/PyTorch_xray/out/netG_epoch_19.pth \
  --netD /home/skutukov96/diplom/PyTorch_xray/out/netD_epoch_19.pth
