#!bash
python3 dcgan/discriminator_run.py --dataroot ~/diplom/Anomaly_Chest_xray --workers 4 \
--netD ~/diplom/PyTorch_xray/out64x64/netD_epoch_98.pth --batchSize 1 --cuda \
--outTXT Discriminator_norm_test_anomaly.txt
