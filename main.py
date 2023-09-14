import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"

from dccl import DCCL

if __name__=='__main__':
    cloud_epochs = 2
    kd_epochs = 50 # 25*lr0.001, 25*lr0.0001
    dc_rounds = 10
    kd_lr = 0.001
    cloud_lr = 0.01
    device_lr = 0.01
    tuning_epochs = 1

    device_cloud = DCCL(cloud_epochs, kd_epochs, dc_rounds, cloud_lr, kd_lr, cloud_lr, device_lr, tuning_epochs)
    device_cloud.dccl()