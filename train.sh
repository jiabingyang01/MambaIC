# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --cuda -d  dataset \
#     -n 128 --lambda 0.025 --epochs 250 --lr_epoch 200 240 --batch-size 6 \
#     --save_path checkpoint/kodak --save --train-dataname flickr30k  --test-dataname Kodak \

CUDA_VISIBLE_DEVICES=0 python train_ori.py --cuda -d  dataset \
    -n 128 --lambda 0.008 --epochs 25 --lr_epoch 15 20 --batch-size 8 \
    --save_path checkpoint/wildfire --save --train-dataname wildfire/train  --test-dataname wildfire/test --save \
    --checkpoint /home/zhaorun/zichen/yjb/projects/CV/MambaIC/checkpoint/wildfire/0.008checkpoint_latest.pth.tar --continue_train \

# CUDA_VISIBLE_DEVICES=0 python train_ori.py --cuda -d  dataset \
#     -n 128 --lambda 0.005 --epochs 25 --lr_epoch 15 20 --batch-size 8 \
#     --save_path checkpoint/wildfire --save --train-dataname wildfire/train  --test-dataname wildfire/test --save \
    # --checkpoint /home/zhaorun/zichen/yjb/projects/CV/MambaIC/checkpoint/wildfire/0.015checkpoint_latest.pth.tar --continue_train \

# torchrun --nproc_per_node=4 train.py --cuda -d  dataset \
#     -n 128 --lambda 0.025 --epochs 250 --lr_epoch 200 240 --batch-size 6 \
#     --save_path checkpoint/wildfire --save --train-dataname wildfire/train  --test-dataname wildfire/test 
