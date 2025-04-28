#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gtb-container_g1.24h
#$ -ac d=nvcr-cuda-12.1.1-ubuntu20.04,d_shm=121G
#$ -N train_UNet_BRIGHT

. ~/net.sh



/home/songjian/anaconda3/envs/vmamba/bin/python script/train_baseline_network.py  \
    --dataset 'BRIGHT' \
    --train_batch_size 16 \
    --eval_batch_size 4 \
    --num_workers 16 \
    --crop_size 640 \
    --max_iters 800000 \
    --learning_rate 1e-4 \
    --model_type 'UNet' \
    --train_dataset_path '/data/ggeoinfo/datasets/DFC2025/Final_datasets/Track_2/dfc25_track2_trainval/train' \
    --train_data_list_path '/home/songjian/project/BRIGHT/dfc25_benchmark/dataset/splitname/train_setlevel.txt' \
    --holdout_dataset_path '/data/ggeoinfo/datasets/DFC2025/Final_datasets/Track_2/dfc25_track2_trainval/train' \
    --holdout_data_list_path '/home/songjian/project/BRIGHT/dfc25_benchmark/dataset/splitname/holdout_setlevel.txt' 