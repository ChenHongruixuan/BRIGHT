#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gtn-container_g1.24h
#$ -ac d=aip-cuda-12.0.1-blender-2,d_shm=121G
#$ -N UNet_BRIGHT_SetLevel_20250503

. ~/net.sh

/home/songjian/anaconda3/envs/vmamba/bin/python /home/songjian/project/BRIGHT/essd/script/standard_ML/train_UNet.py  \
    --dataset 'BRIGHT' \
    --train_batch_size 16 \
    --eval_batch_size 4 \
    --num_workers 16 \
    --crop_size 640 \
    --max_iters 800000 \
    --model_type 'UNet' \
    --learning_rate 1e-4 \
    --model_param_path '/home/songjian/project/BRIGHT/essd/saved_weights' \
    --train_dataset_path '/data/ggeoinfo/datasets/BRIGHT' \
    --train_data_list_path '/home/songjian/project/BRIGHT/essd/dataset/splitname/standard_ML/train_set.txt' \
    --val_dataset_path '/data/ggeoinfo/datasets/BRIGHT' \
    --val_data_list_path '/home/songjian/project/BRIGHT/essd/dataset/splitname/standard_ML/val_set.txt'  \
    --test_dataset_path '/data/ggeoinfo/datasets/BRIGHT' \
    --test_data_list_path '/home/songjian/project/BRIGHT/essd/dataset/splitname/standard_ML/test_set.txt' 