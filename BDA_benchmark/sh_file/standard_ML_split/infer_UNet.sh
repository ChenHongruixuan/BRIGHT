#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gtn-container_g1.24h
#$ -ac d=aip-cuda-12.0.1-blender-2,d_shm=64G
#$ -N infer_UNet_BRIGHT

. ~/net.sh

/home/songjian/anaconda3/envs/vmamba/bin/python /home/songjian/project/BRIGHT/essd/script/standard_ML/infer_UNet.py \
    --model_path  '/home/songjian/project/BRIGHT/essd/saved_weights/BRIGHT/UNet_20250304_221841/best_model.pth' \
    --test_dataset_path '/data/ggeoinfo/datasets/BRIGHT' \
    --test_data_list_path  '/home/songjian/project/BRIGHT/essd/dataset/splitname/standard_ML/test_set.txt'  \
    --output_dir '/home/songjian/project/BRIGHT/essd/inference_results/BRIGHT/UNet/github_test'
    