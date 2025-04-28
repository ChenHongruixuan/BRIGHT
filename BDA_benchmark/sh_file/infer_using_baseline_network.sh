#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gtn-container_g1.24h
#$ -ac d=nvcr-cuda-12.1.1-ubuntu20.04,d_shm=121G
#$ -N infer_SiamCRNN_BRIGHT

. ~/net.sh



/home/songjian/anaconda3/envs/vmamba/bin/python script/infer_using_baseline_network.py  \
    --val_dataset_path '/data/ggeoinfo/datasets/DFC2025/Final_datasets/Track_2/dfc25_track2_trainval/val' \
    --val_data_list_path '/home/songjian/project/BRIGHT/dfc25_benchmark/dataset/splitname/val_setlevel.txt' \
    --existing_weight_path '/home/songjian/project/BRIGHT/dfc25_benchmark/saved_weights/BRIGHT/SiamCRNN_20250111_054333/best_model.pth' \
    --inferece_saved_path '/home/songjian/project/BRIGHT/dfc25_benchmark/inference_results'