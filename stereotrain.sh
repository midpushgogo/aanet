#!/usr/bin/env bash

# Train on Scene Flow training set
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--mode test  \
--data_dir data \
--checkpoint_dir checkpoints/stereo_sceneflow \
--batch_size 14 \
--learning_rate 4e-4 \
--val_batch_size 12 \
--img_height 256 \
--img_width 512 \
--val_img_height 576 \
--val_img_width 960 \
--feature_type stereonet \
--feature_similarity difference \
--refinement_type stereonet \
--aggregation_type stereonet  \
--milestones 20,30,40,50,60 \
--max_epoch 10  \
--print_freq 1 \
--attention
