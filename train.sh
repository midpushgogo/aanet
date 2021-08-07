#!/usr/bin/env bash

# Train on Scene Flow training set
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--mode val \
--data_dir data \
--checkpoint_dir checkpoints/aanet_sceneflow \
--batch_size 12 \
--val_batch_size 12 \
--img_height 288 \
--img_width 576 \
--val_img_height 576 \
--val_img_width 960 \
--feature_type aanet \
--feature_pyramid_network \
--milestones 20,30,40,50,60 \
--max_epoch 64 \
--attention \
--combinecost \
--print_freq 1
