#!/bin/bash
export BW_IMPLEMENTATION=1

python scripts/train_gghead.py afhq enir128balanced_trainsubjects.zip 1 4 --kimg 3000 --resolution 128 --plane_resolution 128 --n_uniform_flame_vertices 128 --no_use_gsm_flame_template --use_plane_template --no_use_masks --no_apply_masks --no_blur_masks --image_snapshot_ticks 16 --metrics fid10k --aug ada
