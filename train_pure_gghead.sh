#!/bin/bash

export BW_IMPLEMENTATION=1

python scripts/train_gghead.py afhq ivgaze_eyes64_balanced_trainsubjects.zip 1 4 --kimg 3000 --resolution 64 --plane_resolution 128 --n_uniform_flame_vertices 128 --no_use_gsm_flame_template --use_plane_template --no_use_masks --no_apply_masks --no_blur_masks --image_snapshot_ticks 16 --metrics fid10k --gghead_pure

python scripts/train_gghead.py afhq ivgaze_faces256_balanced_trainsubjects.zip 1 4 --kimg 26000 --resume_run GGHEAD-1_ffhq512 --overwrite_resolution 256 --overwrite_n_uniform_flame_vertices 384 --overwrite_lambda_tv_uv_rendering 100 --overwrite_lambda_beta_loss 1 --overwrite_use_masks False --overwrite_apply_masks False --no_blur_masks --image_snapshot_ticks 16 --metrics fid10k --gghead_pure

python scripts/train_gghead.py afhq enir128balanced_trainsubjects.zip 1 4 --kimg 3000 --resolution 128 --plane_resolution 128 --n_uniform_flame_vertices 128 --no_use_gsm_flame_template --use_plane_template --no_use_masks --no_apply_masks --no_blur_masks --image_snapshot_ticks 16 --metrics fid10k --gghead_pure

python scripts/train_gghead.py afhq fnir512balanced_trainsubjects.zip 1 4 --kimg 26000 --resume_run GGHEAD-1_ffhq512 --overwrite_resolution 256 --overwrite_n_uniform_flame_vertices 384 --overwrite_lambda_tv_uv_rendering 100 --overwrite_lambda_beta_loss 1 --image_snapshot_ticks 16 --metrics fid10k --gghead_pure

python scripts/train_gghead.py afhq udc_faces_all.zip 1 4 --kimg 26000 --resume_run GGHEAD-1_ffhq512 --overwrite_resolution 256 --overwrite_n_uniform_flame_vertices 384 --overwrite_lambda_tv_uv_rendering 100 --overwrite_lambda_beta_loss 1 --overwrite_use_masks False --overwrite_apply_masks False --no_blur_masks --image_snapshot_ticks 16 --metrics fid10k --gghead_pure

python scripts/train_gghead.py afhq udc_eyes_all.zip 1 4 --kimg 3000 --resolution 128 --plane_resolution 128 --n_uniform_flame_vertices 128 --no_use_gsm_flame_template --use_plane_template --no_use_masks --no_apply_masks --no_blur_masks --image_snapshot_ticks 16 --metrics fid10k --gghead_pure
