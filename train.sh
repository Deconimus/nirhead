#!/bin/bash

export BW_IMPLEMENTATION=1
python scripts/train_gghead.py afhq enir128balanced.zip 1 4 --resume-run gh28_enir_properpose --resume-checkpoint 1000 --kimg 2000 --resolution 128 --plane_resolution 128 --n_uniform_flame_vertices 128 --no_use_gsm_flame_template --use_plane_template --no_use_masks --no_apply_masks --no_blur_masks --image_snapshot_ticks 16 --metrics fid10k --classifier vit16_bp --static_attributes bright_pupil --overwrite-lambda-classifier 0.5
python scripts/train_gghead.py afhq enir128balanced.zip 1 4 --kimg 1000 --resolution 128 --plane_resolution 128 --n_uniform_flame_vertices 128 --no_use_gsm_flame_template --use_plane_template --no_use_masks --no_apply_masks --no_blur_masks --image_snapshot_ticks 16 --metrics fid10k --classifier vit16_bp --static_attributes bright_pupil --lambda-classifier 0.5
