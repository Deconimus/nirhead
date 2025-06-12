#!/bin/bash

python scripts/render_grid.py --dataset /mnt/g/EyesNIR/resample/enir128balanced_resample_trainsubjects.zip -m gh90_enir_resample_balanced_trainsubjects_cl075_eo_plane_attrskip_fromscratch --attribute_gradient eye_open --batch 16 --rows 16 --cols 16 --checkpoints 1000 1600 2000 2600 3000
python scripts/render_grid.py --dataset /mnt/g/EyesNIR/resample/enir128balanced_resample_trainsubjects.zip -m gh91_enir_balanced_trainsubjects_cl075_eo_plane_attrskip_fromscratch --attribute_gradient eye_open --batch 16 --rows 16 --cols 16 --checkpoints 1000 1600 2000 2600 3000
python scripts/render_grid.py --dataset /mnt/g/EyesNIR/resample/enir128balanced_resample_trainsubjects.zip -m gh92_enir_balanced_trainsubjects_cl07_bp_plane_attrskip_fromscratch --attribute_gradient bright_pupil --batch 16 --rows 16 --cols 16 --checkpoints 1000 1600 2000 2600 3000
python scripts/render_grid.py --dataset /mnt/g/EyesNIR/resample/enir128balanced_resample_trainsubjects.zip -m gh93_enir_resample_balanced_trainsubjects_cl075_gz_plane_attrskip_fromscratch --attribute_gradient gaze --batch 5 --rows 15 --cols 15 --subgrids_x 3 --subgrids_y 3 --checkpoints 1000 1600 2000 2600 3000

python scripts/render_video.py -m gh90_enir_resample_balanced_trainsubjects_cl075_eo_plane_attrskip_fromscratch --attribute eo --checkpoints 1600 2000 2600 3000 --seed 136 -t 10 --rows 4 --cols 8 --batch 32
python scripts/render_video.py -m gh91_enir_balanced_trainsubjects_cl075_eo_plane_attrskip_fromscratch --attribute eo --checkpoints 1600 2000 2600 3000 --seed 136 -t 10 --rows 4 --cols 8 --batch 32
python scripts/render_video.py -m gh93_enir_resample_balanced_trainsubjects_cl075_gz_plane_attrskip_fromscratch --attribute gz --checkpoints 1600 2000 2600 3000 --seed 136 -t 15 --rows 4 --cols 8 --batch 32
