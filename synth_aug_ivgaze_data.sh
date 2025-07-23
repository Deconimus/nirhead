#!/bin/bash
export BW_IMPLEMENTATION=1

# Gaze
# TODO: add model
#python scripts/synthesize_trainingdata.py -m gh --batch 8 --res 64 --poses /mnt/d/IVGazeDataset/scripts/pose_eyes.json --classifier models/classifier/vit16_gz_iveyes --labels gaze --dst /mnt/d/IVGazeDataset/synthetic_augmented_classifiers/trainsets/gaze/mixed_real_synthetic_samesize -n 2500
#python scripts/synthesize_trainingdata.py -m gh --batch 8 --res 64 --poses /mnt/d/IVGazeDataset/scripts/pose_eyes.json --classifier models/classifier/vit16_gz_iveyes --labels gaze --dst /mnt/d/IVGazeDataset/synthetic_augmented_classifiers/trainsets/gaze/full_real_synthetic_extended -n 5000
#python scripts/synthesize_trainingdata.py -m gh --batch 8 --res 64 --poses /mnt/d/IVGazeDataset/scripts/pose_eyes.json --classifier models/classifier/vit16_gz_iveyes --labels gaze --dst /mnt/d/IVGazeDataset/synthetic_augmented_classifiers/trainsets/gaze/full_real_synthetic_extended_large -n 50000
#python scripts/synthesize_trainingdata.py -m gh --batch 8 --res 64 --poses /mnt/d/IVGazeDataset/scripts/pose_eyes.json --classifier models/classifier/vit16_gz_iveyes --labels gaze --dst /mnt/d/IVGazeDataset/synthetic_augmented_classifiers/trainsets/gaze/synthetic_narrow -n 2500 --filter_gz_radius 25
#python scripts/synthesize_trainingdata.py -m gh --batch 8 --res 64 --poses /mnt/d/IVGazeDataset/scripts/pose_eyes.json --classifier models/classifier/vit16_gz_iveyes --labels gaze --dst /mnt/d/IVGazeDataset/synthetic_augmented_classifiers/trainsets/gaze/synthetic_narrower -n 2500 --filter_gz_radius 19
#python scripts/synthesize_trainingdata.py -m gh --batch 8 --res 64 --poses /mnt/d/IVGazeDataset/scripts/pose_eyes.json --classifier models/classifier/vit16_gz_iveyes --labels gaze --dst /mnt/d/IVGazeDataset/synthetic_augmented_classifiers/trainsets/gaze/synthetic_wide -n 2500 --filter_gz_deadzone 25
#python scripts/synthesize_trainingdata.py -m gh --batch 8 --res 64 --poses /mnt/d/IVGazeDataset/scripts/pose_eyes.json --classifier models/classifier/vit16_gz_iveyes --labels gaze --dst /mnt/d/IVGazeDataset/synthetic_augmented_classifiers/trainsets/gaze/complement_wide -n 2500 --filter_gz_deadzone 25
#python scripts/synthesize_trainingdata.py -m gh --batch 8 --res 64 --poses /mnt/d/IVGazeDataset/scripts/pose_eyes.json --classifier models/classifier/vit16_gz_iveyes --labels gaze --dst /mnt/d/IVGazeDataset/synthetic_augmented_classifiers/trainsets/gaze/complement_wider -n 2500 --filter_gz_deadzone 19
#python scripts/synthesize_trainingdata.py -m gh --batch 8 --res 64 --poses /mnt/d/IVGazeDataset/scripts/pose_eyes.json --classifier models/classifier/vit16_gz_iveyes --labels gaze --dst /mnt/d/IVGazeDataset/synthetic_augmented_classifiers/trainsets/gaze/complement_narrow -n 2500 --filter_gz_radius 25
