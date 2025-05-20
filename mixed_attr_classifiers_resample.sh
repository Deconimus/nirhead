#!/bin/bash
export BW_IMPLEMENTATION=1

python scripts/train_classifier.py --dataset /mnt/g/EyesNIR/resample/enir128_resample_bp_eo_trainset --model vit --vit_mlp_dim 1024 --vit_heads 8 --vit_depth 4 --epochs 100 --labels bright_pupil eye_open --flip_aug
python scripts/train_classifier.py --dataset /mnt/g/EyesNIR/resample/enir128_resample_bp_eo_trainset --model vit --vit_mlp_dim 2048 --vit_heads 8 --vit_depth 4 --epochs 100 --labels bright_pupil eye_open --flip_aug
python scripts/train_classifier.py --dataset /mnt/g/EyesNIR/resample/enir128_resample_eo_gz_trainset --model vit --vit_mlp_dim 1024 --vit_heads 8 --vit_depth 4 --epochs 100 --labels eye_open gaze --flip_aug
python scripts/train_classifier.py --dataset /mnt/g/EyesNIR/resample/enir128_resample_eo_gz_trainset --model vit --vit_mlp_dim 2048 --vit_heads 8 --vit_depth 4 --epochs 100 --labels eye_open gaze --flip_aug
python scripts/train_classifier.py --dataset /mnt/g/EyesNIR/resample/enir128_resample_bp_gz_trainset --model vit --vit_mlp_dim 1024 --vit_heads 8 --vit_depth 4 --epochs 100 --labels bright_pupil gaze --flip_aug
python scripts/train_classifier.py --dataset /mnt/g/EyesNIR/resample/enir128_resample_bp_gz_trainset --model vit --vit_mlp_dim 2048 --vit_heads 8 --vit_depth 4 --epochs 100 --labels bright_pupil gaze --flip_aug
python scripts/train_classifier.py --dataset /mnt/g/EyesNIR/resample/enir128_resample_bp_eo_gz_trainset --model vit --vit_mlp_dim 1024 --vit_heads 8 --vit_depth 4 --epochs 100 --labels bright_pupil eye_open gaze --flip_aug
python scripts/train_classifier.py --dataset /mnt/g/EyesNIR/resample/enir128_resample_bp_eo_gz_trainset --model vit --vit_mlp_dim 2048 --vit_heads 8 --vit_depth 4 --epochs 100 --labels bright_pupil eye_open gaze --flip_aug

python scripts/train_classifier.py --dataset /mnt/g/EyesNIR/resample/enir128_resample_bp_trainset --model vit --vit_mlp_dim 1024 --vit_heads 8 --vit_depth 4 --epochs 100 --labels bright_pupil --flip_aug
python scripts/train_classifier.py --dataset /mnt/g/EyesNIR/resample/enir128_resample_bp_trainset --model vit --vit_mlp_dim 2048 --vit_heads 8 --vit_depth 4 --epochs 100 --labels bright_pupil --flip_aug
python scripts/train_classifier.py --dataset /mnt/g/EyesNIR/resample/enir128_resample_eo_trainset --model vit --vit_mlp_dim 1024 --vit_heads 8 --vit_depth 4 --epochs 100 --labels eye_open --flip_aug
python scripts/train_classifier.py --dataset /mnt/g/EyesNIR/resample/enir128_resample_eo_trainset --model vit --vit_mlp_dim 2048 --vit_heads 8 --vit_depth 4 --epochs 100 --labels eye_open --flip_aug
python scripts/train_classifier.py --dataset /mnt/g/EyesNIR/resample/enir128_resample_gz_trainset --model vit --vit_mlp_dim 1024 --vit_heads 8 --vit_depth 4 --epochs 100 --labels gaze --flip_aug
python scripts/train_classifier.py --dataset /mnt/g/EyesNIR/resample/enir128_resample_gz_trainset --model vit --vit_mlp_dim 2048 --vit_heads 8 --vit_depth 4 --epochs 100 --labels gaze --flip_aug
