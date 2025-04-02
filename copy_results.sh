#!/bin/bash

echo -n "Enter password for local_admin@192.168.0.84: "
read -s password
echo ""

mkdir -p /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2
mkdir -p /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/code/nirhead
mkdir -p /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/code/eg3d

export SSHPASS="$password"
rsync -P -au --no-perms --rsh "sshpass -e ssh" local_admin@192.168.0.84:~/work/pascal/nirhead/models/gghead/GGHEAD-$1*/fakes[0-9][0-9][0-9][0-9][0-9][0-9].png /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/
rsync -P -au --no-perms --rsh "sshpass -e ssh" local_admin@192.168.0.84:~/work/pascal/nirhead/models/gghead/GGHEAD-$1*/fakes[0-9][0-9][0-9][0-9][0-9][0-9]_depth.png /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/
rsync -P -au --no-perms --rsh "sshpass -e ssh" local_admin@192.168.0.84:~/work/pascal/nirhead/models/gghead/GGHEAD-$1*/*.txt /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/
rsync -P -au --no-perms --rsh "sshpass -e ssh" local_admin@192.168.0.84:~/work/pascal/nirhead/models/gghead/GGHEAD-$1*/*.json /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/
rsync -P -au --no-perms --rsh "sshpass -e ssh" local_admin@192.168.0.84:~/work/pascal/nirhead/models/gghead/GGHEAD-$1*/*.jsonl /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/
rsync -P -au --no-perms --rsh "sshpass -e ssh" local_admin@192.168.0.84:~/work/pascal/nirhead/models/gghead/GGHEAD-$1*/evaluations /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/
rsync -P -au --no-perms --rsh "sshpass -e ssh" local_admin@192.168.0.84:~/work/pascal/nirhead/models/gghead/GGHEAD-$1*/checkpoints/checkpoint-*000.pkl /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/checkpoints/
rsync -P -au --no-perms --filter "+ *000.pth" --filter "+ *.json" --filter "- *.pth" --rsh "sshpass -e ssh" local_admin@192.168.0.84:~/work/pascal/nirhead/models/gghead/GGHEAD-$1*/classifier/ /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/classifier
rsync -P -au --no-perms --ignore-existing --filter "- *__pycache__*" --rsh "sshpass -e ssh" local_admin@192.168.0.84:~/work/pascal/nirhead/src/ /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/code/nirhead/src
rsync -P -au --no-perms --ignore-existing --filter "- *__pycache__*" --rsh "sshpass -e ssh" local_admin@192.168.0.84:~/work/pascal/nirhead/scripts/ /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/code/nirhead/scripts
rsync -P -au --no-perms --ignore-existing --filter "- *__pycache__*" --rsh "sshpass -e ssh" local_admin@192.168.0.84:~/work/pascal/eg3d/eg3d/ /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/code/eg3d/eg3d
