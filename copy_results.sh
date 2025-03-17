#!/bin/bash

echo -n "Enter password for local_admin@192.168.0.84: "
read -s password

sshpass -p $password scp local_admin@192.168.0.84:~/work/pascal/gghead/models/gghead/GGHEAD-$1*/fakes[0-9][0-9][0-9][0-9][0-9][0-9].png /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/
sshpass -p $password scp local_admin@192.168.0.84:~/work/pascal/gghead/models/gghead/GGHEAD-$1*/fakes[0-9][0-9][0-9][0-9][0-9][0-9]_depth.png /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/
sshpass -p $password scp local_admin@192.168.0.84:~/work/pascal/gghead/models/gghead/GGHEAD-$1*/*.txt /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/
sshpass -p $password scp local_admin@192.168.0.84:~/work/pascal/gghead/models/gghead/GGHEAD-$1*/*.json /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/
sshpass -p $password scp local_admin@192.168.0.84:~/work/pascal/gghead/models/gghead/GGHEAD-$1*/*.jsonl /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/
sshpass -p $password scp -r local_admin@192.168.0.84:~/work/pascal/gghead/models/gghead/GGHEAD-$1*/evaluations /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/
sshpass -p $password scp local_admin@192.168.0.84:~/work/pascal/gghead/models/gghead/GGHEAD-$1*/checkpoints/checkpoint-2*000.pkl /mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results/$2/checkpoints/
