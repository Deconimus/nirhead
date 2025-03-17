#!/bin/bash

echo "curl --upload-file /home/local_admin/work/pascal/gghead/models/gghead/GGHEAD-$1*/fakes[0-9][0-9][0-9][0-9][0-9][0-9].png --netrc-file "~/work/pascal/.netrc" smb://192.168.0.30/CCMLOps/2024-11-07-MA_PascalSielski/gghead_results/$2/"
curl --upload-file /home/local_admin/work/pascal/gghead/models/gghead/GGHEAD-$1*/fakes[0-9][0-9][0-9][0-9][0-9][0-9]_depth.png --netrc-file "~/work/pascal/.netrc" smb://192.168.0.30/CCMLOps/2024-11-07-MA_PascalSielski/gghead_results/$2/
curl --upload-file /home/local_admin/work/pascal/gghead/models/gghead/GGHEAD-$1*/*.txt --netrc-file "~/work/pascal/.netrc" smb://192.168.0.30/CCMLOps/2024-11-07-MA_PascalSielski/gghead_results/$2/
curl --upload-file /home/local_admin/work/pascal/gghead/models/gghead/GGHEAD-$1*/*.json --netrc-file "~/work/pascal/.netrc" smb://192.168.0.30/CCMLOps/2024-11-07-MA_PascalSielski/gghead_results/$2/
curl --upload-file /home/local_admin/work/pascal/gghead/models/gghead/GGHEAD-$1*/*.jsonl --netrc-file "~/work/pascal/.netrc" smb://192.168.0.30/CCMLOps/2024-11-07-MA_PascalSielski/gghead_results/$2/
curl --upload-file /home/local_admin/work/pascal/gghead/models/gghead/GGHEAD-$1*/evaluations/*.json --netrc-file "~/work/pascal/.netrc" smb://192.168.0.30/CCMLOps/2024-11-07-MA_PascalSielski/gghead_results/$2/evaluations/
curl --upload-file /home/local_admin/work/pascal/gghead/models/gghead/GGHEAD-$1*/checkpoints/checkpoint-2*000.pkl --netrc-file "~/work/pascal/.netrc" smb://192.168.0.30/CCMLOps/2024-11-07-MA_PascalSielski/gghead_results/$2/checkpoints/
