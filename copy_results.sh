#!/bin/bash

ARG_HOST=${3:-1}
ARG_DST_ROOT=${4:-"/mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results"}

if [[ $ARG_HOST -eq 0 ]]
then
  HOST_NAME="self"
  HOST_ROOT=~
else
  HOST_NAME="local_admin@192.168.0.84"
  HOST_ROOT="local_admin@192.168.0.84:~/work/pascal"
  echo -n "Enter password for $HOST_NAME: "
  read -s password
  echo ""
fi

mkdir -p $ARG_DST_ROOT/$2
mkdir -p $ARG_DST_ROOT/$2/code/nirhead
mkdir -p $ARG_DST_ROOT/$2/code/eg3d

export SSHPASS="$password"
rsync -P -au --no-perms --rsh "sshpass -e ssh" $HOST_ROOT/nirhead/models/gghead/GGHEAD-$1*/fakes[0-9][0-9][0-9][0-9][0-9][0-9].png $ARG_DST_ROOT/$2/
rsync -P -au --no-perms --rsh "sshpass -e ssh" $HOST_ROOT/nirhead/models/gghead/GGHEAD-$1*/fakes[0-9][0-9][0-9][0-9][0-9][0-9]_depth.png $ARG_DST_ROOT/$2/
rsync -P -au --no-perms --rsh "sshpass -e ssh" $HOST_ROOT/nirhead/models/gghead/GGHEAD-$1*/*.txt $ARG_DST_ROOT/$2/
rsync -P -au --no-perms --rsh "sshpass -e ssh" $HOST_ROOT/nirhead/models/gghead/GGHEAD-$1*/*.json $ARG_DST_ROOT/$2/
rsync -P -au --no-perms --rsh "sshpass -e ssh" $HOST_ROOT/nirhead/models/gghead/GGHEAD-$1*/*.jsonl $ARG_DST_ROOT/$2/
rsync -P -au --no-perms --rsh "sshpass -e ssh" $HOST_ROOT/nirhead/models/gghead/GGHEAD-$1*/evaluations $ARG_DST_ROOT/$2/
rsync -P -au --no-perms --rsh "sshpass -e ssh" $HOST_ROOT/nirhead/models/gghead/GGHEAD-$1*/checkpoints/checkpoint-*000.pkl $ARG_DST_ROOT/$2/checkpoints/
rsync -P -au --no-perms --rsh "sshpass -e ssh" $HOST_ROOT/nirhead/models/gghead/GGHEAD-$1*/checkpoints/checkpoint-*600.pkl $ARG_DST_ROOT/$2/checkpoints/
rsync -P -au --no-perms --filter "+ *000.pth" --filter "+ *600.pth" --filter "+ *.json" --filter "- *.pth" --rsh "sshpass -e ssh" $HOST_ROOT/nirhead/models/gghead/GGHEAD-$1*/classifier/ $ARG_DST_ROOT/$2/classifier
rsync -P -au --no-perms --ignore-existing --filter "- *__pycache__*" --rsh "sshpass -e ssh" $HOST_ROOT/nirhead/src/ $ARG_DST_ROOT/$2/code/nirhead/src
rsync -P -au --no-perms --ignore-existing --filter "- *__pycache__*" --rsh "sshpass -e ssh" $HOST_ROOT/nirhead/scripts/ $ARG_DST_ROOT/$2/code/nirhead/scripts
rsync -P -au --no-perms --ignore-existing --filter "- *__pycache__*" --rsh "sshpass -e ssh" $HOST_ROOT/eg3d/eg3d/ $ARG_DST_ROOT/$2/code/eg3d/eg3d
