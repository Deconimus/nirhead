#!/bin/bash

export BW_IMPLEMENTATION=1

bash -x synth_aug_fnir_data.sh
bash -x synth_aug_fnir_classifiers.sh
