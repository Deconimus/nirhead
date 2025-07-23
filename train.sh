#!/bin/bash

export BW_IMPLEMENTATION=1

bash -x synthetic_complement_classifiers.sh
bash -x synth_aug_ivgaze_classifiers.sh
bash -x synth_aug_fnir_classifiers.sh
