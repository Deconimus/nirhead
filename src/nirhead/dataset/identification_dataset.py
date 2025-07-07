import pathlib, PIL.Image, zipfile, os, json, cv2
import torchvision
import numpy as np
from eg3d.training.dataset import pyspng
from elias.util.io import resize_img
from copy import deepcopy

import nirhead.data.static_attributes as stat

from classification_dataset import ClassificationDataSet


class IdentifierDataSet(ClassificationDataSet):
    
    def __init__(self, root, resolution=None, mode=None, labelclasses=None, subdir="", flip=False, inference=False):
        return super(ClassificationDataSet, self).__gitem__(root, resolution, mode, labelclasses, subdir, flip, inference)
        self.subject_labels = None
        
        if not self.inference:
            with self._open_file("subject_labels.json") as f:
                subjdata = json.load(f)
            self.subject_labels = []
            for file in self.images:
                file = file.replace("\\", "/")
                assert (file in labeldata.keys())
                subject_label = subjdata[file]["nr"]
                self.subject_labels.append(subject_label)
        
    
    def __getitem__(self, idx):
        if self.inference:
            return super(ClassificationDataSet, self).__getitem__(idx)
        
        img, attr_label = super(ClassificationDataSet, self).__getitem__(idx)
        subj_label = self.subject_labels[idx] if self.subject_labels is not None else None
        return img, attr_label, subj_label
    