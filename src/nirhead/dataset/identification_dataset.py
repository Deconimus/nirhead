import pathlib, PIL.Image, zipfile, os, json, cv2, random
import torch, torchvision
import numpy as np
from eg3d.training.dataset import pyspng
from elias.util.io import resize_img
from copy import deepcopy

import nirhead.data.static_attributes as stat

from nirhead.dataset.classification_dataset import ClassificationDataSet


class IdentificationDataSet(ClassificationDataSet):
    
    def __init__(self, root, resolution=None, mode=None, labelclasses=None, subdir="", flip=False, inference=False, strict_pose=False, latents=False):
        super(IdentificationDataSet, self).__init__(root, resolution, mode, labelclasses, subdir, flip, inference)
        self.subject_labels = None
        self.subject_labels_index = None
        self.poses = []
        self.latents = []
        
        poses_skipped = 0
        remove_files_indices = []
        if latents:
            with self._open_file("labels.json") as f:
                posedata = json.load(f)
                posekeys = list(posedata.keys())
            for idx, file in enumerate(self.images):
                file = file.replace("\\", "/")
                if not file in posedata.keys() or not "z" in posedata[file].keys() or not "c" in posedata[file].keys():
                    if strict_pose:
                        remove_files_indices.append(idx)
                        continue
                    else:
                        randind = random.randint(0, len(posekeys)-1)
                        pose = posedata[posekeys[randind]]["c"]
                        latent = posedata[posekeys[randind]]["z"]
                        poses_skipped += 1
                else:
                    pose = posedata[file]["c"]
                    latent = posedata[file]["z"]
                pose = torch.tensor(pose, dtype=torch.float32)
                latent = torch.tensor(latent, dtype=torch.float32)
                self.poses.append(pose)
                self.latents.append(latent)
        else:
            if os.path.exists(root + "/d3fr_poses.json"):
                with self._open_file("d3fr_poses.json") as f:
                    posedata = json.load(f)["labels"]
                    posedata = {posedata[i][0]: posedata[i][1] for i in range(len(posedata))}
                    posekeys = list(posedata.keys())
                    # print(posekeys)
                for idx, file in enumerate(self.images):
                    file = file.replace("\\", "/")
                    file = file[file.index("/") + 1:].replace("/", "+")
                    # print(file)
                    if not file in posedata.keys():
                        if strict_pose:
                            remove_files_indices.append(idx)
                            continue
                        else:
                            pose = posedata[posekeys[random.randint(0, len(posekeys) - 1)]]
                            poses_skipped += 1
                    else:
                        pose = posedata[file]
                    pose = torch.tensor(pose, dtype=torch.float32)
                    self.poses.append(pose)
        
        if len(self.poses) <= 0: # no pose data
            poses_skipped = len(self.images)
            if strict_pose:
                remove_files_indices = list(range(len(self.images)))
        if strict_pose:
            remove_files_indices.sort()
            for idx in remove_files_indices[::-1]:
                self.images.pop(idx)
            print(f"Images discarded due to missing pose: {len(remove_files_indices)}")
        else:
            print(f"Poses skipped: {poses_skipped}")
        
        if len(self.poses) <= 0:
            self.poses = None
        if len(self.latents) <= 0:
            self.latents = None
        
        if not inference and not latents:
            with self._open_file("subject_labels.json") as f:
                subjdata = json.load(f)
            self.subject_labels = []
            self.subject_labels_index = {}
            subj_label_counter = 0
            for file in self.images:
                file = file.replace("\\", "/").replace("+", "/")
                if not file in subjdata.keys():
                    print(file)
                assert (file in subjdata.keys())
                subject_label = subjdata[file]["nr"]
                self.subject_labels.append(subject_label)
                if subject_label not in self.subject_labels_index.keys():
                    self.subject_labels_index[subject_label] = subj_label_counter
                    subj_label_counter += 1
    
    def __getitem__(self, idx):
        
        pose = self.poses[idx] if self.poses is not None else None
        latent = self.latents[idx] if self.latents is not None else None
        
        if self.inference:
            return super(IdentificationDataSet, self).__getitem__(idx), pose
        
        img, _ = super(IdentificationDataSet, self).__getitem__(idx)
        subj_label = self.subject_labels[idx] if self.subject_labels is not None else None
        
        if latent is not None:
            return img, pose, latent
        return img, subj_label, pose
    