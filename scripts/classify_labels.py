import os, argparse, pathlib, json, math, multiprocessing
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import nirhead.data.static_attributes as stat
from nirhead.models.classifier import load_classification_model_dir, ClassifierConfig
from nirhead.dataset.classification_dataset import ClassificationDataSet


device = torch.device('cuda')


class PathImageFolder(ImageFolder):
    
    def __getitem__(self, idx):
        return super(PathImageFolder, self).__getitem__(idx), self.samples[idx]


def main(args):
    src_dir = pathlib.Path(args.src)
    dst_file = pathlib.Path(args.dst)
    
    filenamefilter = lambda f: str(f.absolute())[len(str(src_dir.absolute())) + 1:]
    
    dl_workers = multiprocessing.cpu_count() if not args.src.lower().endswith(".zip") else 1
    dataset = PathImageFolder(args.src, transform=lambda img: ClassificationDataSet._image_transform(np.asarray(img, dtype=np.uint8), resolution=args.img_res, mode="gray"))
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=dl_workers, drop_last=False)
    
    json_dst = {"labels": {}}
    if os.path.isfile(dst_file) and os.path.exists(dst_file):
        with open(dst_file, "r") as f:
            json_dst = json.load(f)
    
    model, model_name = load_classification_model_dir(args.classifier, device, -1)
    model.requires_grad_(False)
    
    with torch.no_grad():
        with tqdm(total=len(dataloader)) as pbar:
            for batch_idx, (imgs, files) in enumerate(dataloader):
                imgs = imgs[0].to(device)
                files = list(files[0])
                
                y_pred = model(imgs).to("cpu")
                
                filename_keys = [filenamefilter(pathlib.Path(files[i])) for i in range(y_pred.shape[0])]
                batch_labels = stat.labels_from_attribute_tensor(y_pred, filename_keys, model.static_attributes)
                
                for filekey in batch_labels.keys():
                    if filekey not in json_dst["labels"].keys():
                        json_dst["labels"][filekey] = {}
                    for k in batch_labels[filekey].keys():
                        if (k not in json_dst["labels"][filekey].keys()) or (not args.add_only):
                            json_dst["labels"][filekey][k] = batch_labels[filekey][k]
                
                pbar.update(1)
    
    with open(args.dst, "w+") as f:
        json.dump(json_dst, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", type=str)
    parser.add_argument("-d", "--dst", type=str)
    parser.add_argument("-c", "--classifier", type=str)
    parser.add_argument("--add_only", action="store_true", default=False)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--img_res", type=int, default=128)
    args = parser.parse_args()
    
    main(args)
