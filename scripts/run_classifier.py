import os, gc, argparse, pathlib, multiprocessing, json, math
import torch, torchvision, torchsummary
import numpy as np
from torch import nn
from timeit import default_timer as timer
from tqdm import tqdm
from torch.utils.data import DataLoader

from gghead.models import classifier
from gghead.dataset.classification_dataset import ClassificationDataSet


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert (device == "cuda:0")
    
    model_dir = pathlib.Path(args.model)
    if not os.path.exists(model_dir):
        print(f"Model directory does not exist: {model_dir}")
        return
    model_cfg = {}
    with open(model_dir / "args.json", "r") as f:
        model_cfg = json.load(f)
    weights_file = model_dir / "weights.pth"
    label_classes = args.labels if args.labels else model_cfg["labels"]
    
    dl_workers = multiprocessing.cpu_count() if not args.src.lower().endswith(".zip") else 1
    
    src_dir = pathlib.Path(args.src)
    dataset = ClassificationDataSet(args.src, resolution=model_cfg["img_res"], mode="gray", inference=True)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=dl_workers, drop_last=False)
    
    model, name = classifier.load_classification_model(model_cfg, device, weights_file)
    
    print(f"Labeling {len(dataset)} images:")
    
    labels = {}
    for batch, x in tqdm(enumerate(data_loader), total=int(math.ceil(len(dataset) / args.batch_size))):
        x = x.to(device)
        y_pred = model(x)
        
        idx_start = batch * args.batch_size
        for i in range(y_pred.shape[0]):
            idx = i + idx_start
            if idx > len(dataset):
                break
                
            image_file_rel = dataset.get_image_path(idx)
            labels[image_file_rel] = {}
            
            for cls_idx, cls in enumerate(label_classes):
                if cls_idx > y_pred.shape[1]:
                    break
                labels[image_file_rel][cls] = y_pred[i, cls_idx].item() > 0.5
                
    dst_file = pathlib.Path(args.dst) if args.dst else src_dir / "labels_predicted.json"
    with open(dst_file, "w+") as f:
        json.dump(labels, f, indent=(2 if args.prettyprint else None))
    
    print(dst_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-l", "--labels", nargs='+', default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-s", "--src", type=str)
    parser.add_argument("-d", "--dst", type=str, default=None)
    parser.add_argument("--prettyprint", action="store_true", default=False)
    args = parser.parse_args()

    gc.collect()
    torch.cuda.empty_cache()

    main(args)
