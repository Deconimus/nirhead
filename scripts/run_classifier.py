import os, gc, argparse, pathlib, multiprocessing, json, math
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from nirhead.models import classifier
from nirhead.dataset.classification_dataset import ClassificationDataSet

from label_accuracy import evaluate_accuracy


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert (device == "cuda:0")
    
    model_dir = pathlib.Path(args.model)
    if not os.path.exists(model_dir):
        print(f"Model directory does not exist: {model_dir}")
        return
    with open(model_dir / "args.json", "r") as f:
        model_cfg = classifier.ClassifierConfig.from_json(json.load(f))
    weights_file = model_dir / "weights.pth"
    label_classes = args.labels if args.labels else model_cfg.labels
    
    dl_workers = multiprocessing.cpu_count() if not args.src.lower().endswith(".zip") else 1
    
    dataset = ClassificationDataSet(args.src, resolution=model_cfg.img_res, mode="gray", inference=True)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=dl_workers, drop_last=False)
    
    model, name = classifier.load_classification_model(model_cfg, device, weights_file)
    
    print(f"Labeling {len(dataset)} images:")
    
    labels = predict_labels(model, data_loader, label_classes, device, args.batch_size)
    
    if args.dst:
        dst_file = pathlib.Path(args.dst)
        with open(dst_file, "w+") as f:
            json.dump(labels, f, indent=(2 if args.prettyprint else None))
        print(dst_file)
    
    if args.eval_gt:
        with open(pathlib.Path(args.eval_gt), "r") as f:
            labels_gt = json.load(f)
        evaluate_accuracy(gt=labels_gt, pred=labels, filter=True)
    
    return labels
    

def predict_labels(model, data_loader, label_classes, device, batch_size):
    labels = {}
    for batch, x in tqdm(enumerate(data_loader), total=int(math.ceil(len(data_loader.dataset) / batch_size))):
        x = x.to(device)
        y_pred = model(x)

        idx_start = batch * batch_size
        for i in range(y_pred.shape[0]):
            idx = i + idx_start
            if idx > len(data_loader.dataset):
                break

            image_file_rel = data_loader.dataset.get_image_path(idx)
            labels[image_file_rel] = {}

            for cls_idx, cls in enumerate(label_classes):
                if cls_idx > y_pred.shape[1]:
                    break
                labels[image_file_rel][cls] = y_pred[i, cls_idx].item() > 0.5
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-l", "--labels", nargs='+', default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-s", "--src", type=str)
    parser.add_argument("-d", "--dst", type=str)
    parser.add_argument("--prettyprint", action="store_true", default=False)
    parser.add_argument("--eval_gt", type=str)
    args = parser.parse_args()

    gc.collect()
    torch.cuda.empty_cache()

    main(args)
