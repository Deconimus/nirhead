import argparse, pathlib, json, os
import numpy as np


def main(args):
    gt = {}
    with open(pathlib.Path(args.gt), "r") as f:
        gt = json.load(f)
    pred = {}
    with open(pathlib.Path(args.pred), "r") as f:
        pred = json.load(f)
        
    labels = args.labels

    evaluate_accuracy(gt, pred, labels=labels, filter=args.namefilter)


def evaluate_accuracy(gt, pred, labels=None, filter=False, noprint=False):
    if not labels:
        labels = list(pred[list(pred.keys())[0]].keys())

    correct, n = 0, 0
    for img in pred.keys():
        img_gt = namefilter(img) if filter else img
        if not img_gt in gt.keys(): continue

        eq = all([gt[img_gt][cls] == pred[img][cls] for cls in labels])
        if eq:
            correct += 1
        n += 1

    if n <= 0:
        if not noprint:
            print("Error: no key matches.")
        return

    acc = correct / n
    if not noprint:
        print(f"Accuracy over {n} labeled images is {acc * 100.0:.2f}%")
    return acc


def namefilter(s):
    return s.replace("_l", "").replace("_r", "").replace("_mirr", "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--labels", type=str, default=None)
    parser.add_argument("--namefilter", action="store_true", default=False)
    args = parser.parse_args()
    
    main(args)