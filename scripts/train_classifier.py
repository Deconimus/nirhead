import os, gc, argparse, pathlib, multiprocessing, json, math
import torch
from tqdm import tqdm
from torch import nn
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from dataclasses import asdict

from nirhead.models import classifier
from nirhead.dataset.classification_dataset import ClassificationDataSet

from run_classifier import predict_labels
from label_accuracy import evaluate_accuracy

import nirhead.data.static_attributes as stat


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert(device == "cuda:0")

    dl_workers = multiprocessing.cpu_count() if not args.dataset.lower().endswith(".zip") else 1
    label_classes = stat.normalize_attributes_list(args.labels)
    
    assert(os.path.exists(args.dataset))
    
    trainset = ClassificationDataSet(args.dataset, subdir="train", resolution=args.img_res, mode="gray", labelclasses=label_classes, flip=args.flip_aug)
    testset = ClassificationDataSet(args.dataset, subdir="test", resolution=args.img_res, mode="gray", labelclasses=label_classes)
    dl_train = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=dl_workers, drop_last=True)
    dl_test = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=dl_workers, drop_last=False)

    print(f"Trainset size: {len(trainset)} ({len(trainset) // args.batch_size} batches of {args.batch_size})")
    print(f"Testset size: {len(testset)} ({len(testset) // args.batch_size} batches of {args.batch_size})")
    
    model_cfg = classifier.ClassifierConfig.from_dict(dict(vars(args)))
    model, name = classifier.load_classification_model(model_cfg, device)
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

    train_time_start = timer()

    data = {"train_loss": [], "train_acc": [], "train_mse": [], "test_loss": [], "test_acc": [], "test_mse": []}

    try:
        for epoch in range(args.epochs):
            with tqdm(total=(len(trainset) // args.batch_size)+int(math.ceil(len(testset) / args.batch_size)), leave=False) as pbar:
                train_loss, train_acc, train_mse, train_bce = train_step(dl_train, model, optimizer, label_classes, device, pbar)
                test_loss, test_acc, test_mse, test_bce = test_step(dl_test, model, label_classes, device, pbar)

            print(f"Epoch {epoch:04} | Train: (loss={train_loss:.6f}, acc={train_acc:.3f}, mse={train_mse:.6f}, bce={train_bce:.6f})"+
                  f" | Test (loss={test_loss:.6f}, acc={test_acc:.3f}, mse={test_mse:.6f}, bce={test_bce:.6f})"+
                  f" | CUDA alloc: {torch.cuda.memory_allocated(0)/(2**30):.3f}GB, rsrvd: {torch.cuda.memory_reserved(0)/(2**30):.3}GB")
            
            data["train_loss"].append(float(train_loss))
            data["train_acc"].append(float(train_acc))
            data["train_mse"].append(float(train_mse))
            data["train_mse"].append(float(train_bce))
            data["test_loss"].append(float(test_loss))
            data["test_acc"].append(float(test_acc))
            data["test_mse"].append(float(test_mse))
            data["train_mse"].append(float(test_bce))

            if args.stop_at_acc and test_acc >= args.stop_at_acc:
                print(f"Test accuracy goal reached, stopping training at test_acc={test_acc}")
                break
            if args.stop_at_train_loss and train_loss <= args.stop_at_train_loss:
                print(f"Train loss goal reached, stopping training at train_loss={train_loss}")
                break
            if (epoch+1) % 100 == 0 and args.logdir:
                save_log(data, args.logdir, name)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: cancelling further training, saving logs.")

    train_time_end = timer()
    print(f"Train time on {device}: {(train_time_end-train_time_start):.2f}s")
    print(f"Best test accuracy: {max(data['test_acc'])} (epoch {data['test_acc'].index(max(data['test_acc']))})")

    model_class_concat = "_".join([stat.types[l].short for l in label_classes])
    
    if args.logdir:
        save_log(data, pathlib.Path(args.logdir) / model_class_concat, name)

    if args.savedir:
        # save model weights
        savedir = pathlib.Path(args.savedir) / model_class_concat / name
        os.makedirs(savedir, exist_ok=True)
        weights_file = savedir / ("weights"+ ".pth")
        torch.save(model.state_dict(), weights_file)
        print("Saved model: "+str(weights_file))

        # save model arguments
        args_file = savedir / "args.json"
        with open(args_file, "w+") as f:
            json.dump(model_cfg.to_json(), f, indent=2)
        print("Saved model arguments: " + str(args_file))
        
    if args.eval:
        print(f"Evaluating model on {args.eval[0]}:")
        eval_dataset = ClassificationDataSet(args.eval[0], resolution=args.img_res, mode="gray", inference=True)
        dl_eval = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=dl_workers, drop_last=False)
        
        labels_gt = {}
        with open(pathlib.Path(args.eval[1]), "r") as f:
            labels_gt = json.load(f)
        labels_pred = predict_labels(model, dl_eval, label_classes, device, args.batch_size)
        
        evaluate_accuracy(gt=labels_gt, pred=labels_pred, filter=True)
        

def train_step(data_loader, model, optimizer, static_attributes, device, pbar):
    train_loss, train_acc, train_mse, train_bce = 0.0, 0.0, 0.0, 0.0

    for batch, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)

        y_pred = model(x)

        loss = stat.attributes_loss(y_pred, y, static_attributes)
        train_loss += loss
        train_acc += calc_accuracy(y_pred=y_pred, y_true=y, static_attributes=static_attributes)
        
        with torch.no_grad():
            train_mse += calc_mse(y_pred=y_pred, y_true=y, static_attributes=static_attributes)
            train_bce += calc_bce(y_pred=y_pred, y_true=y, static_attributes=static_attributes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.update(1)
        
    train_loss /= len(data_loader)
    train_acc  /= len(data_loader)
    with torch.no_grad():
        train_mse /= len(data_loader)
        train_bce /= len(data_loader)
    return train_loss, train_acc, train_mse, train_bce


def test_step(data_loader, model, static_attributes, device, pbar):
    test_loss, test_acc, test_mse, test_bce = 0.0, 0.0, 0.0, 0.0
    model.eval() # evaluation mode

    with torch.inference_mode():
        for batch, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)

            y_pred = model(x)

            test_loss += stat.attributes_loss(y_pred, y, static_attributes)
            test_acc += calc_accuracy(y_pred=y_pred, y_true=y, static_attributes=static_attributes)
            test_mse += calc_mse(y_pred=y_pred, y_true=y, static_attributes=static_attributes)
            test_bce += calc_bce(y_pred=y_pred, y_true=y, static_attributes=static_attributes)
            
            pbar.update(1)
        
        test_loss /= len(data_loader)
        test_acc  /= len(data_loader)
        test_mse  /= len(data_loader)
        test_bce  /= len(data_loader)
        return test_loss, test_acc, test_mse, test_bce


def calc_accuracy(y_pred, y_true, static_attributes):
    if stat.get_num_binary_attributes(static_attributes) <= 0:
        return 0.0
    
    y_pred_binary = stat.take_binary_from_attribute_tensor(y_pred, static_attributes)
    y_true_binary = stat.take_binary_from_attribute_tensor(y_true, static_attributes)
    
    pred_onehot = (y_pred_binary >= 0.5)
    acc= 0.0
    correct = torch.eq(y_true_binary, pred_onehot)
    correct_ = correct[:,0]
    for i in range(1, correct.shape[1]):
        correct_ = torch.logical_and(correct_, correct[:,i])
    correct = correct_
    correct_num = correct.sum().item()
    acc = (correct_num / y_pred_binary.shape[0])
    return acc


def calc_mse(y_pred, y_true, static_attributes):
    num_attributes = len(static_attributes)
    num_binary_attributes = stat.get_num_binary_attributes(static_attributes)
    
    loss = 0.0
    idx_off = 0
    for attr in static_attributes:
        dim = stat.types[attr].dim
        if stat.types[attr].dtype == float or stat.types[attr].dtype == int:
            loss += torch.nn.functional.mse_loss(y_pred[:, idx_off:idx_off + dim], y_true[:, idx_off:idx_off + dim])
        idx_off += dim
    loss *= (num_attributes - num_binary_attributes) / num_attributes
    
    return loss


def calc_bce(y_pred, y_true, static_attributes):
    num_attributes = len(static_attributes)
    num_binary_attributes = stat.get_num_binary_attributes(static_attributes)
    
    lambda_binary = num_binary_attributes / num_attributes
    binary_loss = 0.0
    if num_binary_attributes > 0:
        binary_attr_pred = stat.take_binary_from_attribute_tensor(y_pred, static_attributes)
        binary_attr_truth = stat.take_binary_from_attribute_tensor(y_true, static_attributes)
        binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(binary_attr_pred, binary_attr_truth) * lambda_binary
        
    return binary_loss
    

def save_log(data, logdir, name):
    os.makedirs(logdir, exist_ok=True)
    dst_file = logdir / (name + ".json")
    with open(dst_file, "w+") as f:
        json.dump(data, f, indent=2)
    print("Saved log: "+str(dst_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("-l", "--labels", nargs='+', required=True)
    parser.add_argument("--img_res", type=int, default=128)
    parser.add_argument("--crop_h", type=int, default=0)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--flip_aug", action="store_true", default=False)
    
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--savedir", type=str)
    
    parser.add_argument("--resume", type=str)
    parser.add_argument("--eval", type=str, nargs=2)
    parser.add_argument("--stop_at_acc", type=float)
    parser.add_argument("--stop_at_train_loss", type=float)
    
    parser.add_argument("--ch_base", type=int, default=32)
    parser.add_argument("--epilogue", action="store_true", default=False)
    
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--vit_mlp_dim", type=int, default=None)
    parser.add_argument("--vit_dim", type=int, default=None)
    parser.add_argument("--vit_depth", type=int, default=4)
    parser.add_argument("--vit_heads", type=int, default=8)
    args = parser.parse_args()

    gc.collect()
    torch.cuda.empty_cache()

    main(args)
