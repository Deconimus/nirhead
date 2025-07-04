import os, gc, argparse, pathlib, multiprocessing, json, math
import torch
from tqdm import tqdm
from torch import nn
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from dataclasses import asdict

from nirhead.models import classifier, identifier
import nirhead.data.static_attributes as stat

def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert (device == "cuda:0")
    
    dl_workers = multiprocessing.cpu_count() if not args.dataset.lower().endswith(".zip") else 1
    #label_classes = stat.normalize_attributes_list(args.labels)
    #args.labels = label_classes
    
    assert (os.path.exists(args.dataset))
    
    trainset = IdentifierDataSet(args.dataset, subdir="train", resolution=args.img_res, mode="gray")#, labelclasses=label_classes)
    testset = IdentifierDataSet(args.dataset, subdir="test", resolution=args.img_res, mode="gray")#, labelclasses=label_classes)
    dl_train = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=dl_workers, drop_last=True)
    dl_test = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=dl_workers, drop_last=False)
    
    print(f"Trainset size: {len(trainset)} ({len(trainset) // args.batch_size} batches of {args.batch_size})")
    print(f"Testset size: {len(testset)} ({len(testset) // args.batch_size} batches of {args.batch_size})")
    
    if args.resume:
        model, name = identifier.load_identifier_model_dir(args.resume, device)
        with open(pathlib.Path(args.resume) / "args.json", "r") as f:
            model_cfg = identifier.IdentifierConfig.from_json(json.load(f))
        print(f"Resuming {str(pathlib.Path(args.resume))}")
    else:
        model_cfg = identifier.IdentifierConfig.from_dict(dict(vars(args)))
        model, name = identifier.load_identifier_model(model_cfg, device)
    
    if args.dst is not None:
        dst_dir = pathlib.Path(args.dst)
    else:
        dst_dir = pathlib.Path(args.savedir) / name
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    train_time_start = timer()
    
    data = {"train_loss": [], "test_loss": [], "trainsets": []}
    if args.resume:
        history_file = pathlib.Path(args.resume) / "history.json"
        if not os.path.isfile(history_file) or not os.path.exists(history_file):
            print("Warning: Resuming without loaded training history!")
        else:
            with open(pathlib.Path(args.resume) / "history.json", "r") as f:
                data = json.load(f)
            print(f"Loaded history from {str(history_file)}")
    
    epochs_trained = 0
    try:
        for epoch in range(args.epochs):
            with tqdm(total=(len(trainset) // args.batch_size) + int(math.ceil(len(testset) / args.batch_size)), leave=False) as pbar:
                train_loss = train_step(dl_train, model, optimizer, device, pbar)
                test_loss = test_step(dl_test, model, device, pbar)
            
            print(
                f"Epoch {epoch:04} | Train: (loss={train_loss:.6f})" +
                f" | Test (loss={test_loss:.6f})" +
                f" | CUDA alloc: {torch.cuda.memory_allocated(0) / (2 ** 30):.3f}GB, rsrvd: {torch.cuda.memory_reserved(0) / (2 ** 30):.3}GB")
            
            data["train_loss"].append(float(train_loss))
            data["test_loss"].append(float(test_loss))
            
            if args.stop_at_train_loss and train_loss <= args.stop_at_train_loss:
                print(f"Train loss goal reached, stopping training at train_loss={train_loss}")
                break
            if args.stop_at_test_loss and test_loss <= args.stop_at_test_loss:
                print(f"Test loss goal reached, stopping training at test_loss={test_loss}")
                break
            if (epoch + 1) % 100 == 0 and (args.savedir or args.dst):
                save_log(data, dst_dir, "history")
            epochs_trained = epoch + 1
    except KeyboardInterrupt:
        print("KeyboardInterrupt: cancelling further training, saving logs.")
    
    train_time_end = timer()
    print(f"Train time on {device}: {(train_time_end - train_time_start):.2f}s")
    print(f"Best test accuracy: {max(data['test_acc'])} (epoch {data['test_acc'].index(max(data['test_acc']))})")
    
    trainset_relpath = str(pathlib.Path(args.dataset).absolute()).replace("\\", "/")
    if "EyesNIR/" in trainset_relpath:
        trainset_relpath = trainset_relpath[trainset_relpath.index("EyesNIR/") + len("EyesNIR/"):]
    elif "FacesNIR/" in trainset_relpath:
        trainset_relpath = trainset_relpath[trainset_relpath.index("FacesNIR/") + len("FacesNIR/"):]
    
    # save history
    if args.savedir or args.dst:
        data["trainsets"].append((trainset_relpath, epochs_trained))
        save_log(data, dst_dir, "history")
    
    # save weights
    if args.savedir or args.dst:
        # save model weights
        os.makedirs(dst_dir, exist_ok=True)
        weights_file = dst_dir / "weights.pth"
        torch.save(model.state_dict(), weights_file)
        print("Saved model: " + str(weights_file))
        
        # save model arguments
        model_cfg.trainsets.append((trainset_relpath, epochs_trained))  # some data redundancy for safety & convenience
        args_file = dst_dir / "args.json"
        with open(args_file, "w+") as f:
            json.dump(model_cfg.to_json(), f, indent=2)
        print("Saved model arguments: " + str(args_file))


def save_log(data, logdir, name):
    os.makedirs(logdir, exist_ok=True)
    dst_file = logdir / (name + ".json")
    with open(dst_file, "w+") as f:
        json.dump(data, f, indent=2)
    print("Saved log: "+str(dst_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    #parser.add_argument("--flip_aug", action="store_true", default=False)
    parser.add_argument("--loss", type=str, default="mixed")
    parser.add_argument("--savedir", type=str)
    parser.add_argument("--dst", type=str, default=None)
    
    # parser.add_argument("--no_train", type=str)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--stop_at_train_loss", type=float)
    parser.add_argument("--stop_at_test_loss", type=float)
    
    #parser.add_argument("--patch_size", type=int, default=16)
    #parser.add_argument("--vit_mlp_dim", type=int, default=1024)
    #parser.add_argument("--vit_dim", type=int, default=None)
    #parser.add_argument("--vit_depth", type=int, default=4)
    #parser.add_argument("--vit_heads", type=int, default=8)
    args = parser.parse_args()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    main(args)
