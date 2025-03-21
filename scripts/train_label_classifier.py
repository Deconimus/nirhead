import os, gc, argparse, pathlib, multiprocessing, json, time
import torch, torchvision, torchsummary
from torch import nn
from timeit import default_timer as timer
import numpy as np

from torch.utils.data import DataLoader

from gghead.models.classifier import ResNet, TinyVGG
from gghead.dataset.classification_dataset import ClassificationDataSet


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert(device == "cuda:0")

    dl_workers = multiprocessing.cpu_count() if not args.dataset.lower().endswith(".zip") else 1

    trainset = ClassificationDataSet(args.dataset, subdir="train", resolution=args.img_res, mode="gray", labelclasses=args.labels, flip=args.flip_aug)
    testset = ClassificationDataSet(args.dataset, subdir="test", resolution=args.img_res, mode="gray", labelclasses=args.labels)
    dl_train = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=dl_workers, drop_last=True)
    dl_test = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=dl_workers, drop_last=True)

    print(f"Trainset size: {len(trainset)} ({len(trainset) // args.batch_size} batches of {args.batch_size})")
    print(f"Testset size: {len(testset)} ({len(testset) // args.batch_size} batches of {args.batch_size})")

    if args.model.lower() == "resnet":
        model = ResNet(img_res=args.img_res, img_ch=1, num_classes=len(args.labels), crop=(args.crop_h, 0), ch_base=args.ch_base, epilogue=args.epilogue).to(device)
        name = "resnet_chbase" + str(args.ch_base) + ("_epilogue" if args.epilogue else "") + (f"_croph{args.crop_h}" if args.crop_h else "") + "_" + str(int(time.time()))
    elif args.model.lower() == "tvgg" or args.model.lower() == "tinyvgg":
        model = TinyVGG(img_res=args.img_res, input_channel=1, hidden_units=args.ch_base, num_classes=len(args.labels), crop=(args.crop_h, 0)).to(device)
        name = "tinyvgg_units" + str(args.ch_base) + (f"_croph{args.crop_h}" if args.crop_h else "") + "_" + str(int(time.time()))
    #print(model)
    #print("Total parameters: "+str(sum(p.numel() for p in model.parameters())))
    #print(torchsummary.summary(model, input_size=(1, 128, 128)))

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

    train_time_start = timer()

    data = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    try:
        for epoch in range(args.epochs):
            train_loss, train_acc = train_step(dl_train, model, loss_fn, optimizer, device)
            test_loss, test_acc = test_step(dl_test, model, loss_fn, device)

            print(f"Epoch {epoch:04} | Train: (loss={train_loss:.6f}, acc={train_acc:.3f}) | Test (loss={test_loss:.6f}, acc={test_acc:.3f})"+
                  f" | CUDA alloc: {torch.cuda.memory_allocated(0)/(2**30):.3f}GB, rsrvd: {torch.cuda.memory_reserved(0)/(2**30):.3}GB")

            data["train_loss"].append(float(train_loss))
            data["train_acc"].append(float(train_acc))
            data["test_loss"].append(float(test_loss))
            data["test_acc"].append(float(test_acc))

            if args.stop_at_acc and test_acc >= args.stop_at_acc:
                print(f"Test accuracy goal reached, stopping training at test_acc={test_acc}")
                break
            if (epoch+1) % 100 == 0 and args.logdir:
                save_log(data, args.logdir, name)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: cancelling further training, saving logs.")

    train_time_end = timer()
    print(f"Train time on {device}: {(train_time_end-train_time_start):.2f}s")
    print(f"Best test accuracy: {max(data['test_acc'])} (epoch {data['test_acc'].index(max(data['test_acc']))})")

    if args.logdir:
        save_log(data, args.logdir, name)

    if args.savedir:
        # save model weights
        savedir = pathlib.Path(args.savedir) / name
        os.makedirs(savedir, exist_ok=True)
        weights_file = savedir / ("weights"+ ".pth")
        torch.save(model.state_dict(), weights_file)
        print("Saved model: "+str(weights_file))

        # save model arguments
        args_file = savedir / "args.json"
        with open(args_file, "w+") as f:
            json.dump(vars(args), f, indent=2)
        print("Saved model arguments: " + str(args_file))

def train_step(data_loader, model, loss_fn, optimizer, device):
    train_loss, train_acc = 0.0, 0.0

    for batch, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)

        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += calc_accuracy(y_true=y, y_pred=y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc  /= len(data_loader)
    return train_loss, train_acc


def test_step(data_loader, model, loss_fn, device):
    test_loss, test_acc = 0.0, 0.0
    model.eval() # evaluation mode

    with torch.inference_mode():
        for batch, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)

            y_pred = model(x)

            test_loss += loss_fn(y_pred, y)
            test_acc += calc_accuracy(y_true=y, y_pred=y_pred)

        test_loss /= len(data_loader)
        test_acc  /= len(data_loader)
        return test_loss, test_acc


def calc_accuracy(y_true, y_pred):
    pred_onehot = (y_pred >= 0.5)
    correct = torch.eq(y_true, pred_onehot).sum().item()
    acc = (correct / y_pred.numel())
    return acc


def save_log(data, logdir, name):
    logdir = pathlib.Path(args.logdir)
    os.makedirs(logdir, exist_ok=True)
    dst_file = logdir / (name + ".json")
    with open(dst_file, "w+") as f:
        json.dump(data, f, indent=2)
    print("Saved log: "+str(dst_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("--resume", type=str)
    parser.add_argument("-l", "--labels", nargs='+', required=True)
    parser.add_argument("--img_res", type=int, default=128)
    parser.add_argument("--crop_h", type=int, default=0)
    parser.add_argument("--ch_base", type=int, default=32)
    parser.add_argument("--epilogue", action="store_true", default=False)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--savedir", type=str)
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--flip_aug", action="store_true", default=False)
    parser.add_argument("--stop_at_acc", type=float)
    args = parser.parse_args()

    gc.collect()
    torch.cuda.empty_cache()

    main(args)
