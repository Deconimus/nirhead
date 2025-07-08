import os, gc, argparse, pathlib, multiprocessing, json, math, PIL.Image, PIL.ImageOps
import torch, torchvision
from tqdm import tqdm
from torch import nn
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from dataclasses import asdict
from dreifus.image import normalized_torch_to_numpy_img

from gghead.model_manager.finder import find_model_manager

from nirhead.models import classifier, identifier
from nirhead.dataset.identification_dataset import IdentificationDataSet
import nirhead.data.static_attributes as stat


REAL_LABEL = 1.0
FAKE_LABEL = 0.0


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert (device == "cuda:0")
    
    dl_workers = multiprocessing.cpu_count() if not args.dataset.lower().endswith(".zip") else 1
    #label_classes = stat.normalize_attributes_list(args.labels)
    #args.labels = label_classes
    
    assert (os.path.exists(args.dataset))
    
    trainset = IdentificationDataSet(args.dataset, subdir="train", resolution=args.img_res, mode="gray", flip=False, strict_pose=True)#, labelclasses=label_classes)
    testset = IdentificationDataSet(args.dataset, subdir="test", resolution=args.img_res, mode="gray", inference=True, flip=False)#, labelclasses=label_classes)
    dl_train = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=dl_workers, drop_last=True)
    dl_test = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=dl_workers, drop_last=False)
    
    print(f"Trainset size: {len(trainset)} ({len(trainset) // args.batch_size} batches of {args.batch_size})")
    #print(f"Testset size: {len(testset)} ({len(testset) // args.batch_size} batches of {args.batch_size})")
    
    if args.resume:
        model, name = identifier.load_identifier_model_dir(args.resume, device)
        with open(pathlib.Path(args.resume) / "args.json", "r") as f:
            model_cfg = identifier.IdentifierConfig.from_json(json.load(f))
        print(f"Resuming {str(pathlib.Path(args.resume))}")
    else:
        model_cfg = identifier.IdentifierConfig.from_dict(dict(vars(args)))
        model, name = identifier.load_identifier_model(model_cfg, device)
    
    model_manager = find_model_manager(args.synth_model)
    checkpoint = model_manager._resolve_checkpoint_id(-1)
    print(f"Loading {args.synth_model} at checkpoint {checkpoint}")
    nirhead_model = model_manager.load_checkpoint(checkpoint, load_ema=True).to(device)
    nirhead_model.requires_grad_(False)
    
    if args.dst is not None:
        dst_dir = pathlib.Path(args.dst)
    else:
        dst_dir = pathlib.Path(args.savedir) / name
    grids_dir = dst_dir / "grids"
    
    optimizerG = torch.optim.Adam(params=model.G.parameters(), lr=0.0001)
    optimizerD = torch.optim.Adam(params=model.D.parameters(), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    train_time_start = timer()
    
    data = {"train_loss_D": [], "train_loss_G": [], "train_acc": [], "train_D_x": [], "train_D_G_z1": [], "train_D_G_z2": [], "test_loss_D": [], "test_loss_G": [], "test_acc": [], "trainsets": []}
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
        test_grid(0, dl_test, model, nirhead_model, grids_dir, device, args)
        for epoch in range(args.epochs):
            #with tqdm(total=(len(trainset) // args.batch_size) + int(math.ceil(len(testset) / args.batch_size)), leave=False) as pbar:
            with tqdm(total=len(trainset) // args.batch_size, leave=False) as pbar:
                train_loss_D, train_loss_G, train_acc, train_D_x, train_D_G_z1, train_D_G_z2 = train_step(dl_train, model, nirhead_model, optimizerG, optimizerD, criterion, device, args, pbar)
                #test_loss_D, test_loss_G, test_acc = test_step(dl_test, model, nirhead_model, device, pbar)
                test_grid(epoch+1, dl_test, model, nirhead_model, grids_dir, device, args)
            
            print(
                f"Epoch {epoch:04} | Train: (loss_D={train_loss_D:.6f}, loss_G={train_loss_G:.6f}, acc={train_acc:.6f}, D_x={train_D_x:.6f}, D_G_z1={train_D_G_z1:.6f}, D_G_z2={train_D_G_z2:.6f})" +
                #f" | Test (loss_D={test_loss_D:.6f}, loss_G={test_loss_G:.6f}, acc={test_acc:.6f}))" +
                f" | CUDA alloc: {torch.cuda.memory_allocated(0) / (2 ** 30):.3f}GB, rsrvd: {torch.cuda.memory_reserved(0) / (2 ** 30):.3}GB")
            
            data["train_loss_D"].append(float(train_loss_D))
            data["train_loss_G"].append(float(train_loss_G))
            data["train_acc"].append(float(train_acc))
            data["train_D_x"].append(float(train_D_x))
            data["train_D_G_z1"].append(float(train_D_G_z1))
            data["train_D_G_z2"].append(float(train_D_G_z2))
            #data["test_loss_D"].append(float(test_loss_D))
            #data["test_loss_G"].append(float(test_loss_G))
            #data["test_acc"].append(float(test_acc))
            
            if args.stop_at_train_loss and train_loss <= args.stop_at_train_loss:
                print(f"Train loss goal reached, stopping training at train_loss={train_loss}")
                break
            #if args.stop_at_test_loss and test_loss <= args.stop_at_test_loss:
            #    print(f"Test loss goal reached, stopping training at test_loss={test_loss}")
            #    break
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


def train_step(data_loader, model, nirhead_model, optimizerG, optimizerD, criterion, device, args, pbar):
    train_loss_D, train_loss_G, train_acc, train_D_x, train_D_G_z1, train_D_G_z2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for batch, (real_imgs, subj_labels, real_c) in enumerate(data_loader):
        real_imgs = real_imgs.to(device)
        real_c = real_c.to(device)
        
        # see: https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        
        optimizerD.zero_grad()
        
        # -- D train step --
        # real image batch:
        
        label = torch.full((real_imgs.size(0),), REAL_LABEL, dtype=torch.float32, device=device)
        
        logits = model.discriminate(real_imgs, subj_labels)
        loss_Dreal = criterion(logits, label)
        loss_Dreal.backward()
        D_x = logits.mean().item()
        
        # fake image batch:
        
        z, fake_c = model(real_imgs)
        
        with torch.no_grad():
            w = nirhead_model.mapping(z, real_c, None, truncation_psi=0.7)
            nirhead_output = nirhead_model.synthesis(w, real_c, noise_mode='const', return_masks=False,
                                             neural_rendering_resolution=args.img_res,
                                             return_uv_map=False)
            fake_imgs = nirhead_output["image"]
            fake_imgs = torchvision.transforms.functional.rgb_to_grayscale(fake_imgs, num_output_channels=1)
        
        label.fill_(FAKE_LABEL)
        
        logits = model.discriminate(fake_imgs.detach(), subj_labels)
        loss_Dfake = criterion(logits, label)
        loss_Dfake.backward()
        
        D_G_z1 = logits.mean().item()
        loss_D = loss_Dreal + loss_Dfake
        
        optimizerD.step()
        
        # -- G train step --
        
        optimizerG.zero_grad()
        label.fill_(REAL_LABEL)
        
        logits = model.discriminate(fake_imgs, subj_labels)
        loss_G = criterion(logits, label)
        loss_G += torch.nn.functional.mse_loss(fake_c, real_c)
        loss_G.backward()
        
        D_G_z2 = logits.mean().item()
        
        optimizerG.step()
        
        train_loss_D += loss_D.item()
        train_loss_G += loss_G.item()
        train_D_x += D_x
        train_D_G_z1 += D_G_z1
        train_D_G_z2 += D_G_z2
        
        pbar.update(1)
    
    train_loss_D /= len(data_loader)
    train_loss_G /= len(data_loader)
    train_acc /= len(data_loader)
    train_D_x /= len(data_loader)
    train_D_G_z1 /= len(data_loader)
    train_D_G_z2 /= len(data_loader)
    return train_loss_D, train_loss_G, train_acc, train_D_x, train_D_G_z1, train_D_G_z2


def test_step(data_loader, model, nirhead_model, device, pbar):
    test_loss_D, test_loss_G, test_acc = 0.0, 0.0, 0.0
    model.eval()  # evaluation mode
    
    with torch.inference_mode():
        for batch, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            
            #y_pred = model(x)
            
            #test_loss += stat.attributes_loss(y_pred, y, loss_type=loss_type)
            #test_acc += calc_accuracy(y_pred=y_pred, y_true=y, static_attributes=static_attributes)
            
            pbar.update(1)
        
        test_loss_D /= len(data_loader)
        test_loss_G /= len(data_loader)
        test_acc /= len(data_loader)
        return test_loss_D, test_loss_G, test_acc
    

def test_grid(epoch, data_loader, model, nirhead_model, grids_dir, device, args):
    grid_real_imgs = []
    grid_fake_imgs_realc = []
    grid_fake_imgs_fakec = []
    
    with torch.inference_mode():
        for batch, (real_imgs, real_c) in enumerate(data_loader):
            real_imgs = real_imgs.to(device)
            real_c = real_c.to(device)
            
            z, fake_c = model(real_imgs)
            
            grid_real_imgs += [PIL.Image.fromarray(normalized_torch_to_numpy_img(real_imgs[i].repeat(3, 1, 1))) for i in range(real_imgs.shape[0])]
            
            # images synthesized with generated pose c
            w = nirhead_model.mapping(z, fake_c, None, truncation_psi=0.7)
            nirhead_output = nirhead_model.synthesis(w, fake_c, noise_mode='const', return_masks=False,
                                                     neural_rendering_resolution=args.img_res,
                                                     return_uv_map=False)
            fake_imgs = nirhead_output["image"]
            grid_fake_imgs_fakec += [PIL.ImageOps.grayscale(PIL.Image.fromarray(normalized_torch_to_numpy_img(fake_imgs[i]))) for i in range(fake_imgs.shape[0])]
            
            # images synthesized with real pose c
            w = nirhead_model.mapping(z, real_c, None, truncation_psi=0.7)
            nirhead_output = nirhead_model.synthesis(w, real_c, noise_mode='const', return_masks=False,
                                                     neural_rendering_resolution=args.img_res,
                                                     return_uv_map=False)
            fake_imgs = nirhead_output["image"]
            grid_fake_imgs_realc += [PIL.ImageOps.grayscale(PIL.Image.fromarray(normalized_torch_to_numpy_img(fake_imgs[i]))) for i in range(fake_imgs.shape[0])]
    
    os.makedirs(grids_dir, exist_ok=True)
    num_cols, num_rows = 16, 4
    
    for grid_fake_imgs, infix in [(grid_fake_imgs_realc, "real_c"), (grid_fake_imgs_fakec, "fake_c")]:
        grid_image = PIL.Image.new("L", (num_cols * args.img_res, num_rows * 2 * args.img_res), 0)
        for idx, (real_img, fake_img) in enumerate(zip(grid_real_imgs, grid_fake_imgs)):
            grid_image.paste(real_img, box=((idx % num_cols) * args.img_res, (idx // num_cols) * 2 * args.img_res))
            grid_image.paste(fake_img, box=((idx % num_cols) * args.img_res, ((idx // num_cols) * 2 + 1) * args.img_res))
        
        grid_image.save(grids_dir / ("grid_"+infix+"_"+str(epoch).zfill(4)+".png"))
    

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
    parser.add_argument("--synth_model", type=str, default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--img_res", type=int, default=256)
    parser.add_argument("--subject_hash", type=str, default="binary", choices=["binary", "sha256"])
    #parser.add_argument("--flip_aug", action="store_true", default=False)
    parser.add_argument("--loss", type=str, default="mixed")
    parser.add_argument("--savedir", type=str, default="/mnt/g/FacesNIR/models/identifier")
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
