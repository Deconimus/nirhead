import os, gc, argparse, pathlib, multiprocessing, json, math
import torch
from tqdm import tqdm
from torch import nn
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from dataclasses import asdict

from nirhead.models import classifier
from nirhead.dataset.classification_dataset import ClassificationDataSet

from run_classifier_old import predict_labels
from label_accuracy import evaluate_accuracy

import nirhead.data.static_attributes as stat


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert (device == "cuda:0")
    dl_workers = multiprocessing.cpu_count() if not args.dataset.lower().endswith(".zip") else 1
    assert (os.path.exists(args.dataset))
    
    dst_dir = pathlib.Path(args.dst)
    os.makedirs(dst_dir, exist_ok=True)
    
    testset = ClassificationDataSet(args.dataset, subdir="test", resolution=args.img_res, mode="gray", labelclasses=["gaze"])
    dl_test = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=dl_workers, drop_last=False)
    
    print(f"Testset size: {len(testset)} ({len(testset) // args.batch_size} batches of {args.batch_size})")
    
    model, name = classifier.load_classification_model_dir(args.model, device)
    with open(pathlib.Path(args.model) / "args.json", "r") as f:
        model_cfg = classifier.ClassifierConfig.from_json(json.load(f))
    print(f"Loaded {str(pathlib.Path(args.model))}")
    
    with tqdm(total=len(dl_test)) as pbar:
        mean_rmse, mean_pitch_rmse, mean_yaw_rmse, rmse_tuples = test_step(dl_test, model, device, pbar)
    
    print(f"Mean RMSE: {mean_rmse}, Mean Pitch RMSE: {mean_pitch_rmse}, Mean Yaw RMSE: {mean_yaw_rmse}")
    #print(mae_tuples)
    
    out_data = { "mean_rmse": mean_rmse.item(), "mean_pitch_rmse": mean_pitch_rmse.item(), "mean_yaw_rmse": mean_yaw_rmse.item(), "rmse_tuples": rmse_tuples }
    
    with open(dst_dir / ("gaze_analysis_" + pathlib.Path(args.model).name + ".json"), "w+") as f:
        json.dump(out_data, f, indent=2)
    
    
def test_step(data_loader, model, device, pbar):
    mean_rmse, mean_pitch_rmse, mean_yaw_rmse  = 0.0, 0.0, 0.0
    rmse_tuples = []
    model.eval()  # evaluation mode
    
    with torch.inference_mode():
        for batch, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            
            y_pred = model(x)
            
            mean_rmse += torch.sqrt(torch.nn.functional.mse_loss(y_pred, y))
            mean_pitch_rmse += torch.sqrt(torch.nn.functional.mse_loss(y_pred[:, 0], y[:, 0]))
            mean_yaw_rmse += torch.sqrt(torch.nn.functional.mse_loss(y_pred[:,1], y[:,1]))
            
            mae_local = torch.sqrt(torch.nn.functional.mse_loss(y_pred, y, reduction="none"))
            mae_local = torch.mean(mae_local, dim=1)
            
            y = y.cpu()
            rmse_samples = mae_local.cpu()
            
            for i in range(mae_local.shape[0]):
                rmse_tuples.append([y[i].tolist(), rmse_samples[i].item()])
            
            pbar.update(1)
        
        mean_rmse /= len(data_loader)
        mean_pitch_rmse /= len(data_loader)
        mean_yaw_rmse /= len(data_loader)
        
        return mean_rmse, mean_pitch_rmse, mean_yaw_rmse, rmse_tuples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("--dst", type=str, default=None, required=True)
    parser.add_argument("--model", type=str, default=None, required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("--img_res", type=int, default=128)
    args = parser.parse_args()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    main(args)
