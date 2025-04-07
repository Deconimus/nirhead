import os, argparse, pathlib, json
import torch, onnx, onnxscript
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataclasses import asdict

from nirhead.models import classifier
import nirhead.data.static_attributes as stat


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_dir = pathlib.Path(args.model)
    cfg_file = model_dir / "args.json"
    weights_file = model_dir / "weights.pth"
    
    with open(cfg_file, "r") as f:
        cfg = classifier.ClassifierConfig.from_json(json.load(f))
    model, model_name = classifier.load_classification_model(cfg, device, weights_file=weights_file)
    
    model_name = "_".join([stat.types[attr].short for attr in model.static_attributes])+"_"+model_name
    dst_file = pathlib.Path(".") / f"{model_name}.onnx"
    if args.dst:
        dst_file = pathlib.Path(args.dst)
    
    input_stub = (torch.randn(1,1,128,128).to(device),)
    #onnx_model = torch.onnx.export(
    torch.onnx.export(
        model=model,
        args=input_stub,
        f=dst_file,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    #onnx_model.optimize()
    #onnx_model.save(dst_file)
    print(dst_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-d", "--dst", type=str)
    args = parser.parse_args()
    
    main(args)
    