import pathlib, json, argparse, os

def main(args):

    if os.path.exists("models/gghead/"+args.model):
        src = pathlib.Path("models/gghead/"+args.model+"/evaluations")
    else:
        src = pathlib.Path(args.model+"/evaluations")
        
    eval_files = sorted(list(src.rglob("evaluation_ckpt-*_ema.json")))
    
    fids = []
    for file in eval_files:
        with open(file, "r") as f:
            data = json.load(f)
        for k in data.keys():
            if data[k] is not None:
                fids.append(float(data[k]))
                break
    
    fids = sorted(fids)
    print(f"Min FID: {fids[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    args = parser.parse_args()
    
    main(args)
    