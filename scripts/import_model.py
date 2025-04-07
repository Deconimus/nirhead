import argparse, pathlib, zipfile, os, shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def main(args):
    src_dir = pathlib.Path(args.src) / args.model
    dst_dir = pathlib.Path(args.dst) / args.model[5:] if args.model.lower().startswith("kw") else args.model
    
    if not os.path.isdir(src_dir):
        print(f"Error: \"{src_dir}\" not found.")
        return
    
    files = [f for f in src_dir.glob("*") if not str(f).lower().endswith(".png")]
    files += list((src_dir / "evaluations").rglob("*"))
    files += [f for f in (src_dir / "classifier").rglob("*") if not str(f).lower().endswith(".pth")]
    
    if args.all or args.images:
        files += list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.bmp"))
    else:
        images = list(src_dir.glob("fakes*.png"))
        images.sort()
        if len(images) > 0:
            files += images[-2:]
    
    get_checkpoint_nr = lambda f: int(f.name[f.name.index("-") + 1:f.name.rindex(".")])
    
    checkpoints = list((src_dir / "checkpoints").rglob("*.pkl"))
    checkpoints.sort(key=get_checkpoint_nr)
    if not args.all:
        if args.checkpoints is not None:
            chks = [chk for chk in checkpoints if get_checkpoint_nr(chk) in args.checkpoints]
            if -1 in args.checkpoints and not checkpoints[-1] in chks:
                chks += [checkpoints[-1]]
            checkpoints = chks
        else:
            checkpoints = [checkpoints[-1]]
    files += checkpoints
    
    for cls_dir in (src_dir / "classifier").glob("*"):
        if not os.path.isdir(cls_dir):
            continue
        cls_checkpoints = list((cls_dir / "checkpoints").rglob("*.pth"))
        cls_checkpoints.sort(key=get_checkpoint_nr)
        if not args.all:
            if args.checkpoints is not None:
                chks = [chk for chk in cls_checkpoints if get_checkpoint_nr(chk) in args.checkpoints]
                if -1 in args.checkpoints and not cls_checkpoints[-1] in chks:
                    chks += [cls_checkpoints[-1]]
                cls_checkpoints = chks
            else:
                cls_checkpoints = [cls_checkpoints[-1]]
        files += cls_checkpoints
    
    with tqdm(total=len(files)) as pbar:
        with ThreadPoolExecutor(max_workers=args.threads) as xec:
            for file in files:
                relpath = str(file.absolute())[len(str(src_dir.absolute())) + 1:]
                dst_file = dst_dir / relpath
                os.makedirs(dst_file.parent, exist_ok=True)
                
                xec.submit(copy_file, file, dst_file, args.force_overwrite, pbar)
    
    print(dst_dir.absolute())


def copy_file(src_file, dst_file, force_overwrite, pbar=None):
    newer = (not os.path.exists(dst_file)) or os.path.getmtime(src_file) > os.path.getmtime(dst_file)
    if newer or force_overwrite:
        shutil.copy2(src_file, dst_file)
    
    if pbar is not None:
        pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-c", "--checkpoints", type=int, nargs="+", default=None)
    parser.add_argument("-s", "--src", type=str, default="/mnt/cccavefs/ccmlops/2024-11-07-MA_PascalSielski/gghead_results")
    parser.add_argument("-d", "--dst", type=str, default="models/gghead")
    parser.add_argument("-a", "--all", action="store_true", default=False)
    parser.add_argument("-i", "--images", action="store_true", default=False)
    parser.add_argument("-f", "--force_overwrite", action="store_true", default=False)
    parser.add_argument("-t", "--threads", type=int, default=8)
    args = parser.parse_args()
    
    main(args)
