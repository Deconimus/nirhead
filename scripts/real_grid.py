import  json, argparse, pathlib, os, PIL.Image, PIL.ImageOps, random
import numpy as np
from tqdm import tqdm


def main(args):
    src_dir = pathlib.Path(args.src)
    dst_file = pathlib.Path(args.dst)
    if not args.dst.endswith(".png") and not args.dst.endswith(".jpg"):
        dst_file = pathlib.Path(args.dst) / ("grid_" + src_dir.parent.name + "_" + src_dir.name + ".png")
    
    all_images = list(src_dir.rglob("*.png"))
    random.shuffle(all_images)
    images = all_images[:args.cols * args.rows]
    
    img_grid = PIL.Image.new("L", size=(args.cols * args.res, args.rows * args.res))
    for idx, img_file in enumerate(tqdm(images)):
        img = PIL.Image.open(img_file)
        width, height = img.size
        if width != args.res or height != args.res:
            img = img.resize((args.res, args.res), PIL.Image.LANCZOS)
        img_grid.paste(img, box=((idx % args.cols) * args.res, (idx // args.cols) * args.res))
    
    os.makedirs(dst_file.parent, exist_ok=True)
    img_grid.save(dst_file)
    print(dst_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--res", type=int, default=128)
    args = parser.parse_args()
    
    main(args)
    