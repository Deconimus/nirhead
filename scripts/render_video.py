import argparse, pathlib, os, cv2, PIL.Image, PIL.ImageOps, math, random, imageio_ffmpeg
import torch
import numpy as np
from tqdm import tqdm

from dreifus.camera import PoseType, CameraCoordinateConvention
from dreifus.image import normalized_torch_to_numpy_img
from dreifus.matrix import Pose
from eg3d.datamanager.nersemble import encode_camera_params
from gaussian_splatting.arguments import PipelineParams2
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.cameras import pose_to_rendercam
from eg3d.training.training_loop import setup_snapshot_image_grid, save_image_grid

from gghead.model_manager.finder import find_model_manager
from gghead.constants import DEFAULT_INTRINSICS
from gghead.dataset.image_folder_dataset import GGHeadImageFolderDataset, GGHeadImageFolderDatasetConfig

import nirhead.data.static_attributes as stat


if os.name == 'nt' and 'CONDA_PREFIX' in os.environ:
    # Necessary hack as on some systems, conda sometimes installs C++ headers into "Library/include" instead of just "include" ...
    os.environ["NVCC_PREPEND_FLAGS"] = f"-I{os.environ['CONDA_PREFIX']}\Library\include"

device = torch.device('cuda')


def main(args):
    chkpoints = [-1]
    if args.checkpoints:
        chkpoints = args.checkpoints
    
    model_manager = find_model_manager(args.model)
    model_short = args.model[:args.model.index("_")]
    time_attribute = stat.normalize_attributes_list(args.attribute)[0] if args.attribute is not None else None
    vid_opts_prefix = ("_" + stat.types[time_attribute].short) if time_attribute is not None else ""
    vid_opts_suffix = f"_{args.cols}_{args.rows}" + (f"_seed{args.seed}" if args.seed != 136 else "") + ("_lossy" if args.lossy else "")
    
    num_frames = args.fps * args.time
    
    for chk in chkpoints:
        checkpoint = model_manager._resolve_checkpoint_id(chk)
        print(f"Loading {args.model} at checkpoint {checkpoint}")
        model = model_manager.load_checkpoint(checkpoint, load_ema=True).to(device)
        
        img_file_name = f"video{vid_opts_prefix}_{model_short}_chk{checkpoint}{vid_opts_suffix}.mkv"
        dst_file = pathlib.Path(".") / "renderings" / img_file_name
        if args.dst:
            dst_file = pathlib.Path(args.dst)
        os.makedirs(dst_file.parent, exist_ok=True)
        
        dataset = None
        if args.dataset is not None:
            dataset = GGHeadImageFolderDataset(GGHeadImageFolderDatasetConfig(path=args.dataset, resolution=args.res, use_labels=True))
        
        rng = torch.Generator(device)
        if not args.no_seed:
            rng.manual_seed(args.seed)
            random.seed(args.seed)
        grid_size = (clamp(args.cols, 1, 7680 // args.res), clamp(args.rows, 1, 4320 // args.res))
        
        # Load c poses from dataset if provided:
        if dataset:
            c_list = [dataset.get_label(idx) for idx in range(len(dataset))]
            random.shuffle(c_list)
            grid_c = torch.from_numpy(np.stack(c_list[:grid_size[0] * grid_size[1]], 0)).to(device)
        else:
            pose_front = Pose(
                matrix_or_rotation=np.eye(3),
                translation=(0, 0, 2.7),
                pose_type=PoseType.CAM_2_WORLD,
                camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL)
            c_front = torch.from_numpy(encode_camera_params(pose_front, DEFAULT_INTRINSICS)).to(device).unsqueeze(0)
            #poses = [pose_front.rotate_euler(order="XYZ", euler_y=math.radians(min(abs(i) * (5.0 / (grid_size[0] - 1 // 2)), 5.0) * (-1.0 if i < 0 else 1.0)), inplace=False) for i in range(-grid_size[0] // 2, grid_size[0] - (grid_size[0] // 2))]
            poses = [pose_front] * grid_size[0]
            
            c_list = [encode_camera_params(p, DEFAULT_INTRINSICS) for p in poses] * grid_size[1]
            grid_c = torch.from_numpy(np.stack(c_list, 0)).to(device)
        
        grid_z = torch.randn([grid_size[0] * grid_size[1], model.z_dim], device=rng.device, generator=rng).to(device)
        
        grid_z = grid_z.split(args.batch)
        grid_c = grid_c.split(args.batch)
        
        video_writer = imageio_ffmpeg.write_frames(
            str(dst_file), (grid_size[0]*args.res, grid_size[1]*args.res),
            fps=args.fps,
            codec="ffv1" if not args.lossy else "libx265",
            quality=10 if not args.lossy else 7,
            pix_fmt_in="gray",
            pix_fmt_out="gray",
        )
        video_writer.send(None)
        with tqdm(total=grid_size[0] * grid_size[1] * num_frames // args.batch) as pbar:
            for i in range(num_frames):
                attr_val = attribute_value_step(time_attribute, i / args.fps, args.speed)
                frame = render_grid_frame(model, dataset, grid_size, grid_z, grid_c, attr_val, args, pbar)
                video_writer.send(np.asarray(frame, dtype=np.uint8))
        video_writer.close()
        print(dst_file)


def render_grid_frame(model, dataset, grid_size, grid_z, grid_c, attr_val, args, pbar):
    attr_val = attr_val.reshape((1,-1)).repeat(args.batch, 1)
    images = []
    with torch.no_grad():
        for idx in range(grid_size[0] * grid_size[1] // args.batch):
            z = grid_z[idx]
            c = grid_c[idx]
            attr = attr_val
            
            w = model.mapping(z, c, attr, truncation_psi=0.7)
            output = model.synthesis(w, c, noise_mode='const', return_masks=False,
                                     neural_rendering_resolution=args.res,
                                     return_uv_map=False)
            images += [PIL.ImageOps.grayscale(PIL.Image.fromarray(normalized_torch_to_numpy_img(output["image"][i, ...]))) for i in range(output["image"].shape[0])]
            pbar.update(1)
    
    img_grid = PIL.Image.new("L", size=(grid_size[0] * args.res, grid_size[1] * args.res))
    for idx, img in enumerate(images):
        img_grid.paste(img, box=((idx % grid_size[0]) * args.res, (idx // grid_size[0]) * args.res))
    return img_grid


def attribute_value_step(attr, t, speed):
    t *= speed * 0.5
    val = None
    if attr == "eye_open":
        val = (math.cos(math.pi * t) + 1.0) * 0.5
    elif attr == "gaze":
        t = t % 7.5
        if t < 2.0:
            x = math.sin(math.pi * t) * (math.pi / 2)
            y = 0.0
        elif t < 3.5:
            x = 0.0
            y = math.sin(math.pi * t) * (math.pi / 2)
        else:
            x = math.cos(math.pi * t) * (math.pi / 2)
            y = math.sin(math.pi * t) * (math.pi / 2)
        val = [y, x]
    
    tensor = torch.tensor(val, dtype=torch.float32, requires_grad=False, device=device)
    return tensor
    

def clamp(x, lo, hi):
    return min(hi, max(lo, x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-b", "--batch", type=int, default=4)
    parser.add_argument("--dataset", type=str, default=None)  # Optional: dataset to sample poses from, otherwise poses are always frontal and uniformly rotated along X-axis
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("-t", "--time", type=int, default=10)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--dst", type=str, default=None)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=None)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--seed", type=int, default=136)
    parser.add_argument("--no_seed", action="store_true", default=False)
    parser.add_argument("--attribute", type=str, nargs="+", default=None)  # grid acts as gradient over given attribute
    parser.add_argument("--lossy", action="store_true", default=False)
    args = parser.parse_args()
    
    main(args)
