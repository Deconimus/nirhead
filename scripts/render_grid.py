import argparse, pathlib, os, cv2, PIL.Image, PIL.ImageOps, math, random
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
    grid_opts_name_str = '_' + stat.types[args.attribute_gradient].short + '_grad' if args.attribute_gradient is not None else ''
    
    for chk in chkpoints:
        checkpoint = model_manager._resolve_checkpoint_id(chk)
        print(f"Loading {args.model} at checkpoint {checkpoint}")
        model = model_manager.load_checkpoint(checkpoint, load_ema=True).to(device)
        
        img_file_name = f"grid{grid_opts_name_str}_{model_short}_chk{checkpoint}.png"
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
        grid_size = (np.clip(7680 // args.res, 1, args.cols), np.clip(4320 // args.res, 1, args.rows))
        
        subgrids_x = clamp(args.subgrids_x, 1, grid_size[0])
        subgrids_y = clamp(args.subgrids_y, 1, grid_size[1])
        
        with tqdm(total=grid_size[0]*grid_size[1] // args.batch) as pbar:
            if args.subgrids_x > 1 or args.subgrids_y > 1:
                subgrid_size = (grid_size[0] // subgrids_x, grid_size[1] // subgrids_y)
                img_grid = PIL.Image.new("L", size=(grid_size[0] * args.res, grid_size[1] * args.res))
                for y in range(subgrids_y):
                    for x in range(subgrids_x):
                        img = render_grid(model, subgrid_size, rng, dataset, args, pbar)
                        img_grid.paste(img, box=(x * subgrid_size[0] * args.res, y * subgrid_size[1] * args.res))
            else:
                img_grid = render_grid(model, grid_size, rng, dataset, args, pbar)
        img_grid.save(dst_file)
        print(dst_file)
    

def render_grid(model, grid_size, rng, dataset, args, pbar):
    # Load c poses from dataset if provided:
    if dataset:
        c_list = [dataset.get_label(idx) for idx in range(len(dataset))]
        random.shuffle(c_list)
        grid_c = torch.from_numpy(np.stack(c_list, 0)).to(device)
    else:
        pose_front = Pose(
            matrix_or_rotation=np.eye(3),
            translation=(0, 0, 2.7),
            pose_type=PoseType.CAM_2_WORLD,
            camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL)
        c_front = torch.from_numpy(encode_camera_params(pose_front, DEFAULT_INTRINSICS)).to(device).unsqueeze(0)
        poses = [pose_front.rotate_euler(order="XYZ", euler_y=math.radians(min(abs(i) * (5.0 / (grid_size[0] - 1 // 2)), 5.0) * (-1.0 if i < 0 else 1.0)), inplace=False) for i in range(-grid_size[0] // 2, grid_size[0] - (grid_size[0] // 2))]
        
        c_list = [encode_camera_params(p, DEFAULT_INTRINSICS) for p in poses] * grid_size[1]
        grid_c = torch.from_numpy(np.stack(c_list, 0)).to(device)
        
    grid_z = torch.randn([grid_size[0] * grid_size[1], model.z_dim], device=rng.device, generator=rng).to(device)
    grid_attr = stat.random_attribute_tensor(model._config.static_attributes, grid_size[0] * grid_size[1], device=device, rng=rng)
    
    if args.bright_pupil is not None:
        grid_attr[:, 0] = 1.0 if args.bright_pupil else 0.0
    
    if args.attribute_gradient:
        attr_indices = stat.attribute_indices(model._config.static_attributes)
        
        grad_attr_idx = attr_indices[args.attribute_gradient]
        grad_attr_dim = stat.types[args.attribute_gradient].dim
        grad_fun = lambda x, size: (x * (stat.types[args.attribute_gradient].high - stat.types[args.attribute_gradient].low)) / (size - 1) + stat.types[args.attribute_gradient].low
        
        if stat.types[args.attribute_gradient].dtype == bool:
            for row in range(grid_size[1] // 2):
                for col in range(grid_size[0]):
                    grid_z[(row * 2 + 1) * grid_size[0] + col, :] = grid_z[(row * 2) * grid_size[0] + col, :]  # copy z vals
                    grid_c[(row * 2 + 1) * grid_size[0] + col, :] = grid_c[(row * 2) * grid_size[0] + col, :]  # copy c vals
                    grid_attr[(row * 2 + 1) * grid_size[0] + col, :] = grid_attr[(row * 2) * grid_size[0] + col, :]  # copy attr vals
                    grid_attr[(row * 2) * grid_size[0] + col, grad_attr_idx] = 0.0
                    grid_attr[(row * 2 + 1) * grid_size[0] + col, grad_attr_idx] = 1.0
        elif stat.types[args.attribute_gradient].dtype == float:
            for row in range(1 if grad_attr_dim <= 1 else 0, grid_size[1]):
                for col in range(grid_size[0]):
                    src_col = col if grad_attr_dim <= 1 else 0
                    grid_z[row * grid_size[0] + col, :] = grid_z[0 + src_col, :]  # copy z vals
                    grid_c[row * grid_size[0] + col, :] = grid_c[0 + src_col, :]  # copy c vals
                    grid_attr[row * grid_size[0] + col, :] = grid_attr[0 + src_col, :]  # copy attr vals
            for row in range(grid_size[1]):
                attr_val_y = grad_fun(row, grid_size[1])
                for col in range(grid_size[0]):
                    grid_attr[row * grid_size[0] + col, grad_attr_idx] = attr_val_y
                    if grad_attr_dim > 1:
                        attr_val_x = grad_fun(col, grid_size[0])
                        grid_attr[row * grid_size[0] + col, grad_attr_idx + 1] = attr_val_x
                        #print(f"pitch={attr_val_y}, yaw={attr_val_x}")
    
    grid_c = grid_c.split(args.batch)
    grid_z = grid_z.split(args.batch)
    grid_attr = grid_attr.split(args.batch)
    
    images = []
    with torch.no_grad():
        for idx in range(grid_size[0] * grid_size[1] // args.batch):
            # idx = row*(grid_size[0]//args.batch)+col
            z = grid_z[idx]
            c = grid_c[idx]
            attr = grid_attr[idx]
            
            w = model.mapping(z, c, attr, truncation_psi=0.7)
            output = model.synthesis(w, c, noise_mode='const', return_masks=False,
                                     neural_rendering_resolution=args.res,
                                     return_uv_map=False)
            images += [PIL.ImageOps.grayscale(PIL.Image.fromarray(normalized_torch_to_numpy_img(output["image"][i, ...]))) for i in range(output["image"].shape[0])]
            pbar.update(1)
    
    img_grid = PIL.Image.new("L", size=(grid_size[0] * args.res, grid_size[1] * args.res))
    for idx, img in enumerate(images):
        img_grid.paste(img, box=((idx % grid_size[0]) * args.res, (idx // grid_size[1]) * args.res))
    return img_grid
    

def clamp(x, lo, hi):
    return min(hi, max(lo, x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-b", "--batch", type=int, default=4)
    parser.add_argument("--dataset", type=str, default=None) # Optional: dataset to sample poses from, otherwise poses are always frontal and uniformly rotated along X-axis
    parser.add_argument("--dst", type=str, default=None)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=None)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--seed", type=int, default=136)
    parser.add_argument("--no_seed", action="store_true", default=False)
    parser.add_argument("--bright_pupil", type=bool, default=None)
    parser.add_argument("--attribute_gradient", type=str, default=None) # grid acts as gradient over given attribute
    parser.add_argument("--subgrids_x", type=int, default=1)
    parser.add_argument("--subgrids_y", type=int, default=1)
    args = parser.parse_args()
    
    main(args)
    