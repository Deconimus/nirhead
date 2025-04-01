import argparse, pathlib, os, cv2, PIL.Image, math
import torch
import numpy as np
from dreifus.camera import PoseType, CameraCoordinateConvention
from dreifus.image import normalized_torch_to_numpy_img
from dreifus.matrix import Pose
from eg3d.datamanager.nersemble import encode_camera_params
from gaussian_splatting.arguments import PipelineParams2
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.cameras import pose_to_rendercam
from gghead.model_manager.finder import find_model_manager
from gghead.constants import DEFAULT_INTRINSICS
from eg3d.training.training_loop import setup_snapshot_image_grid, save_image_grid


if os.name == 'nt' and 'CONDA_PREFIX' in os.environ:
    # Necessary hack as on some systems, conda sometimes installs C++ headers into "Library/include" instead of just "include" ...
    os.environ["NVCC_PREPEND_FLAGS"] = f"-I{os.environ['CONDA_PREFIX']}\Library\include"

device = torch.device('cuda')


def main(args):
    checkpoint = -1
    if args.checkpoint:
        checkpoint = args.checkpoint
        
    dst_file = pathlib.Path(".") / ("grid_"+args.model+".png")
    if args.dst:
        dst_file = pathlib.Path(args.dst)
        
    model_manager = find_model_manager(args.model)
    checkpoint = model_manager._resolve_checkpoint_id(checkpoint)
    print(checkpoint)
    model = model_manager.load_checkpoint(checkpoint, load_ema=True).to(device)
    
    rng = torch.Generator(device)
    if not args.no_seed:
        rng.manual_seed(args.seed)
    grid_size = (np.clip(7680 // args.res, 1, args.cols), np.clip(4320 // args.res, 1, args.rows))
    
    pose_front = Pose(
        matrix_or_rotation=np.eye(3),
        translation=(0, 0, 2.7),
        pose_type=PoseType.CAM_2_WORLD,
        camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL)
    c_front = torch.from_numpy(encode_camera_params(pose_front, DEFAULT_INTRINSICS)).to(device).unsqueeze(0)
    
    poses = [pose_front.rotate_euler(order="XYZ", euler_y=math.radians(min(abs(i) * (2.0 / (grid_size[0]-1 // 2)), 2.0) * (-1.0 if i < 0 else 1.0)), inplace=False) for i in range(-grid_size[0]//2, grid_size[0]-(grid_size[0]//2))]
    c_list = [encode_camera_params(p, DEFAULT_INTRINSICS) for p in poses] * grid_size[1]
    grid_c = torch.from_numpy(np.stack(c_list, 0)).to(device).split(args.batch)
    
    grid_z = torch.randn([grid_size[0] * grid_size[1], model.z_dim], device=device)
    grid_attr = (torch.rand([grid_size[0] * grid_size[1], len(model._config.static_attributes)], dtype=torch.float32) + 0.5).int().float().to(device)
    
    if args.bright_pupil is not None:
        grid_attr[:,0] = 1.0 if args.bright_pupil else 0.0
    
    if args.attribute_gradient:
        for row in range(grid_size[1] // 2):
            for col in range(grid_size[0]):
                grid_z[(row*2+1)*grid_size[0] + col, :] = grid_z[(row*2)*grid_size[0] + col, :]
                grid_attr[(row*2)*grid_size[0] + col, 0] = 0.0
                grid_attr[(row*2+1)*grid_size[0] + col, 0] = 1.0
                
    grid_z = grid_z.split(args.batch)
    grid_attr = grid_attr.split(args.batch)
    
    images = []
    with torch.no_grad():
        for row in range(grid_size[1]):
            for col in range(grid_size[0] // args.batch):
                idx = row*(grid_size[0]//args.batch)+col
                z = grid_z[idx]
                c = grid_c[idx]
                attr = grid_attr[idx]
                
                w = model.mapping(z, c, attr, truncation_psi=0.7)
                output = model.synthesis(w, c, noise_mode='const', return_masks=False,
                                         sh_ref_cam=pose_front,
                                         neural_rendering_resolution=args.res,
                                         return_uv_map=False)
                images += [PIL.Image.fromarray(normalized_torch_to_numpy_img(output["image"][i,...])) for i in range(output["image"].shape[0])]
    
    img_grid = PIL.Image.new("RGB", size=(grid_size[0]*args.res, grid_size[1]*args.res))
    for idx, img in enumerate(images):
        img_grid.paste(img, box=((idx%grid_size[0])*args.res, (idx//grid_size[1])*args.res))
    img_grid.save(dst_file)
    print(dst_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-b", "--batch", type=int, default=4)
    parser.add_argument("--dst", type=str, default=None)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--seed", type=int, default=136)
    parser.add_argument("--no_seed", action="store_true", default=False)
    parser.add_argument("--bright_pupil", type=bool, default=None)
    parser.add_argument("--attribute_gradient", type=str, default=None) # grid acts as gradient over given attribute
    args = parser.parse_args()
    
    main(args)
    