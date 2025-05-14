import argparse, pathlib, os, cv2, PIL.Image, PIL.ImageOps, math, random, imageio_ffmpeg, json
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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
from nirhead.models.classifier import load_classification_model_dir


if os.name == 'nt' and 'CONDA_PREFIX' in os.environ:
    # Necessary hack as on some systems, conda sometimes installs C++ headers into "Library/include" instead of just "include" ...
    os.environ["NVCC_PREPEND_FLAGS"] = f"-I{os.environ['CONDA_PREFIX']}\Library\include"

device = torch.device('cuda')


def main(args):
    static_attributes = stat.normalize_attributes_list(stat.labels)
    
    dataset_poses = None
    if args.poses is not None:
        with open(args.poses, "r") as f:
            d3fr_json = json.load(f)
        dataset_poses = {p[0]: p[1] for p in d3fr_json["labels"]}
        dataset_pose_keys = sorted(list(dataset_poses.keys()))
    
    num_samples = args.num
    num_batches = int(math.ceil(args.num / args.batch))
    
    dst_dir = pathlib.Path(args.dst)
    os.makedirs(dst_dir, exist_ok=True)
    
    chkpt = -1
    if args.checkpoint is not None:
        chkpt = args.checkpoint
    
    model_manager = find_model_manager(args.model)
    model_short = args.model[:args.model.index("_")]
    
    checkpoint = model_manager._resolve_checkpoint_id(chkpt)
    print(f"Loading {args.model} at checkpoint {checkpoint}")
    model = model_manager.load_checkpoint(checkpoint, load_ema=True).to(device)
    
    classifier = None
    if args.classifier and static_attributes:
        classifier, classifier_name = load_classification_model_dir(args.classifier, device=device)
    
    rng = torch.Generator(device)
    if not args.no_seed:
        rng.manual_seed(args.seed)
        random.seed(args.seed)
    
    # latent z data
    z_batches = torch.randn([1, model.z_dim], device=rng.device, generator=rng).to("cpu")
    
    # pose data
    if dataset_poses:
        c_list = [dataset_poses[dataset_pose_keys[idx % len(dataset_pose_keys)]] for idx in range(num_samples)]
        random.shuffle(c_list)
        c_batches = torch.from_numpy(np.stack(c_list, 0)).to("cpu")
    else:
        pose_front = Pose(
            matrix_or_rotation=np.eye(3),
            translation=(0, 0, 2.7),
            pose_type=PoseType.CAM_2_WORLD,
            camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL)
        c_front = torch.from_numpy(encode_camera_params(pose_front, DEFAULT_INTRINSICS)).to(device).unsqueeze(0)
        c_batches = torch.from_numpy(np.stack([c_front] * num_samples, 0)).to("cpu")
    
    # attributes data
    attr_batches = stat.random_attribute_tensor(static_attributes, num_samples, device="cpu", rng=rng) if static_attributes else []
    # TODO: minpulate attr based on arguments
    # Note: for bp, copy every other z and c and have only attr different (0 and 1), to have perfect separation
    
    z_batches    = z_batches.split(args.batch)
    c_batches    = c_batches.split(args.batch)
    attr_batches = attr_batches.split(args.batch)
    
    labels = {"labels": {}}
    dst_file_labels = dst_dir / "labels.json"
    if os.path.isfile(dst_file_labels) and os.path.exists(dst_file_labels):
        with open(dst_file_labels, "r") as f:
            labels = json.load(f)
    
    with torch.no_grad():
        with ThreadPoolExecutor(max_workers=8) as xec:
            for i in tqdm(range(len(z_batches))):
                batch_size = z_batches[i].shape[0]
                
                images = render_batch(model, z_batches[i].to(device), c_batches[i].to(device), attr_batches[i].to(device), batch_size)
                image_names = [str(i*args.batch+j).zfill(6)+".png" for j in range(len(images))]
                
                if classifier is not None and static_attributes is not None:
                    batch_labels = label_images(images, image_names, static_attributes, classifier)
                    for k in batch_labels.keys():
                        labels["labels"][k] = batch_labels[k]
                
                for j in range(len(images)):
                    xec.submit(save_image, images[j], dst_dir / image_names[j])
    
    with open(dst_file_labels, "w+") as f:
        json.dump(labels, f, indent=2)
    print(dst_file_labels)
    

def render_batch(model, z, c, attr, batch_size):
    w = model.mapping(z, c, attr, truncation_psi=0.7)
    output = model.synthesis(w, c, noise_mode='const', return_masks=False,
                             neural_rendering_resolution=args.res,
                             return_uv_map=False)
    images = [PIL.ImageOps.grayscale(PIL.Image.fromarray(normalized_torch_to_numpy_img(output["image"][i, ...]))) for i in range(output["image"].shape[0])]
    return images


def label_images(images, image_names, static_attributes, classifier):
    x = torch.tensor(np.stack([np.asarray(img, np.uint8) for img in images], 0)).to(device)
    y_pred = classifier(x).to("cpu")
    return stat.labels_from_attributes(y_pred, image_names, static_attributes)


def save_image(image, dst_file):
    image.save(dst_file, compress_level=0)


def clamp(x, lo, hi):
    return min(hi, max(lo, x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("-b", "--batch", type=int, default=4)
    parser.add_argument("-n", "--num", type=int, required=True) # num samples to generate
    parser.add_argument("--poses", type=str, default="/mnt/g/EyesNIR/scripts/enir_d3fr_pose.json")  # Optional: file to sample poses from, otherwise poses are always frontal and uniformly rotated along X-axis
    parser.add_argument("--dst", type=str, default=None)
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--seed", type=int, default=136)
    parser.add_argument("--labels", type=str, nargs="+", default=None)  # grid acts as gradient over given attributes
    parser.add_argument("--classifier", type=str, default=None) # to specify labelling classifier for output images
    
    parser.add_argument("--filter_bp", type=int, default=None)
    parser.add_argument("--filter_eo_min", type=float, default=None)
    parser.add_argument("--filter_eo_max", type=float, default=None)
    args = parser.parse_args()
    
    main(args)
