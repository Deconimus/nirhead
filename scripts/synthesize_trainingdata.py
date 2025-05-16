import argparse, pathlib, os, PIL.Image, PIL.ImageOps, math, random, json, shutil
import torch, torchvision
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
    static_attributes = stat.normalize_attributes_list(args.labels)
    
    dataset_poses = None
    if args.poses is not None:
        with open(args.poses, "r") as f:
            d3fr_json = json.load(f)
        dataset_poses = {p[0]: p[1] for p in d3fr_json["labels"]}
        dataset_pose_keys = sorted(list(dataset_poses.keys()))
    
    num_samples = args.num
    num_batches = int(math.ceil(args.num / args.batch))
    
    chkpt = -1
    if args.checkpoint is not None:
        chkpt = args.checkpoint
    
    model_manager = find_model_manager(args.model)
    model_short = args.model[:args.model.index("_")] if args.model.startswith("gh") else args.model
    
    dst_dir = pathlib.Path(args.dst) / model_short
    dst_img_dir = dst_dir / "train"
    os.makedirs(dst_img_dir, exist_ok=True)
    
    checkpoint = model_manager._resolve_checkpoint_id(chkpt)
    print(f"Loading {args.model} at checkpoint {checkpoint}")
    model = model_manager.load_checkpoint(checkpoint, load_ema=True).to(device)
    
    classifier = None
    if args.classifier and static_attributes:
        classifier, classifier_name = load_classification_model_dir(args.classifier, device=device)
    
    rng = torch.Generator(device)
    rng.manual_seed(args.seed)
    random.seed(args.seed)
    
    # latent z data
    z_batches = torch.randn([num_samples, model.z_dim], device=rng.device, generator=rng).to("cpu")
    
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
    attr_indices = stat.attribute_indices(static_attributes)
    
    # TODO: support for eye_open and gaze control (should happen earlier than bp control right here)
    if "bright_pupil" in attr_indices.keys():
        attr_idx = attr_indices["bright_pupil"]
        if args.filter_bp is not None:
            attr_batches[:,attr_idx] = bool(args.filter_bp)
        else:
            for i in range(attr_batches.shape[0] // 2):
                z_batches[i*2+1, :] = z_batches[i*2, :]
                c_batches[i*2+1, :] = c_batches[i*2, :]
                attr_batches[i*2+1, :] = attr_batches[i*2, :]
                attr_batches[i*2,   attr_idx] = False
                attr_batches[i*2+1, attr_idx] = True
    
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
                
                images, image_tensors = render_batch(model, z_batches[i].to(device), c_batches[i].to(device), attr_batches[i].to(device), batch_size)
                image_names = ["synthetic_"+str(i*args.batch+j).zfill(6)+".png" for j in range(len(images))]
                
                if classifier is not None and static_attributes is not None:
                    batch_labels = label_images(image_tensors, image_names, static_attributes, classifier)
                    for k in batch_labels.keys():
                        labels["labels"][k] = batch_labels[k]
                
                for j in range(len(images)):
                    xec.submit(save_image, images[j], dst_img_dir / image_names[j])
    
    with open(dst_file_labels, "w+") as f:
        json.dump(labels, f, indent=2)
    print(dst_file_labels)
    
    # if dst_dir has a subfolder "base", copy files from there and merge labels.json files
    dst_base_dir = dst_dir.parent / "base"
    if os.path.exists(dst_base_dir) and os.path.isdir(dst_base_dir) and os.path.exists(dst_base_dir / "labels.json"):
        # merge labels with base/labels
        with open(dst_base_dir / "labels.json", "r") as f:
            base_labels = json.load(f)
        for k in base_labels.keys():
            labels[k] = base_labels[k]
        with open(dst_file_labels, "w+") as f:
            json.dump(labels, f, indent=2)
            
        # copy files from base
        for d in os.listdir(dst_base_dir):
            subdir = dst_base_dir / d
            if not os.path.isdir(subdir): continue
            os.makedirs(dst_dir / d, exist_ok=True)
            for file in subdir.glob("*"):
                dst_file = dst_dir / str(file.absolute())[len(str(dst_base_dir.absolute()))+1:]
                shutil.copy2(str(file), str(dst_file))
        
        print(f"Merged base subdir into {dst_dir.name}.")
    

def render_batch(model, z, c, attr, batch_size):
    w = model.mapping(z, c, attr, truncation_psi=0.7)
    output = model.synthesis(w, c, noise_mode='const', return_masks=False,
                             neural_rendering_resolution=args.res,
                             return_uv_map=False)
    image_tensors = output["image"]
    images = [PIL.ImageOps.grayscale(PIL.Image.fromarray(normalized_torch_to_numpy_img(output["image"][i, ...]))) for i in range(output["image"].shape[0])]
    return images, image_tensors


def label_images(images, image_names, static_attributes, classifier):
    x = torchvision.transforms.functional.rgb_to_grayscale(images.to(device))
    y_pred = classifier(x).to("cpu")
    y_pred_filtered = stat.take_from_attribute_tensor(y_pred, classifier.static_attributes, static_attributes)
    labels = stat.labels_from_attribute_tensor(y_pred_filtered, image_names, static_attributes)
    return labels


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
