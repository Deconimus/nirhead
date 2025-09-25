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

from gghead.model_manager.finder import find_model_manager, full_run_name
from gghead.constants import DEFAULT_INTRINSICS
from gghead.dataset.image_folder_dataset import GGHeadImageFolderDataset, GGHeadImageFolderDatasetConfig

import nirhead.data.static_attributes as stat
from nirhead.models.classifier import load_classification_model_dir


if os.name == 'nt' and 'CONDA_PREFIX' in os.environ:
    # Necessary hack as on some systems, conda sometimes installs C++ headers into "Library/include" instead of just "include" ...
    os.environ["NVCC_PREPEND_FLAGS"] = f"-I{os.environ['CONDA_PREFIX']}\Library\include"

device = torch.device('cuda')

NUM_ATTR_BINS = { "bright_pupil": 2, "eye_open": 3, "gaze": 5 }


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
    
    args.model = full_run_name(args.model)
    model_manager = find_model_manager(args.model)
    model_short = args.model[:args.model.index("_")] if args.model.startswith("gh") else args.model
    
    if not args.nosubdir:
        dst_dir = pathlib.Path(args.dst) / model_short
        dst_img_dir = dst_dir / "train"
    else:
        dst_dir = pathlib.Path(args.dst)
        dst_img_dir = dst_dir
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
    attr_batches = stat.random_attribute_tensor(static_attributes, num_samples, device="cpu", rng=rng) if static_attributes else torch.tensor([], dtype=torch.float32)
    attr_indices = stat.attribute_indices(static_attributes) if static_attributes else {}
    
    # -- Even-out Attribute Distribution ---------------------------------------
    
    # TODO: add option to "even out" a distribution found in a given dataset by synthesizing data that will balance value bins for gaze and optionally also for eye_open
    if args.augment_distribution:
        real_bins = get_real_data_bins(dst_dir.parent / "base", static_attributes)
        print(f"Real Bins: {real_bins}")
        
        # greedy determine number of synth samples per bin
        synth_bins = {k: [[0 for _ in range(NUM_ATTR_BINS[k])] for _ in range(stat.types[k].dim)] for k in real_bins.keys()}
        for i in range(num_samples):
            for attr in synth_bins.keys():
                for elem_idx in range(stat.types[attr].dim):
                    smallest_bin, smallest_bin_num = 0, -1
                    for b in range(len(synth_bins[attr][elem_idx])):
                        bin_num = synth_bins[attr][elem_idx][b] + real_bins[attr][elem_idx][b]
                        if smallest_bin_num < 0 or bin_num < smallest_bin_num:
                            smallest_bin = b
                            smallest_bin_num = bin_num
                    synth_bins[attr][elem_idx][smallest_bin] += 1
        print(f"Synthetic Bins: {synth_bins}")
        
        # generate attribute values
        tensors = []
        for attr in static_attributes:
            attr_dim = stat.types[attr].dim
            attr_dtype = stat.types[attr].dtype
            attr_low = stat.types[attr].low
            attr_high = stat.types[attr].high
            for elem_idx in range(attr_dim):
                ts = []
                for b, bin_samples in enumerate(synth_bins[attr][elem_idx]):
                    shape = (bin_samples, 1)
                    t = None
                    if attr_dtype == bool:
                        if b == 0:
                            t = torch.zeros(shape, dtype=torch.float32, device="cpu")
                        else:
                            t = torch.ones(shape, dtype=torch.float32, device="cpu")
                    else:
                        low = attr_low + (attr_high - attr_low) * (b / len(synth_bins[attr][elem_idx]))
                        high = attr_low + (attr_high - attr_low) * ((b+1) / len(synth_bins[attr][elem_idx]))
                        if attr_dtype == int:
                            t = torch.randint(int(low), int(high), shape, dtype=torch.int32, device="cpu").to(dtype=torch.float32)
                        elif attr_dtype == float:
                            t = torch.rand(shape, dtype=torch.float32, device="cpu") * (high - low) + low
                    ts.append(t)
                attr_tensor = torch.cat(ts, dim=0)
                attr_tensor = attr_tensor[torch.randperm(attr_tensor.shape[0])]
                tensors.append(attr_tensor)
        attr_batches = torch.cat(tensors, dim=1)
        print(f"attributes tensor shape: {attr_batches.shape}")
    
    
    # -- Attribute Control -----------------------------------------------------
    
    if "gaze" in attr_indices.keys():
        attr_idx = attr_indices["gaze"]
        
        if args.filter_gz_radius is not None:
            args.filter_gz_pitch_min = -args.filter_gz_radius
            args.filter_gz_pitch_max = args.filter_gz_radius
            args.filter_gz_yaw_min = -args.filter_gz_radius
            args.filter_gz_yaw_max = args.filter_gz_radius
            
        pitch_low, pitch_high = stat.types["gaze"].low, stat.types["gaze"].high
        yaw_low, yaw_high = stat.types["gaze"].low, stat.types["gaze"].high
        if args.filter_gz_pitch_min is not None: pitch_low  = math.radians(args.filter_gz_pitch_min)
        if args.filter_gz_pitch_max is not None: pitch_high = math.radians(args.filter_gz_pitch_max)
        if args.filter_gz_yaw_min is not None: yaw_low  = math.radians(args.filter_gz_yaw_min)
        if args.filter_gz_yaw_max is not None: yaw_high = math.radians(args.filter_gz_yaw_max)
        if not (pitch_low == stat.types["gaze"].low and pitch_high == stat.types["gaze"].high):
            attr_batches[:, attr_idx + 0] = (attr_batches[:, attr_idx + 0] - stat.types["gaze"].low) * ((pitch_high - pitch_low) / (stat.types["gaze"].high - stat.types["gaze"].low)) + pitch_low
        if not (yaw_low == stat.types["gaze"].low and yaw_high == stat.types["gaze"].high):
            attr_batches[:, attr_idx + 1] = (attr_batches[:, attr_idx + 1] - stat.types["gaze"].low) * ((yaw_high - yaw_low) / (stat.types["gaze"].high - stat.types["gaze"].low)) + yaw_low
            
        if args.filter_gz_deadzone is not None:
            gz_deadzone_rad = math.radians(args.filter_gz_deadzone)
            for i in range(attr_batches.shape[0]):
                if attr_batches[i, attr_idx+0] <= 0.0:
                    attr_batches[i, attr_idx+0] = (attr_batches[i, attr_idx+0] - pitch_low) * ((-gz_deadzone_rad - pitch_low) / (pitch_high - pitch_low)) + pitch_low
                else:
                    attr_batches[i, attr_idx+0] = (attr_batches[i, attr_idx+0] - pitch_low) * ((pitch_high - gz_deadzone_rad) / (pitch_high - pitch_low)) + gz_deadzone_rad
                if attr_batches[i, attr_idx+1] <= 0.0:
                    attr_batches[i, attr_idx+1] = (attr_batches[i, attr_idx+1] - yaw_low) * ((-gz_deadzone_rad - yaw_low) / (yaw_high - yaw_low)) + yaw_low
                else:
                    attr_batches[i, attr_idx+1] = (attr_batches[i, attr_idx+1] - yaw_low) * ((yaw_high - gz_deadzone_rad) / (yaw_high - yaw_low)) + gz_deadzone_rad
    
    if "eye_open" in attr_indices.keys():
        attr_idx = attr_indices["eye_open"]
        low, high = 0.0, 1.0
        if args.filter_eo_min is not None:
            low = args.filter_eo_min
        if args.filter_eo_max is not None:
            high = args.filter_eo_max
        if not (low == 0.0 and high == 1.0):
            attr_batches[:, attr_idx] = attr_batches[:, attr_idx] * (high - low) + low # transform uniform random values from [0,1] to [low,high]
    
    for label_name, label_filter_arg in [("bright_pupil", args.filter_bp), ("facial_hair", args.filter_fh), ("glasses", args.filter_gl)]:
        if label_name in attr_indices.keys():
            attr_idx = attr_indices[label_name]
            if label_filter_arg is not None:
                attr_batches[:,attr_idx] = bool(label_filter_arg)
            elif args.augment_distribution is None: # 50/50 distribution, if not augmenting given data
                for i in range(attr_batches.shape[0] // 2):
                    z_batches[i*2+1, :] = z_batches[i*2, :]
                    c_batches[i*2+1, :] = c_batches[i*2, :]
                    attr_batches[i*2+1, :] = attr_batches[i*2, :]
                    attr_batches[i*2,   attr_idx] = False
                    attr_batches[i*2+1, attr_idx] = True
    
    z_batches    = z_batches.split(args.batch)
    c_batches    = c_batches.split(args.batch)
    attr_batches = attr_batches.split(args.batch) if static_attributes else None
    
    
    # -- Synthesize Images -----------------------------------------------------
    
    labels = {}
    dst_file_labels = dst_dir / "labels.json"
    if os.path.isfile(dst_file_labels) and os.path.exists(dst_file_labels):
        with open(dst_file_labels, "r") as f:
            labels = json.load(f)
    
    with torch.no_grad():
        with ThreadPoolExecutor(max_workers=8) as xec:
            for i in tqdm(range(len(z_batches))):
                batch_size = z_batches[i].shape[0]
                z_batch = z_batches[i].to(device)
                c_batch = c_batches[i].to(device)
                attr_batch = attr_batches[i].to(device) if static_attributes else None
                
                images, image_tensors = render_batch(model, z_batch, c_batch, attr_batch, batch_size)
                image_names = ["synthetic_"+str(i*args.batch+j).zfill(6)+".png" for j in range(len(images))]
                
                if (classifier is not None and static_attributes is not None) or args.store_latent:
                    batch_labels = label_images(image_tensors, image_names, static_attributes, classifier, z_batches[i], c_batches[i], args)
                    for k in batch_labels.keys():
                        labels["train/"+k] = batch_labels[k]
                
                for j in range(len(images)):
                    xec.submit(save_image, images[j], dst_img_dir / image_names[j])
    
    with open(dst_file_labels, "w+") as f:
        json.dump(labels, f, indent=2)
    print(dst_file_labels)
    
    
    # -- Add Labels and Images from Base-Folder --------------------------------
    
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
            copy_files = list(subdir.glob("*"))
            with tqdm(total=len(copy_files)) as pbar:
                with ThreadPoolExecutor(max_workers=16) as xec:
                    for file in subdir.glob("*"):
                        dst_file = dst_dir / str(file.absolute())[len(str(dst_base_dir.absolute()))+1:]
                        xec.submit(copy_file, str(file), str(dst_file), pbar)
        
        print(f"Merged base subdir into {dst_dir.name}.")
    

def render_batch(model, z, c, attr, batch_size):
    w = model.mapping(z, c, attr, truncation_psi=0.7)
    output = model.synthesis(w, c, noise_mode='const', return_masks=False,
                             neural_rendering_resolution=args.res,
                             return_uv_map=False)
    image_tensors = output["image"]
    images = [PIL.ImageOps.grayscale(PIL.Image.fromarray(normalized_torch_to_numpy_img(output["image"][i, ...]))) for i in range(output["image"].shape[0])]
    return images, image_tensors


def label_images(images, image_names, static_attributes, classifier, z, c, args):
    labels = { k: {} for k in image_names }
    if classifier is not None and static_attributes is not None:
        x = torchvision.transforms.functional.rgb_to_grayscale(images.to(device))
        if x.shape[2] != classifier.img_res or x.shape[3] != classifier.img_res:
            x = torchvision.transforms.functional.resize(x, [classifier.img_res, classifier.img_res])
        y_pred = classifier(x).to("cpu")
        y_pred_filtered = stat.take_from_attribute_tensor(y_pred, classifier.static_attributes, static_attributes)
        labels = stat.labels_from_attribute_tensor(y_pred_filtered, image_names, static_attributes)
    if args.store_latent:
        for i in range(images.shape[0]):
            labels[image_names[i]]["z"] = z[i].tolist()
            labels[image_names[i]]["c"] = c[i].tolist()
    return labels


def get_real_data_bins(dir, static_attributes):
    labels = None
    with open(dir / "labels.json", "r") as f:
        labels = json.load(f)
    bins = { attr_key: [[0 for _ in range(NUM_ATTR_BINS[attr_key])] for _ in range(stat.types[attr_key].dim)] for attr_key in static_attributes }
    for file_key in labels.keys():
        for attr_key in labels[file_key].keys():
            if not attr_key in static_attributes: continue
            if stat.types[attr_key].dtype == bool:
                if stat.types[attr_key].dim == 1:
                    bins[attr_key][0][int(labels[file_key][attr_key])] += 1
                else:
                    for i in range(stat.types[attr_key].dim):
                        bins[attr_key][i][int(labels[file_key][attr_key][i])] += 1
            else:
                if stat.types[attr_key].dim == 1:
                    bins[attr_key][0][get_bin(labels[file_key][attr_key], attr_key)] += 1
                else:
                    for i in range(stat.types[attr_key].dim):
                        bins[attr_key][i][get_bin(labels[file_key][attr_key][i], attr_key)] += 1
    return bins


def get_bin(val, attribute):
    normval = (val - stat.types[attribute].low) / (stat.types[attribute].high - stat.types[attribute].low)
    if normval <= 0.0: return 0
    if normval >= 1.0: return NUM_ATTR_BINS[attribute]-1
    return int(normval * NUM_ATTR_BINS[attribute])

def save_image(image, dst_file):
    image.save(dst_file, compress_level=0)
    
def copy_file(src, dst, pbar):
    shutil.copy2(src, dst)
    pbar.update(1)


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
    parser.add_argument("--labels", type=str, nargs="+", default=None)  # gradient over given attributes
    parser.add_argument("--classifier", type=str, default=None) # to specify labelling classifier for output images
    parser.add_argument("--store_latent", action="store_true", default=False)
    parser.add_argument("--nosubdir", action="store_true", default=False)
    
    parser.add_argument("--augment_distribution", action="store_true", default=False)
    
    parser.add_argument("--filter_bp", type=int, default=None)
    parser.add_argument("--filter_fh", type=int, default=None)
    parser.add_argument("--filter_gl", type=int, default=None)
    parser.add_argument("--filter_eo_min", type=float, default=None)
    parser.add_argument("--filter_eo_max", type=float, default=None)
    parser.add_argument("--filter_gz_pitch_min", type=float, default=None)
    parser.add_argument("--filter_gz_pitch_max", type=float, default=None)
    parser.add_argument("--filter_gz_yaw_min", type=float, default=None)
    parser.add_argument("--filter_gz_yaw_max", type=float, default=None)
    parser.add_argument("--filter_gz_deadzone", type=float, default=None)
    parser.add_argument("--filter_gz_radius", type=float, default=None)
    args = parser.parse_args()
    
    main(args)
