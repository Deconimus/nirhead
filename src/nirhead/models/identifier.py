import os.path, pathlib, time, inspect, json, hashlib
from typing import Optional, List, Dict, Tuple
import numpy as np
import torch, torchvision
from torch import nn
from tqdm import tqdm
from vit_pytorch import ViT, SimpleViT
from copy import deepcopy
from dataclasses import dataclass, field

from elias.config import Config
from eg3d.torch_utils import misc

from gghead.models.gaussian_discriminator import GaussianDiscriminator, GaussianDiscriminatorConfig, MappingNetworkConfig, DiscriminatorBlockConfig, DiscriminatorEpilogueConfig
from nirhead.eg3d.training.networks_stylegan2 import Conv2dLayer, MinibatchStdLayer, FullyConnectedLayer

import nirhead.data.static_attributes as stat


@dataclass
class IdentifierConfig(Config):
    synth_model: str = None
    latent_z_dim: int = 512
    pose_c_dim: int = 25
    img_res: int = 256
    img_ch: int = 1
    mbstd_group_size: int = 4
    mbstd_num_channels: int = 1
    subject_hash: str = "binary"


class IdentifierGenerator(GaussianDiscriminator):
    
    def __init__(self, cfg: IdentifierConfig, discriminator_cfg: GaussianDiscriminatorConfig):
        super(IdentifierGenerator, self).__init__(discriminator_cfg)
        self.b4 = None
        self.cfg = cfg
        
        in_channels = self.channels_dict[4]
        activation = "lrelu"
        epilogue_res = 4
        
        self.fromrgb = Conv2dLayer(cfg.img_ch, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=cfg.mbstd_group_size, num_channels=cfg.mbstd_num_channels) if cfg.mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + cfg.mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=discriminator_cfg.conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (epilogue_res ** 2), cfg.latent_z_dim + cfg.pose_c_dim, activation=activation)
        #self.out = FullyConnectedLayer(cfg.latent_z_dim + cfg.pose_c_dim, cfg.latent_z_dim + cfg.pose_c_dim, activation="linear")
        self.c_out = FullyConnectedLayer(cfg.pose_c_dim, cfg.pose_c_dim, activation="linear")
        
    def forward(self, img: torch.Tensor, update_emas=False, alpha_new_layers: float = 1, **block_kwargs):
        _ = update_emas  # unused
        assert(img is not None)
        
        # extracting image features x via ResNet
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)
            #print(f"b{res}")
        
        x = x.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        #if self.architecture == "skip":
        #    misc.assert_shape(img, [None, self.cfg.img_ch, self.cfg.img_res, self.cfg.img_res])
        #    img = img.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        #    x = x + self.fromrgb(img)
        
        # embedding x as latent vector output
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        #x = self.out(x)
        
        z = x[:, :self.cfg.latent_z_dim]
        assert(z.shape[1] == self.cfg.latent_z_dim)
        
        c = x[:, self.cfg.latent_z_dim:]
        c = torch.nn.functional.leaky_relu(c, inplace=False)
        c = self.c_out(c)
        assert(c.shape[1] == self.cfg.pose_c_dim)
        
        # fix last row of cam_2_world matrix
        c[:, 12:15] = 0.0
        c[:, 15] = 1.0
        
        return z, c
    

class Identifier(nn.Module):
    
    def __init__(self, cfg: IdentifierConfig, device: str):
        super(Identifier, self).__init__()
        self.cfg = cfg
        self.device = device
        
        # our "generator" is actually a discriminator without the binary classifiaction, leaving the output at the required latent z dimensionality
        g_cfg = GaussianDiscriminatorConfig(
            mapping_network_config = MappingNetworkConfig(),
            block_config = DiscriminatorBlockConfig(),  # TODO
            epilogue_config = DiscriminatorEpilogueConfig(),
            architecture = 'resnet',  # Architecture: 'orig', 'skip', 'resnet'.
            channel_base = 32768,  # Overall multiplier for the number of channels.
            channel_max = 512,  # Maximum number of channels in any layer.
            num_fp16_res = 4,  # Use FP16 for the N highest resolutions.
            conv_clamp = 256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
            disc_c_noise = 0,
            #c_dim = 0,
            cmap_dim = 0,  # Dimensionality of mapped conditioning label, None = default.
            #img_resolution = cfg.img_res,
            #img_channels = cfg.img_ch,
        )
        g_cfg.c_dim = 0
        g_cfg.img_resolution = cfg.img_res
        g_cfg.img_channels = cfg.img_ch
        g_cfg.pretrained_resolution = None
        self.G = IdentifierGenerator(cfg, g_cfg).to(device)
        
        # our discriminator, which checks if generated or real images are believable images for the target subject_label (during training)
        # we use the conditioning vector c to condition the model for our subject labels (hashes of strings)
        d_cfg = GaussianDiscriminatorConfig(
            mapping_network_config = MappingNetworkConfig(), # TODO
            block_config = DiscriminatorBlockConfig(), # TODO
            epilogue_config = DiscriminatorEpilogueConfig(), # TODO
            architecture = 'resnet',  # Architecture: 'orig', 'skip', 'resnet'.
            channel_base = 32768,  # Overall multiplier for the number of channels.
            channel_max = 512,  # Maximum number of channels in any layer.
            num_fp16_res = 4,  # Use FP16 for the N highest resolutions.
            conv_clamp = 256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
            disc_c_noise = 0,
            #c_dim = 8, # Conditioning vector input dimensionality
            cmap_dim = 16, # Dimensionality of mapped conditioning label (output), None = default.
            #img_resolution = cfg.img_res,
            #img_channels = cfg.img_ch,
        )
        d_cfg.c_dim = 8 if self.cfg.subject_hash == "binary" else 32
        d_cfg.img_resolution = cfg.img_res
        d_cfg.img_channels = cfg.img_ch
        d_cfg.pretrained_resolution = None
        self.D = GaussianDiscriminator(d_cfg).to(device)
        
    def forward(self, img: torch.Tensor):
        return self.G(img)
    
    def discriminate(self, img: torch.Tensor, subject_labels: List[int]):
        subject_hash = self.create_subject_hash(subject_labels)
        if self.device is not None:
            subject_hash = subject_hash.to(self.device)
        logits = self.D(img={"image": img}, c=subject_hash)
        return torch.squeeze(logits)
    
    # adjust c_dim in discriminator when changing this function!
    def create_subject_hash(self, subject_labels: List[int]):
        subject_hashes = []
        if self.cfg.subject_hash == "sha256":
            # create sha256 hash of subject string label, map to tensor with each byte interpreted as float32 in value range [-1,1]
            subject_hashes = [torch.from_numpy(np.frombuffer(hashlib.sha256(str(subject_label).encode("utf-8")).digest(), dtype=np.uint8).astype(np.float32) / 127.5 - 1.0) for subject_label in subject_labels]
        elif self.cfg.subject_hash == "binary":
            # binary encoding of subject number
            subject_hashes = [torch.tensor([float(c) for c in bin(subject_label)[2:].zfill(8)], dtype=torch.float32) for subject_label in subject_labels]
        
        return torch.stack(subject_hashes, dim=0)
        
    
def load_identifier_model(cfg: IdentifierConfig, device: str, weights_file=None):
    model = Identifier(cfg=cfg, device=device)
    name = "identifier_"+pathlib.Path(cfg.synth_model).name+"_"+str(int(time.time()))
    
    if weights_file:
        with open(weights_file, "rb") as f:
            model.load_state_dict(torch.load(f, weights_only=True, map_location=device))
            
    return model, name


def load_identifier_model_dir(identifier_dir, device: str, checkpoint: int = -1):
    identifier_dir = pathlib.Path(identifier_dir) if isinstance(identifier_dir, str) else identifier_dir
    if os.path.isdir(identifier_dir / "checkpoints") and len(list((identifier_dir / "checkpoints").glob("*.pth"))) > 0:
        checkpoint_list = list((identifier_dir / "checkpoints").glob("*.pth"))
        checkpoint_list.sort(key=lambda f: int(f.name[11:-4]))
        checkpoint_dict = {int(file.name[11:-4]): file for file in checkpoint_list}
        if checkpoint < 0:
            identifier_weights = checkpoint_list[checkpoint]
        else:
            identifier_weights = checkpoint_dict[checkpoint]
    else:
        identifier_weights = identifier_dir / "weights.pth"
    
    with open(identifier_dir / "args.json", "r") as f:
        identifier_cfg = identifier_dir.from_json(json.load(f))
    
    return load_identifier_model(
        cfg=identifier_cfg,
        weights_file=identifier_weights,
        device=device,
    )
