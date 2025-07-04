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

from gghead.models.gaussian_discriminator import GaussianDiscriminator, GaussianDiscriminatorConfig, MappingNetworkConfig, DiscriminatorBlockConfig, DisciminatorEpilogueConfig

import nirhead.data.static_attributes as stat


@dataclass
class IdentifierConfig(Config):
    synth_model: str = None
    latent_z_dim: int = 512
    img_res: int = 256
    img_ch: int = 1
    mbstd_group_size: int = 4
    mbstd_num_channels: int = 1


class IdentifierGenerator(GaussianDiscriminator):
    
    def __init__(self, cfg: IdentifierConfig, discriminator_cfg: GaussianDiscriminatorConfig):
        super(IdentifierGenerator, self).__init__(discriminator_cfg)
        self.b4 = None
        self.cfg = cfg
        
        in_channels = self.channels_dict[4]
        self.fromrgb = Conv2dLayer(cfg.img_ch, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=cfg.mbstd_group_size, num_channels=cfg.mbstd_num_channels) if cfg.mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + cfg.mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), cfg.latent_z_dim, activation=activation)
        #self.out = FullyConnectedLayer(cfg.latent_z_dim, cfg.latent_z_dim)
        
    def forward(self, img: Dict, c, update_emas=False, alpha_new_layers: float = 1, **block_kwargs):
        img = img['image']
        _ = update_emas  # unused
        
        # extracting image features x via ResNet
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            if self.pretrained_resolution is not None and res > self.pretrained_resolution:
                # Smoothly blend in outputs of newly added layers (that are not trained yet)
                x, img = block(x, img, **block_kwargs)  # TODO: force x closer to 0 in the beginning? Then discriminator should be more similar to pre-trained
                x = alpha_new_layers * x
            else:
                x, img = block(x, img, **block_kwargs)
        
        # embedding x as latent vector output
        misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
        img = img.to(dtype=dtype)
        x = x + self.fromrgb(img)
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        #x = self.out(x)
        
        return x
    

class Identifier(nn.Module):
    
    def __init__(self, cfg: IdentifierConfig, device: str):
        super(Identifier, self).__init__()
        self.cfg = cfg
        self.device = device
        
        # our "generator" is actually a discriminator without the binary classifiaction, leaving the output at the required latent z dimensionality
        g_cfg = GaussianDiscriminatorConfig(
            mapping_network_config = None,
            block_config = DiscriminatorBlockConfig(),  # TODO
            epilogue_config = None,
            architecture = 'resnet',  # Architecture: 'orig', 'skip', 'resnet'.
            channel_base = 32768,  # Overall multiplier for the number of channels.
            channel_max = 512,  # Maximum number of channels in any layer.
            num_fp16_res = 4,  # Use FP16 for the N highest resolutions.
            conv_clamp = 256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
            disc_c_noise = 0,
            c_dim = 0,
            cmap_dim = 0,  # Dimensionality of mapped conditioning label, None = default.
            img_resolution = cfg.img_res,
            img_channels = cfg.img_ch,
        )
        self.G = IdentifierGenerator(cfg, g_cfg)
        
        # our discriminator, which checks if generated or real images are believable images for the target subject_label (during training)
        # we use the conditioning vector c to condition the model for our subject labels (hashes of strings)
        d_cfg = GaussianDiscriminatorConfig(
            mapping_network_config = MappingNetworkConfig(), # TODO
            block_config = DiscriminatorBlockConfig(), # TODO
            epilogue_config = DisciminatorEpilogueConfig(), # TODO
            architecture = 'resnet',  # Architecture: 'orig', 'skip', 'resnet'.
            channel_base = 32768,  # Overall multiplier for the number of channels.
            channel_max = 512,  # Maximum number of channels in any layer.
            num_fp16_res = 4,  # Use FP16 for the N highest resolutions.
            conv_clamp = 256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
            disc_c_noise = 0,
            c_dim = 32,
            cmap_dim = 16, # Dimensionality of mapped conditioning label, None = default.
            img_resolution = cfg.img_res,
            img_channels = cfg.img_ch,
        )
        self.D = GaussianDiscriminator(d_cfg)
        
    def forward(self, img: torch.Tensor):
        return self.G(img)
    
    def discriminate(self, img: torch.Tensor, subject_label: str):
        # create sha256 hash of subject string label, map to tensor with each byte interpreted as float32 in value range [-1,1]
        subject_hash = torch.from_numpy(np.frombuffer(hashlib.sha256(subject_label.encode("utf-8")), dtype=np.uint8).astype(np.float32) / 127.5 - 1.0)
        if device is not None:
            subject_hash = subject_hash.to(device)
        return self.D(img={"image": img}, c=subject_hash)
    
    

def load_identifier_model(cfg: ClassifierConfig, device: str, weights_file=None):
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
