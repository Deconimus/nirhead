import os.path, pathlib, time, inspect, json, hashlib
from typing import Optional, List, Dict, Tuple
import numpy as np
import torch, torchvision
from torch import nn
from torch.utils.flop_counter import suffixes
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
    conditioning_map_dim: int = 16
    discriminator_fc_dim: int = 512
    discriminator_out_dim: int = 1
    generator_z_tanh: bool = True
    generator_random_concat: bool = False
    no_discriminator: bool = False
    add_image_mse: bool = False
    trainsets: Optional[List[Tuple[str, int]]] = field(default_factory=list)


class IdentifierGenerator(GaussianDiscriminator):
    
    def __init__(self, cfg: IdentifierConfig, g_cfg: GaussianDiscriminatorConfig, device):
        super(IdentifierGenerator, self).__init__(g_cfg)
        self.b4 = None
        self.cfg = cfg
        self.device = device
        
        in_channels = self.channels_dict[4]
        activation = "lrelu"
        epilogue_res = 4
        
        self.fromrgb = Conv2dLayer(cfg.img_ch, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=cfg.mbstd_group_size, num_channels=cfg.mbstd_num_channels) if cfg.mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + cfg.mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=g_cfg.conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (epilogue_res ** 2), cfg.latent_z_dim + cfg.pose_c_dim, activation="linear")
        #self.out = FullyConnectedLayer(cfg.latent_z_dim + cfg.pose_c_dim, cfg.latent_z_dim + cfg.pose_c_dim, activation="linear")
        
        if cfg.generator_random_concat:
            self.z_out = FullyConnectedLayer(cfg.latent_z_dim * 2, cfg.latent_z_dim, activation="linear")
    
    
    def forward(self, img: torch.Tensor, img_hash: torch.Tensor = None, update_emas=False, alpha_new_layers: float = 1, **block_kwargs):
        _ = update_emas  # unused
        assert(img is not None)
        
        img_hash = images_hash(img, img.device) if self.cfg.generator_random_concat else None
        
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
        if self.cfg.generator_z_tanh:
            z = torch.nn.functional.tanh(z)
        if self.cfg.generator_random_concat:
            z_concat_noise = torch.concat([z, img_hash], dim=1)
            z = self.z_out(z_concat_noise)
            if self.cfg.generator_z_tanh:
                z = torch.nn.functional.tanh(z)
                
        c = x[:, self.cfg.latent_z_dim:]
        
        # fix last row of cam_2_world matrix
        c[:, 12:15] = 0.0
        c[:, 15] = 1.0
        
        assert(z.shape[1] == self.cfg.latent_z_dim)
        assert(c.shape[1] == self.cfg.pose_c_dim)
        
        return z, c
    


class IdentifierDiscriminator(GaussianDiscriminator):
    
    def __init__(self, cfg: IdentifierConfig, d_cfg: GaussianDiscriminatorConfig, device):
        super(IdentifierDiscriminator, self).__init__(d_cfg)
        self.cfg = cfg
        self.device = device
        
        if cfg.conditioning_map_dim is None or cfg.conditioning_map_dim <= 0:
            self.b4 = None
            self.mapping = None
        
        in_channels = self.channels_dict[4]
        activation = "lrelu"
        epilogue_res = 4
        
        self.fromrgb = Conv2dLayer(cfg.img_ch, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=cfg.mbstd_group_size, num_channels=cfg.mbstd_num_channels) if cfg.mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + cfg.mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=d_cfg.conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (epilogue_res ** 2), cfg.discriminator_fc_dim, activation=activation)
        self.out = FullyConnectedLayer(cfg.discriminator_fc_dim, max(cfg.discriminator_out_dim, 1), activation="linear")
    
    
    def forward(self, img: torch.Tensor, c: torch.Tensor, update_emas=False, alpha_new_layers: float = 1, **block_kwargs):
        _ = update_emas  # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)
        
        if self.cfg.conditioning_map_dim is not None and self.cfg.conditioning_map_dim > 0:
            cmap = None
            if self.c_dim > 0:
                if self.disc_c_noise > 0: c += torch.randn_like(c) * c.std(0) * self.disc_c_noise
                cmap = self.mapping(None, c)
            x = self.b4(x, img, cmap)
        else:
            x = x.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            if self.mbstd is not None:
                x = self.mbstd(x)
            x = self.conv(x)
            x = self.fc(x.flatten(1))
            x = self.out(x)
        
        return x



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
        self.G = IdentifierGenerator(cfg, g_cfg, device).to(device)
        
        if not cfg.no_discriminator:
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
                cmap_dim = cfg.conditioning_map_dim if cfg.subject_hash is not None else 0, # Dimensionality of mapped conditioning label (output), None = default.
                #img_resolution = cfg.img_res,
                #img_channels = cfg.img_ch,
            )
            c_dim_map = { "binary": 8, "sha256": 32, None: 0 }
            d_cfg.c_dim = c_dim_map[cfg.subject_hash]
            d_cfg.img_resolution = cfg.img_res
            d_cfg.img_channels = cfg.img_ch
            d_cfg.pretrained_resolution = None
            self.D = IdentifierDiscriminator(cfg, d_cfg, device).to(device)
        
        
    def forward(self, img: torch.Tensor, img_hash: torch.Tensor = None):
        return self.G(img, img_hash)
    
    def discriminate(self, img: torch.Tensor, subject_labels: List[int]):
        if self.cfg.no_discriminator:
            return None
        
        subject_hash = None
        if self.cfg.subject_hash is not None:
            subject_hash = self.create_subject_hash(subject_labels)
            if self.device is not None:
                subject_hash = subject_hash.to(self.device)
        
        logits = self.D(img=img, c=subject_hash)
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
    
    if weights_file:
        with open(weights_file, "rb") as f:
            model.load_state_dict(torch.load(f, weights_only=True, map_location=device))
    
    suffix = (cfg.subject_hash if cfg.subject_hash is not None else "classes") if not cfg.no_discriminator else "imageloss"
    if cfg.generator_random_concat:
        suffix += "_noise"
    if not cfg.generator_z_tanh:
        suffix += "_notanh"
    if cfg.add_image_mse:
        suffix += "_addimagemse"
    
    name = "identifier_" + pathlib.Path(cfg.synth_model).name + "_" + str(int(time.time())) + "_" + suffix
    
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


def images_hash(imgs, device=None):
    imgs = imgs.cpu()
    hashes = []
    for i in range(imgs.shape[0]):
        hash = np.frombuffer(hashlib.sha512(np.array(imgs[i]).astype(np.uint16).tobytes()).digest(), dtype=np.uint8)
        hash = np.concatenate([np.frombuffer(hashlib.sha512(hash[i * 8:(i + 1) * 8].tobytes()).digest(), dtype=np.uint8) for i in range(8)], axis=0)
        hash = hash.astype(np.float32) / 127.5 - 1.0
        hashes.append(torch.from_numpy(hash))
    imgs_hash = torch.stack(hashes, dim=0)
    if device is not None:
        imgs_hash = imgs_hash.to(device)
    return imgs_hash
