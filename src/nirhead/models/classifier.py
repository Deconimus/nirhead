import time, inspect
from typing import Optional
import numpy as np
import torch, torchvision
from torch import nn
from tqdm import tqdm
from vit_pytorch import ViT, SimpleViT
from copy import deepcopy
from dataclasses import dataclass

from elias.config import Config
from eg3d.torch_utils import misc


@dataclass
class ClassifierConfig(Config):
    model: str = "simplevit"
    labels: Optional[list] = None
    img_res: int = 128
    crop_h: int = 0
    patch_size: int = 16
    vit_mlp_dim: int = 1024
    vit_dim: Optional[int] = None
    vit_heads: int = 8
    vit_depth: int = 4
    ch_base: int = 32
    epilogue: bool = False
    

def make_crop_module(img_res: int, crop, identity=True):
    crop = (crop, crop) if not isinstance(crop, tuple) else crop
    crop_y = img_res if crop[0] <= 0 else crop[0]
    crop_x = img_res if crop[1] <= 0 else crop[1]
    crop = (crop_y, crop_x)
    if crop[0] < img_res or crop[1] < img_res:
        return nn.Sequential(
            torchvision.transforms.CenterCrop(crop),
            torchvision.transforms.Resize((img_res, img_res)),
        )
    return nn.Identity() if identity else None


class Classifier(nn.Module):
    
    def __init__(self, img_res: int, img_ch: int, label_classes: list, crop=(0,0)):
        super().__init__()
        self.img_res = img_res
        self.img_ch = img_ch
        self.label_classes = [l.lower().strip() for l in label_classes]
        self.num_classes = len(label_classes)
        
        self.crop = (crop, crop) if not isinstance(crop, tuple) else crop
        crop_y = img_res if self.crop[0] <= 0 else self.crop[0]
        crop_x = img_res if self.crop[1] <= 0 else self.crop[1]
        self.crop = (crop_y, crop_x)
        
    def get_labels_tensor(self, label_tensor, label_classes):
        label_indices = [self.label_classes.index(l.lower().strip()) for l in label_classes]
        assert(len(label_indices) == len(label_classes))
        idx = torch.tensor(label_indices, device=label_tensor.device)
        idx = idx.reshape((1,-1)).repeat((label_tensor.shape[0], 1))
        return torch.take_along_dim(label_tensor, idx, dim=1)


class TinyVGG(Classifier):

    def __init__(self, img_res: int, input_channel: int, hidden_units: int, label_classes: list, crop=(0,0)):
        super().__init__(img_res, input_channel, label_classes, crop)

        blocks = []
        blocks.append(make_crop_module(img_res, crop))
        blocks.append(nn.Sequential(
            nn.Conv2d(in_channels=input_channel,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        ))
        blocks.append(nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ))
        self.cnn = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units**2 * self.img_res, out_features=self.num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.cnn(x)
        x = self.classifier(x)
        return x


class ResNetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride=2, num_convolutions=2, dtype=torch.float32):
        super(ResNetBlock, self).__init__()
        self.num_convolutions = num_convolutions
        self.stride = stride
        self.dtype = dtype

        conv_layers = []
        for i in range(self.num_convolutions):
            conv_layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                stride=(self.stride if i == self.num_convolutions-1 else 1),
                padding=1,
                dtype=self.dtype
            ))
            conv_layers.append(nn.BatchNorm2d(out_channels, dtype=self.dtype))
            if i < self.num_convolutions-1:
                conv_layers.append(nn.LeakyReLU())

        self.cnn = nn.Sequential(*conv_layers)

        self.downsample = None
        if self.stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, dtype=self.dtype),
                nn.BatchNorm2d(out_channels, dtype=self.dtype)
            )

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        if (x.dtype != self.dtype):
            x = x.to(dtype=self.dtype)
        residual = self.downsample(x) if self.downsample else x
        x = self.cnn(x)
        x += residual
        x = self.activation(x)
        return x


class ResNet(Classifier):

    def __init__(self, img_res: int, img_ch: int, label_classes: list, crop=(0, 0), ch_base=32, epilogue=False):
        super(ResNet, self).__init__(img_res, img_ch, label_classes, crop)
        self.ch_base = ch_base
        self.epilogue = epilogue

        img_res_log2 = int(np.log2(img_res))
        self.block_resolutions = [2 ** i for i in range(img_res_log2, 2 if self.epilogue else 3, -1)]

        blocks = []
        blocks.append(make_crop_module(img_res, crop))

        for idx, res in enumerate(self.block_resolutions):
            in_ch = self.ch_base * (self.img_res // (res * 2)) if idx > 0 else self.img_ch
            out_ch = self.ch_base * (self.img_res // res)
            stride = 1 if self.epilogue and idx == len(self.block_resolutions)-1 else 2 # last block won't downsample, output res is 8x8
            dtype = torch.float32 # torch.float32 if res <= 2**4 else torch.float16

            blocks.append(ResNetBlock(in_ch, out_ch, stride=stride, num_convolutions=2, dtype=dtype))

        blocks.append(nn.AvgPool2d(kernel_size=8, stride=1)) # downsample 8x8 to "flat" 1x1
        self.cnn = nn.Sequential(*blocks)

        num_features = out_ch # self.ch_base * (self.img_res // (self.block_resolutions[-1]))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=num_features, out_features=self.num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.cnn(x)
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        x = self.classifier(x)
        return x


class VIT(Classifier):

    def __init__(self, img_res: int, img_ch: int, label_classes: list, crop=(0, 0), patch_size=16, depth=6, heads=16, mlp_dim=1024, dim=None):
        super(VIT, self).__init__(img_res, img_ch, label_classes, crop)
        self.patch_size = patch_size
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dim = dim

        self.crop = make_crop_module(self.img_res, self.crop)
        self.vit = SimpleViT(
            image_size = self.img_res,
            channels = self.img_ch,
            patch_size = self.patch_size,
            num_classes = self.num_classes,
            dim = self.dim if self.dim else self.mlp_dim//2, #1024,
            depth = self.depth,
            heads = self.heads,
            mlp_dim = self.mlp_dim,
        )

    def forward(self, x: torch.Tensor):
        x = self.crop(x)
        x = self.vit(x)
        #print(x.shape)
        return x


def load_classification_model(cfg: ClassifierConfig, device: str, weights_file=None):
    model = None
    name = ""
    
    if cfg.model.lower() == "resnet":
        model = ResNet(img_res=cfg.img_res, img_ch=1, label_classes=cfg.labels, crop=(cfg.crop_h, 0), ch_base=cfg.ch_base, epilogue=cfg.epilogue).to(device)
        name = "resnet_chbase" + str(cfg.ch_base) + ("_epilogue" if cfg.epilogue else "") + (f"_croph{cfg.crop_h}" if cfg.crop_h else "")  + "_" + str(int(time.time()))
        
    elif cfg.model.lower() == "tvgg" or cfg.model.lower() == "tinyvgg":
        model = TinyVGG(img_res=cfg.img_res, input_channel=1, hidden_units=cfg.ch_base, label_classes=cfg.labels, crop=(cfg.crop_h, 0)).to(device)
        name = "tinyvgg_units" + str(cfg.ch_base) + (f"_croph{cfg.crop_h}" if cfg.crop_h else "")  + "_" + str(int(time.time()))
        
    elif cfg.model.lower() == "vit" or cfg.model.lower() == "simplevit":
        mlp_dim = cfg.vit_mlp_dim if cfg.vit_mlp_dim else cfg.ch_base
        dim = cfg.vit_dim if cfg.vit_dim else None
        heads = cfg.vit_heads if cfg.vit_heads else 8
        depth = cfg.vit_depth if cfg.vit_depth else 4
        
        model = VIT(img_res=cfg.img_res, img_ch=1, label_classes=cfg.labels, crop=(cfg.crop_h, 0), patch_size=cfg.patch_size, mlp_dim=mlp_dim, dim=dim, heads=heads, depth=depth).to(device)
        name = f"simplevit{cfg.patch_size}_mlpdim{mlp_dim}" + (f"_dim{dim}" if dim else "") + f"_heads{heads}_depth{depth}" + (f"_croph{cfg.crop_h}" if cfg.crop_h else "") + "_" + str(int(time.time()))
        
    if weights_file:
        with open(weights_file, "rb") as f:
            model.load_state_dict(torch.load(f, weights_only=True, map_location=device))
    
    return model, name

