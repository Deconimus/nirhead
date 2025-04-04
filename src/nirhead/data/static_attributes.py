import torch
import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class StaticAttributeTypeInfo:
    name: str
    short: str
    dtype: type
    dim: int
    low: Optional[float] = None
    high: Optional[float] = None
    

types_list = [
    StaticAttributeTypeInfo("bright_pupil", "bp", bool,  1),
    StaticAttributeTypeInfo("glasses",      "g",  bool,  1),
    StaticAttributeTypeInfo("eye_open",     "eo", float, 1, low=0.0, high=1.0),
]
types = { attr.name: attr for attr in types_list }
_types_indices = { attr.name: idx for idx, attr in enumerate(types_list) }

default_attribute_lambdas = None # { attr.name: 1.0 for attr in types_list }


def attributes_loss(attr_pred: torch.Tensor,
                    attr_truth: torch.Tensor,
                    static_attributes: List[str],
                    #attribute_lambdas: Optional[List[float]] = default_attribute_lambdas
                    ):
    num_attributes = len(static_attributes)
    num_binary_attributes = get_num_binary_attributes(static_attributes)
    
    lambda_binary = num_binary_attributes / num_attributes
    binary_loss = 0.0
    if num_binary_attributes > 0:
        binary_attr_pred  = take_binary_from_attribute_tensor(attr_pred, static_attributes)
        binary_attr_truth = take_binary_from_attribute_tensor(attr_truth, static_attributes)
        binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(binary_attr_pred, binary_attr_truth) * lambda_binary
    
    discrete_loss = 0.0
    if num_binary_attributes < num_attributes:
        idx_off = 0
        for attr in static_attributes:
            dim = types[attr].dim
            
            loss_fun = None # or fun loss?
            if types[attr].dtype == float or types[attr].dtype == int:
                loss_fun = torch.nn.functional.mse_loss
                
            if loss_fun is not None:
                attr_lambda = 1.0 # if attribute_lambdas is None or attr not in attribute_lambdas.keys() else attribute_lambdas[attr]
                discrete_loss += loss_fun(attr_pred[:,idx_off:idx_off+dim], attr_truth[:,idx_off:idx_off+dim]) * attr_lambda
                
            idx_off += dim
            
        discrete_loss *= (num_attributes - num_binary_attributes) / num_attributes
    
    #print(f"{binary_loss} {discrete_loss}")
    return binary_loss + discrete_loss


def take_from_attribute_tensor(attributes_tensor_src, static_attributes_src, static_attributes_dst):
    if static_attributes_src == static_attributes_dst:
        return attributes_tensor_src
    assert (all([(attr in static_attributes_src) for attr in static_attributes_dst]))
    attr_dim_dst = attributes_dim(static_attributes_dst)
    
    attr_offsets_src = {}
    idx_offset = 0
    for attr in static_attributes_src:
        attr_offsets_src[attr] = idx_offset
        idx_offset += types[attr].dim
    
    attr_indices = []
    for attr in static_attributes_dst:
        for i in range(types[attr].dim):
            attr_indices.append(attr_offsets_src[attr] + i)
    assert (len(attr_indices) == attr_dim_dst)
    
    idx = torch.tensor(attr_indices, device=attributes_tensor_src.device, dtype=torch.long)
    idx = idx.reshape((1, -1)).repeat((attributes_tensor_src.shape[0], 1))
    
    return torch.take_along_dim(attributes_tensor_src, idx, dim=1)

    
def take_binary_from_attribute_tensor(attr_tensor: torch.Tensor, static_attributes: List[str]):
    if all([(types[attr].dtype == bool) for attr in static_attributes]):
        return attr_tensor
    
    attr_indices = []
    idx_offset = 0
    for attr in static_attributes:
        if types[attr].dtype == bool:
            for i in range(types[attr].dim):
                attr_indices.append(idx_offset + i)
        idx_offset += types[attr].dim
    
    idx = torch.tensor(attr_indices, device=attr_tensor.device, dtype=torch.long)
    idx = idx.reshape((1, -1)).repeat((attr_tensor.shape[0], 1))
    
    return torch.take_along_dim(attr_tensor, idx, dim=1)


def random_attribute_tensor(static_attributes: List[str], size: int = 1, device: Optional[str] = None):
    tensors = []
    
    for attr in static_attributes:
        shape = (size, types[attr].dim)
        low = types[attr].low
        high = types[attr].high
        
        if types[attr].dtype == bool:
            t = (torch.rand(shape, device=device, dtype=torch.float32) + 0.5).int().float()  # .bool().float() is buggy
        elif types[attr].dtype == int:
            l = int(low) if low is not None else int(-(2 ** 31))
            h = int(high) if low is not None else int((2 ** 31) - 1)
            t = torch.randint(l, h + 1, shape, device=device, dtype=torch.int32)
        elif types[attr].dtype == float:
            if low is not None and high is not None:
                t = torch.rand(shape, device=device, dtype=torch.float32) * (high - low) + low
            else:
                t = torch.randn(shape, device=device, dtype=torch.float32)
                if low is not None or high is not None:
                    t = torch.clamp(t, min=low, max=high)
        else:
            raise NotImplementedError(f"{types[attr].dtype} not supported!")
        tensors.append(t)
    
    tensor = torch.cat(tensors, dim=1)
    if size <= 1:
        tensor = tensor.squeeze()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def attributes_dim(static_attributes: Optional[List[str]]):
    if static_attributes is None:
        return 0
    return sum([types[attr].dim for attr in static_attributes])


def normalize_attributes_list(static_attributes: Optional[List[str]]):
    if static_attributes is None:
        return None
    attrs = [attr.lower().strip() for attr in static_attributes]
    attrs.sort(key=lambda x: _types_indices[x])
    return attrs


def get_num_binary_attributes(static_attributes: List[str]):
    return len([attr for attr in static_attributes if types[attr].dtype == bool])
