import torch
import numpy as np
import yaml
from collections import namedtuple
from torch_geometric.data import Data
import os
import torch.nn as nn

def construct_rotation_matrix(X):
    """
    Constructs an accurate rotation matrix that rotates the vector X onto the X-axis.

    Args:
    X (torch.Tensor): A 3-dimensional vector in double precision.

    Returns:
    torch.Tensor: A 3x3 rotation matrix in double precision.
    """
    # Converting to double precision for higher accuracy
    X = X.float()

    # Normalizing X
    norm_X = torch.norm(X)
    if norm_X == 0:
        return torch.eye(3, dtype=torch.float)  # Return identity matrix if X is a zero vector

    # Target vector on the X-axis
    target = torch.tensor([norm_X, 0, 0], dtype=torch.float)

    # Rotation axis (cross product of X and target)
    rotation_axis = torch.cross(X, target)
    rotation_axis_norm = torch.norm(rotation_axis)
    if rotation_axis_norm == 0:
        return torch.eye(3, dtype=torch.float)  # Return identity matrix if X is already on the X-axis

    rotation_axis_normalized = rotation_axis / rotation_axis_norm

    # The angle of rotation is the arccosine of the dot product of normalized X and target
    angle = torch.acos(torch.dot(X / norm_X, target / norm_X))

    # Constructing the rotation matrix using Rodrigues' rotation formula
    K = torch.tensor([
        [0, -rotation_axis_normalized[2], rotation_axis_normalized[1]],
        [rotation_axis_normalized[2], 0, -rotation_axis_normalized[0]],
        [-rotation_axis_normalized[1], rotation_axis_normalized[0], 0]
    ], dtype=torch.float)
    rotation_matrix = torch.eye(3, dtype=torch.float) + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)

    return rotation_matrix

def norm_torch(data,scale=3):
    upper = data.mean() + scale*data.std()
    lower = data.mean() - scale*data.std()      
    data = torch.clamp(data,lower,upper)
    if data.std() != 0:
        data = (data - data.mean())/data.std()
    else:
        data = (data - data.mean())
    return data

