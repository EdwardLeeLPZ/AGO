# ------------------------------------------------------------------------
# AGO
# Copyright (c) 2025 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------

import torch


def ray_sample(coords, center, grid_shape):
    '''Sample points along the ray from center to coords
    Args:
        coords: (n, 3) tensor, coordinates of the target points
        center: (3,) tensor, coordinates of the center point
        grid_shape: (3,) tensor, shape of the grid
    Returns:
        ray_coords: (m, 3) tensor, unique coordinates of the points along the rays
    '''
    diff = torch.abs(coords - center).float() # (n, 3)
    sign = torch.sign(coords - center) # (n, 3)
    primary_axis = torch.argmax(diff, axis=-1) # (n,)

    step = diff[torch.arange(diff.shape[0]), primary_axis] # (n,)
    delta = sign * diff / step[:, None] # (n, 3)

    max_step = step.max()
    ray_coords = center + torch.arange(0, max_step, 1, device=coords.device).reshape(-1, 1, 1) * delta.reshape(1, -1, 3) # (max_step, n, 3)

    # filter out points larger than step
    valid = torch.arange(0, max_step, 1, device=coords.device).reshape(-1, 1) < step.reshape(1, -1)
    ray_coords = ray_coords[valid]

    # remove repeated coordinates and out of range coordinates
    ray_coords = ray_coords.round().long().reshape(-1, 3) # (max_step * n, 3)
    ray_coords = torch.unique(ray_coords, dim=0).long() # (m, 3)
    ray_coords = ray_coords[(ray_coords >= 0).all(dim=-1) & (ray_coords < (grid_shape.to(coords.device)-1)).all(dim=-1)]
    return ray_coords