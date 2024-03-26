from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


def convert_array_to_torch(array, device: Optional[torch.device], dtype: Optional[torch.dtype] = None):
    if isinstance(array, torch.Tensor):
        return array.to(device)
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).to(device).to(dtype)
    else:
        return torch.tensor(array, device=device, dtype=dtype)


def thetaphi2xyz(
    thetaphi: Union[torch.Tensor, np.ndarray],
    normal: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = [0.0, 0.0, 1.0],
    tangent: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = [1.0, 0.0, 0.0],
    reverse_phi: bool = False,
    assume_normalized: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    thetaphi : [theta, phi]
    normal: [x, y, z], default [0, 0, 1]
    tangent: [x, y, z], default [1, 0, 0]
    reverse_theta: if set to True, phi positive is clockwise.
    assume_normalized: if set to True, normal and tangent are assumed to be normalized.
    return : [x, y, z]
    """
    if isinstance(thetaphi, torch.Tensor):
        device = thetaphi.device
        dtype = thetaphi.dtype
        normal = convert_array_to_torch(normal, device=device, dtype=dtype)
        tangent = convert_array_to_torch(tangent, device=device, dtype=dtype)
        module = torch
    else:
        normal = np.array(normal)
        tangent = np.array(tangent)
        module = np
    if not assume_normalized:
        normal = normalize(normal)
        tangent = normalize(tangent)
    binormal = module.cross(normal, tangent)
    if reverse_phi:
        binormal *= -1
    xyz = module.cos(thetaphi[..., 0:1]) * normal
    sin_theta = module.sin(thetaphi[..., 0:1])
    xyz += sin_theta * module.cos(thetaphi[..., 1:2]) * tangent
    xyz += sin_theta * module.sin(thetaphi[..., 1:2]) * binormal
    return xyz


def xyz2thetaphi(
    xyz: Union[torch.Tensor, np.ndarray],
    normal: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = [0.0, 0.0, 1.0],
    tangent: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = [1.0, 0.0, 0.0],
    reverse_phi: bool = False,
    assume_normalized: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    xyz : [x, y, z]
    normal: [x, y, z], default [0, 0, 1]
    tangent: [x, y, z], default [1, 0, 0]
    reverse_theta: if set to True, phi positive is clockwise.
    assume_normalized: if set to True, xyz, normal and tangent are assumed to be normalized.
    return : [theta, phi] (theta in [0, pi], phi in (-pi, pi))
    """
    if isinstance(xyz, torch.Tensor):
        device = xyz.device
        dtype = xyz.dtype
        normal = convert_array_to_torch(normal, device=device, dtype=dtype)
        tangent = convert_array_to_torch(tangent, device=device, dtype=dtype)
        module = torch
    else:
        normal = np.array(normal)
        tangent = np.array(tangent)
        module = np
    if not assume_normalized:
        xyz = normalize(xyz)
        normal = normalize(normal)
        tangent = normalize(tangent)
    binormal = module.cross(normal, tangent)
    if reverse_phi:
        binormal *= -1
    theta = module.arccos(module.matmul(xyz, normal[..., None]))[..., 0]
    phi = module.arctan2(module.matmul(xyz, binormal[..., None]), module.matmul(xyz, tangent[..., None]))[..., 0]
    return module.stack((theta, phi), -1)


def normalize(
    xyz: Union[torch.Tensor, np.ndarray],
    dim: int = -1,
    eps: float = 1e-12,
) -> Union[torch.Tensor, np.ndarray]:
    assert xyz.shape[-1] == 3
    if isinstance(xyz, torch.Tensor):
        return torch.nn.functional.normalize(xyz, dim=dim, eps=eps)
    else:
        length = np.linalg.norm(xyz, axis=dim, keepdims=True)
        length = np.clip(length, eps, None)
        return xyz / length


def mirmap2envmap(
    mirmap: torch.Tensor,
    output_shape: tuple,
    view: Union[torch.Tensor, List[float]] = [0, 0, 1],
    top: Union[torch.Tensor, List[float]] = [0, 1, 0],
    envmap_zenith: Union[torch.Tensor, List[float]] = [0, 1, 0],
    envmap_left_edge: Union[torch.Tensor, List[float]] = [0, 0, -1],
    reverse_azimuth: bool = True,
    log_scale_interpolation: bool = False,
) -> torch.Tensor:
    assert view == [0, 0, 1], "now support [0,0,1] view direction"
    device = mirmap.device
    dtype = mirmap.dtype
    view = convert_array_to_torch(view, device=device, dtype=dtype)
    top = convert_array_to_torch(top, device=device, dtype=dtype)
    height, width = mirmap.shape[-2:]
    OH, OW = output_shape
    theta = (torch.arange(OH, device=device) + 0.5) * (torch.pi / OH)
    phi = (torch.arange(OW, device=device) + 0.5) * (torch.pi * 2 / OW)
    if reverse_azimuth:
        phi = -phi
    thetaphi = torch.stack(torch.meshgrid(theta, phi, indexing="ij"), axis=-1)
    xyz = thetaphi2xyz(thetaphi, normal=envmap_zenith, tangent=envmap_left_edge)
    normal_map = xyz2thetaphi(normalize(xyz + view), normal=top, tangent=view)
    u = normal_map[..., 1] * (2 / torch.pi)
    v = normal_map[..., 0] * (2 / torch.pi) - 1
    uv = torch.stack([u, v], axis=-1)
    if log_scale_interpolation:
        mirmap = torch.log(mirmap.clip(1e-7))
    envmap = torch.nn.functional.grid_sample(
        mirmap,
        uv[None].expand(mirmap.size(0), -1, -1, -1),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    if log_scale_interpolation:
        envmap = torch.exp(envmap)
    return envmap


def gen_sphere_normals_realcentering(radius, edge=0):
    # real centering
    """Generate a set of normals of a spherical object from an orthographic camera."""
    normals = np.zeros((radius * 2, radius * 2, 3), dtype=np.float32)
    x = np.linspace(-radius + 0.5, radius - 0.5, num=2 * radius, endpoint=True)
    y = np.linspace(radius - 0.5, -radius + 0.5, num=2 * radius, endpoint=True)
    x, y = np.meshgrid(x, y)

    zsq = radius**2 - (x**2 + y**2)

    normals[..., 0] = x
    normals[..., 1] = y
    normals[zsq >= 0.0, 2] = np.sqrt(zsq[zsq >= 0.0])
    normals[...] /= np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
    normals[zsq < 0.0] = 0.0

    xx, yy = np.ogrid[0 : radius * 2, 0 : radius * 2]
    xx, yy = xx + 0.5, yy + 0.5
    mask = ((xx - radius) ** 2 + (yy - radius) ** 2) <= ((radius - edge) * (radius - edge))

    return normals * mask[..., None], mask


def refmap2refimg_torch(refmap: torch.Tensor, radius: int = None, return_mask: bool = False) -> torch.Tensor:
    """
    input: [(Batch), Channel, Height, Width]
    return: [(Batch), Channel, Height, Width]
    """
    if radius is None:
        radius = max(refmap.shape[-2:])
    res = radius * 2
    dtype = refmap.dtype
    device = refmap.device
    height, width = refmap.shape[-2:]
    sphere_normal_map, mask = gen_sphere_normals_realcentering(radius)  # x, y, z
    sphere_normal_map = torch.from_numpy(sphere_normal_map).to(device=device, dtype=dtype)
    mask = torch.from_numpy(mask).to(device=device)
    uv = xyz2thetaphi(sphere_normal_map[mask, :], [0, 1, 0], [-1, 0, 0])  # [masked HxW, 2(theta, phi)]
    uv = uv.flip(-1) * (2 / torch.pi) - 1
    batch_flag = True
    if refmap.ndim == 3:
        batch_flag = False
        refmap = refmap[None]
    mirimg = torch.zeros((refmap.shape[0], refmap.shape[-3], res, res), dtype=dtype, device=device)
    mirimg[:, :, mask] = torch.nn.functional.grid_sample(
        refmap, uv.expand(refmap.size(0), 1, -1, -1), mode="bilinear", padding_mode="border", align_corners=False
    )[:, :, 0, :]
    if not batch_flag:
        mirimg = mirimg[0]
    if return_mask:
        return mirimg, mask
    return mirimg


def envmap2mirmap(
    envmap: torch.Tensor,
    output_shape: Tuple[int, int],
    flip_horizontal: bool = False,
    view_from: Union[torch.Tensor, List[float]] = [1, 0, 0],
    top: Union[torch.Tensor, List[float]] = [0, 1, 0],
    envmap_zenith: Union[torch.Tensor, List[float]] = [0, 1, 0],
    envmap_left_edge: Union[torch.Tensor, List[float]] = [0, 0, -1],
    reverse_azimuth_envmap: bool = True,
    mitigate_aliasing: bool = True,
    log_scale_interpolation: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    device = envmap.device
    dtype = envmap.dtype
    view_from = convert_array_to_torch(view_from, device=device, dtype=dtype)
    top = convert_array_to_torch(top, device=device, dtype=dtype)
    if (torch.einsum("...i, ...i -> ...", view_from, top) != 0).any():
        top = normalize(torch.cross(torch.cross(view_from, top), view_from))
    height, width = envmap.shape[-2:]
    OH, OW = output_shape
    if mitigate_aliasing:
        H = W = min(height, width) if OH < height else max(OH, OW)
    else:
        H, W = OH, OW
    theta = (torch.arange(H, device=device, dtype=dtype) + 0.5) * (torch.pi / H)
    phi = (torch.arange(W, device=device, dtype=dtype) - (W - 1) / 2) * (torch.pi / W)
    thetaphi = torch.stack(torch.meshgrid(theta, phi, indexing="ij"), -1)
    xyz = thetaphi2xyz(thetaphi, normal=top, tangent=view_from, reverse_phi=flip_horizontal)
    xyz_env = normalize(2 * torch.matmul(xyz, view_from[..., None]) * xyz - view_from)
    thetaphi_env = xyz2thetaphi(xyz_env, normal=envmap_zenith, tangent=envmap_left_edge, reverse_phi=reverse_azimuth_envmap)
    u = thetaphi_env[..., 1] % (2 * torch.pi) / torch.pi - 1
    v = thetaphi_env[..., 0] * (2 / torch.pi) - 1
    uv = torch.stack([u, v], axis=-1)
    if uv.dim() == 3:
        uv = uv[None].expand(envmap.size(0), -1, -1, -1)
    if log_scale_interpolation:
        envmap = torch.log(envmap.clip(1e-7))
    mirmap = torch.nn.functional.grid_sample(envmap, uv, mode="bilinear", padding_mode="border", align_corners=False)
    mirmap = torch.nn.functional.adaptive_avg_pool2d(mirmap, output_shape)
    if log_scale_interpolation:
        mirmap = torch.exp(mirmap)
    return mirmap


def mirimg2envmap(
    refimg: torch.Tensor,
    output_shape: Tuple[int, int],
    view_from: Union[torch.Tensor, List[float]] = [0, 0, 1],
    top: Union[torch.Tensor, List[float]] = [0, 1, 0],
    envmap_zenith: Union[torch.Tensor, List[float]] = [0, 1, 0],
    envmap_left_edge: Union[torch.Tensor, List[float]] = [0, 0, -1],
    reverse_azimuth: bool = True,
    log_scale_interpolation: bool = False,
) -> torch.Tensor:
    """
    realcentering
    refimg: [BS, channle, Height, Width]
    output: [BS, channle, Height, Width]
    """
    device = refimg.device
    dtype = refimg.dtype
    view_from = convert_array_to_torch(view_from, device=device, dtype=dtype)
    top = convert_array_to_torch(top, device=device, dtype=dtype)
    if (torch.einsum("...i, ...i -> ...", view_from, top) != 0).any():
        top = normalize(torch.cross(torch.cross(view_from, top), view_from))
    OH, OW = output_shape
    theta = (torch.arange(OH, device=device, dtype=dtype) + 0.5) * (torch.pi / OH)
    phi = (torch.arange(OW, device=device, dtype=dtype) + 0.5) * (torch.pi * 2 / OW)
    thetaphi = torch.stack(torch.meshgrid(theta, phi, indexing="ij"), axis=-1)
    xyz = thetaphi2xyz(thetaphi, normal=envmap_zenith, tangent=envmap_left_edge, reverse_phi=reverse_azimuth)
    normal_map = xyz2thetaphi(normalize(xyz + view_from), normal=top, tangent=torch.cross(view_from, top))
    normal_map = normal_map - (torch.pi / 2)
    theta, phi = normal_map[..., 0], normal_map[..., 1]
    v = torch.sin(theta)
    u = torch.cos(theta) * torch.sin(phi)
    uv = torch.stack([u, v], axis=-1)
    if uv.dim() == 3:
        uv = uv[None].expand(refimg.size(0), -1, -1, -1)
    if log_scale_interpolation:
        refimg = torch.log(refimg.clip(1e-7))
    envmap = torch.nn.functional.grid_sample(refimg, uv, mode="bilinear", padding_mode="border", align_corners=False)
    if log_scale_interpolation:
        envmap = torch.exp(envmap)
    return envmap


def mirimg2envmap_numpy(
    refimg: np.ndarray,
    output_shape: tuple,
    view_from: Union[np.ndarray, List[float]] = [0, 0, 1],
    top: Union[np.ndarray, List[float]] = [0, 1, 0],
    envmap_zenith: Union[np.ndarray, List[float]] = [0, 1, 0],
    envmap_left_edge: Union[np.ndarray, List[float]] = [0, 0, -1],
    reverse_azimuth: bool = True,
    log_scale_interpolation: bool = False,
) -> np.ndarray:
    """
    realcentering
    refimg: [Height, Width, Channel]
    output: [Height, Width, Channel]
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    refimg = torch.from_numpy(refimg).permute(2, 0, 1)[None].to(device)  # [1, C, H, W]
    envmap = mirimg2envmap(
        refimg,
        output_shape,
        view_from=view_from,
        top=top,
        envmap_zenith=envmap_zenith,
        envmap_left_edge=envmap_left_edge,
        reverse_azimuth=reverse_azimuth,
        log_scale_interpolation=log_scale_interpolation,
    )
    return envmap[0].permute(1, 2, 0).cpu().numpy()


def rotate_envmap(
    envmap: torch.Tensor,
    src_envmap_zenith: Union[np.ndarray, torch.Tensor, List[float]] = [0, 1, 0],
    src_envmap_left_edge: Union[np.ndarray, torch.Tensor, List[float]] = [0, 0, -1],
    tgt_envmap_zenith: Union[np.ndarray, torch.Tensor, List[float]] = None,
    tgt_envmap_left_edge: Union[np.ndarray, torch.Tensor, List[float]] = None,
    out_shape: Tuple = None,
):
    """
    envmap: [BS, C, H, W]
    coordinates: [BS, 3]
    rmat: rotation matrix [BS, 3, 3] from vector on src to vec on tgt (x_t = R * x_s)
    """
    spec_src_coord = src_envmap_zenith is not None and src_envmap_left_edge is not None
    spec_tgt_coord = tgt_envmap_zenith is not None and tgt_envmap_left_edge is not None

    height, width = envmap.shape[-2:]
    tgt_height, tgt_width = (height, width) if out_shape is None else out_shape

    device = envmap.device
    dtype = envmap.dtype
    src_envmap_zenith = convert_array_to_torch(src_envmap_zenith, device=device, dtype=dtype)[..., None, None, :]
    src_envmap_left_edge = convert_array_to_torch(src_envmap_left_edge, device=device, dtype=dtype)[..., None, None, :]
    tgt_envmap_zenith = convert_array_to_torch(tgt_envmap_zenith, device=device, dtype=dtype)[..., None, None, :]
    tgt_envmap_left_edge = convert_array_to_torch(tgt_envmap_left_edge, device=device, dtype=dtype)[..., None, None, :]

    h_shift, w_shift = torch.pi / tgt_height / 2, torch.pi / tgt_width
    theta = torch.linspace(h_shift, torch.pi - h_shift, tgt_height, device=device)
    phi = torch.linspace(w_shift, torch.pi * 2 - w_shift, tgt_width, device=device)
    thetaphi_map = torch.stack(torch.meshgrid(theta, phi, indexing="ij"), axis=-1)
    xyz_map = thetaphi2xyz(thetaphi_map, normal=tgt_envmap_zenith, tangent=tgt_envmap_left_edge, reverse_phi=True)  # [BS, H, W, 3]
    thetaphi_map = xyz2thetaphi(xyz_map, normal=src_envmap_zenith, tangent=src_envmap_left_edge, reverse_phi=True)  # [BS, H, W, 2]

    # [theta, phi]: [[0, pi], [-pi, pi]] -> [[-1, 1], [-1, 1]]
    thetaphi_map[..., 0] /= torch.pi / 2
    thetaphi_map[..., 0] += -1
    thetaphi_map[..., 1] /= torch.pi
    thetaphi_map[..., 1] %= 2
    thetaphi_map[..., 1] -= 1
    envmap = torch.nn.functional.grid_sample(
        envmap,
        thetaphi_map.flip(-1),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return envmap
