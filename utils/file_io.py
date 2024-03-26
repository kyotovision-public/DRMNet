import struct
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch


def load_exr(path: Path, as_torch: bool = False, channel_first: bool = False) -> Union[np.ndarray, torch.Tensor]:
    # not support alpha channel
    img: np.ndarray = cv2.cvtColor(cv2.imread(str(path), -1)[..., :3], cv2.COLOR_BGR2RGB)
    if channel_first:
        img = img.transpose(2, 0, 1)
    if as_torch:
        img: torch.Tensor = torch.from_numpy(img)
    return img


def save_exr(path: Path, img: Union[np.ndarray, torch.Tensor], channel_first: bool = False):
    if isinstance(img, torch.Tensor):
        img: np.ndarray = img.detach().cpu().numpy()
    if channel_first:
        img = img.transpose(1, 2, 0)
    return cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def save_png(path: Path, ldr: Union[np.ndarray, torch.Tensor], channel_first: bool = False, mask: Union[np.ndarray, torch.Tensor] = None):
    # mask: [H, W]
    if isinstance(ldr, torch.Tensor):
        ldr: np.ndarray = ldr.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask: np.ndarray = mask.detach().cpu().numpy()
    if channel_first:
        ldr = ldr.transpose(1, 2, 0)
    ldr = ldr[:, :, :3]
    ldr = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB)
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[:, :, None]
        ldr = np.concatenate([ldr, mask], axis=-1)
    return cv2.imwrite(str(path), ldr * 255)


def load_png(path: Path, as_torch: bool = False, channel_first: bool = False):
    # not support alpha channel
    img = cv2.imread(str(path), -1)
    if img.ndim == 3 and img.shape[-1] == 3:
        img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if channel_first:
        if img.ndim == 2:
            img = img[:, :, None]
        img = img.transpose(2, 0, 1)
    if as_torch:
        img: torch.Tensor = torch.from_numpy(img)
    return img / 255.0


BRDF_SAMPLING_RES_THETA_H = 90
BRDF_SAMPLING_RES_THETA_D = 90
BRDF_SAMPLING_RES_PHI_D = 360
RED_SCALE = 1.0 / 1500.0
GREEN_SCALE = 1.15 / 1500.0
BLUE_SCALE = 1.66 / 1500.0


def save_merl(data: np.ndarray, filename: Path):
    """save a merl format brdf

    :param np.ndarray data: an array of brdf with shape of (3, ThetaH, ThetaD, PhiD)
    :param Path filename: the path to save (.binary)
    """
    data = np.reshape(data, [3, -1])
    data = data / np.array([RED_SCALE, GREEN_SCALE, BLUE_SCALE]).reshape(3, 1)
    data = data.flatten()
    with open(filename, "wb") as f:
        f.write(struct.pack("iii", BRDF_SAMPLING_RES_THETA_H, BRDF_SAMPLING_RES_THETA_D, BRDF_SAMPLING_RES_PHI_D // 2))
        for i in range(data.shape[0]):
            f.write(struct.pack("d", data[i]))


def load_merl(filename: Path) -> np.ndarray:
    """load a merl format brdf

    :param Path filename: the path to load (.binary)
    :return np.ndarray: an array of brdf with shape of (3, ThetaH, ThetaD, PhiD)
    """
    N_DIM = BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D // 2
    with open(filename, "rb") as f:
        dim = struct.unpack("iii", f.read(4 * 3))
        n = dim[0] * dim[1] * dim[2]
        if n != N_DIM:
            raise ValueError("invalid BRDF file")

        data = np.empty(3 * n, dtype=np.float32)
        for i in range(3 * n):
            data[i] = struct.unpack("d", f.read(8))[0]

        # color x theta_h x theta_d x phi_d
        data = data.reshape(3, BRDF_SAMPLING_RES_THETA_H, BRDF_SAMPLING_RES_THETA_D, BRDF_SAMPLING_RES_PHI_D // 2)
        data *= np.reshape(np.array([RED_SCALE, GREEN_SCALE, BLUE_SCALE]), [3, 1, 1, 1])

        return data
