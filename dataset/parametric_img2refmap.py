import math
import random
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch

from models.obsnet import ObsNetDiffusion
from utils.file_io import load_exr
from utils.transform import thetaphi2xyz

from .basedataset import BaseDataset


class ParametricImg2RefmapDataset(BaseDataset):
    def __init__(
        self,
        size: int,
        split: str,
        data_root: str,
        shape_root: str,
        zdim: int,
        transform_func: str = "log",
        clamp_before_exp: float = 0,
        return_envmap: bool = False,
        return_obj: bool = False,
        refmap_key: str = "LrK",
        epoch_bias: int = 0,
        epoch_cycle: int = 1000,
        return_cache=True,
        refmap_cache_root: Optional[str] = None,
        objimg_cache_root: str = None,
        preload_envmap: bool = False,
    ):
        super().__init__(size, transform_func=transform_func, clamp_before_exp=clamp_before_exp)

        assert split in ["train", "val", "test"]
        self.split = split
        self.root = Path(data_root)
        self.data_name = self.root.name
        assert self.data_name in ["LavalIndoor+PolyHaven_2k"]
        self.t = "train" if split in ["train", "val"] else "test"
        with open(f"data/datalists/{self.data_name}/envs_{split}.txt", "r") as f:
            self.envs = f.read().splitlines()

        if shape_root is not None:
            self.shape_root = Path(shape_root)
            self.shape_set_name = self.shape_root.name
            with open(f"data/datalists/{self.shape_set_name}/shapes_{split}.txt", "r") as f:
                self.shape_tags = f.read().rstrip().splitlines()
            self.shape_len = len(self.shape_tags)

        self.zdim = zdim

        self.return_envmap = return_envmap
        self.return_obj = return_obj

        self.generator = torch.Generator()
        self.current_epoch = 0

        self.model: ObsNetDiffusion = None
        self.refmap_key = refmap_key

        self.epoch_bias = epoch_bias
        self.epoch_cycle = epoch_cycle

        self.return_cache = return_cache
        self.refmap_cache_root = Path(refmap_cache_root) if refmap_cache_root is not None else None
        self.objimg_cache_root = Path(objimg_cache_root) if objimg_cache_root is not None else None
        if self.return_cache:
            assert self.refmap_cache_root is not None, "specify refmap_cache_root to return cache"

        self.preload_envmap = preload_envmap
        if self.return_envmap and preload_envmap:
            self.envmaps = {}
            for env in self.envs:
                env_name = env[:-4]
                self.envmaps[env_name] = load_exr(self.root / f"{env_name}.exr", as_torch=True)

    def __len__(self):
        return len(self.envs)

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def set_generator(self, idx: int, epoch: int = None):
        if self.split == "train":
            epoch = epoch or self.current_epoch
            epoch = epoch + self.epoch_bias
            if epoch >= self.epoch_cycle:
                epoch = epoch % self.epoch_cycle
            self.generator.manual_seed((epoch) * len(self) + idx)
        elif self.split == "val":
            self.generator.manual_seed(idx)
            self.generator.manual_seed(torch.empty((), dtype=torch.int64).random_(generator=self.generator).item())
        elif self.split == "test":
            self.generator.manual_seed(idx)
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
            self.generator.manual_seed(torch.empty((), dtype=torch.int64).random_(generator=self.generator).item())
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def __getitem__(self, idx: int):
        env_name = self.envs[idx][:-4]
        self.set_generator(idx)
        zK = torch.rand((self.zdim,), generator=self.generator)
        data = {}
        data["zK"] = zK
        data["envmap_name"] = env_name

        normalized_k = torch.rand((), generator=self.generator)
        data["normalized_k"] = normalized_k

        phi = (torch.rand((), generator=self.generator) * 64).int() / 64 * torch.pi * 2 - torch.pi
        theta = (torch.rand((), generator=self.generator) * 0 + 0.5) * torch.pi  # Invalid
        view_from = thetaphi2xyz(torch.stack([theta, phi]), normal=[0, 1, 0], tangent=[0, 0, 1])
        data["view_from"] = view_from

        mask_idx = torch.rand((), generator=self.generator).item()  # to keep consistency of sampling

        if self.split == "train":
            bias = len(self) * self.current_epoch
        else:
            bias = 0
        shape_idx = (idx + bias) % self.shape_len
        obj_name = self.shape_tags[shape_idx]
        data["obj_name"] = obj_name

        seem_need_envmap = True
        seem_need_obj = True
        if self.model is not None:
            brdf_param_names = self.model.refmap_renderer.brdf_param_names
            size = self.model.refmap_renderer.refmap_res
            denoise_suffix = f"_{self.model.refmap_renderer.denoise}denoise" if self.model.refmap_renderer.denoise else ""
            cache_dir = self.refmap_cache_root / "-".join(brdf_param_names) / f"{size}x{size}_spp{spp}{denoise_suffix}/"
            torch.set_printoptions(precision=4, sci_mode=True)

            def get_cache(z) -> torch.Tensor:
                pieces_key = "b" + str(z)[7:-1] + "v" + str(view_from)[7:-1]
                pieces_key = pieces_key.replace("\n", "").replace(" ", "")
                filename = pieces_key + ".pt"
                cache_file_path = cache_dir / env_name / filename
                if not cache_file_path.exists():
                    return False, torch.full((3, size, size), torch.nan)
                for _ in range(3):
                    try:
                        cache: dict = torch.load(cache_file_path, map_location="cpu")
                    except Exception as e:
                        print(cache_file_path)
                        print(e)
                        sleep(0.01)
                    else:
                        break
                else:
                    return False, torch.full((3, size, size), torch.nan)
                if (
                    cache.get("envmap_name") == env_name
                    and cache.get("brdf_param_names") == brdf_param_names
                    and torch.allclose(cache.get("zk"), z)
                    and torch.allclose(cache.get("view_from"), view_from)
                    and cache.get("refmap_res") == size
                ):
                    if (cache.get("zk") == z).all():
                        return True, cache.get("rendering_results"), True
                    else:
                        return True, cache.get("rendering_results"), True
                else:
                    return False, torch.full((3, size, size), torch.nan)

            refmap = get_cache(zK)
            data[self.refmap_key] = refmap[1]
            seem_need_envmap = not refmap[0]

            size = self.model.img_renderer.image_size
            spp = self.model.img_renderer.spp
            denoise_suffix = f"_{self.model.img_renderer.denoise}denoise" if self.model.img_renderer.denoise else ""
            cache_dir = self.objimg_cache_root / "-".join(brdf_param_names) / f"{size[0]}x{size[1]}_spp{spp}{denoise_suffix}/"
            torch.set_printoptions(precision=4, sci_mode=True)

            def get_cache(z) -> torch.Tensor:
                pieces_key = "b" + str(z)[7:-1] + "v" + str(view_from)[7:-1]
                pieces_key = pieces_key.replace("\n", "").replace(" ", "")
                filename = pieces_key + ".pt"
                cache_file_path = cache_dir / env_name / obj_name / filename
                if not cache_file_path.exists():
                    return (False,)
                for _ in range(3):
                    try:
                        cache: dict = torch.load(cache_file_path, map_location="cpu")
                    except Exception as e:
                        print(cache_file_path)
                        print(e)
                        sleep(0.01)
                    else:
                        break
                else:
                    return (False,)
                if (
                    cache.get("envmap_name") == env_name
                    and cache.get("brdf_param_names") == brdf_param_names
                    and cache.get("obj_name") == obj_name
                    and torch.allclose(cache.get("zk"), z)
                    and torch.allclose(cache.get("view_from"), view_from)
                    and cache.get("image_size") == size
                ):
                    return (
                        True,
                        cache.get("rendering_results_image"),
                        cache.get("rendering_results_normal"),
                        cache.get("rendering_results_depth"),
                    )
                else:
                    return (False,)

            result, *cache = get_cache(zK)
            if result:
                img, normal, depth = cache
                data["img"] = img
                data["img_normal"] = normal
                data["img_depth"] = depth
            else:
                data["img"] = data["img_normal"] = torch.full((3, *size), torch.nan)
                data["img_depth"] = torch.full((1, *size), torch.nan)
            seem_need_obj = not result

            ##### get raw refmap cache #####
            size = self.model.img_renderer.image_size
            spp = self.model.img_renderer.spp
            denoise_suffix = f"_{self.model.img_renderer.denoise}denoise" if self.model.img_renderer.denoise else ""
            cache_dir = self.objimg_cache_root / "-".join(brdf_param_names) / f"{size[0]}x{size[1]}_spp{spp}{denoise_suffix}_rawrefmap/"
            torch.set_printoptions(precision=4, sci_mode=True)

            def get_cache(z) -> torch.Tensor:
                pieces_key = "b" + str(z)[7:-1] + "v" + str(view_from)[7:-1]
                pieces_key = pieces_key.replace("\n", "").replace(" ", "")
                filename = pieces_key + ".pt"
                cache_file_path = cache_dir / env_name / obj_name / filename
                if not cache_file_path.exists():
                    return (False,)
                for _ in range(3):
                    try:
                        cache: dict = torch.load(cache_file_path, map_location="cpu")
                    except Exception as e:
                        print(cache_file_path)
                        print(e)
                        sleep(0.01)
                    else:
                        break
                else:
                    return (False,)
                if (
                    cache.get("envmap_name") == env_name
                    and cache.get("brdf_param_names") == brdf_param_names
                    and cache.get("obj_name") == obj_name
                    and torch.allclose(cache.get("zk"), z)
                    and torch.allclose(cache.get("view_from"), view_from)
                    and cache.get("image_size") == size
                ):
                    return (
                        True,
                        cache.get("raw_refmap"),
                        cache.get("raw_refmask"),
                    )
                else:
                    return (False,)

            result, *cache = get_cache(zK)
            if result:
                data["raw_refmap"], data["raw_refmask"] = cache
            else:
                refmap_size = self.model.refmap_renderer.image_size
                data["raw_refmap"] = torch.full((3, *refmap_size), torch.nan)
                data["raw_refmask"] = torch.full((*refmap_size,), torch.nan)

        if self.return_envmap and seem_need_envmap:
            if self.preload_envmap:
                envmap = self.envmaps[env_name]
            else:
                envmap = load_exr(self.root / f"{env_name}.exr", as_torch=True)
            data["envmap"] = envmap
        elif self.return_envmap:
            # skip loading environment map.
            try:
                envmap_size = self.model.renderer.envmap_size
            except Exception:
                envmap_size = (1000, 2000)
            data["envmap"] = torch.full((*envmap_size, 3), torch.nan, dtype=torch.float)

        if self.return_obj and seem_need_obj:
            data["obj_shape"] = torch.load(self.shape_root / f"{obj_name}.pt")

        data["tag"] = env_name

        return data
