from pathlib import Path
from time import sleep
from typing import Dict, List, Optional, Union

import cv2
import torch

from models.drmnet import DRMNet
from models.obsnet import ObsNetDiffuion
from utils.file_io import load_exr
from utils.transform import thetaphi2xyz

from .basedataset import BaseDataset


class ParametricRefmapDataset(BaseDataset):
    def __init__(
        self,
        size: int,
        split: str,
        data_root: str,
        zdim: int,
        transform_func: str = "log",
        clamp_before_exp: float = 0,
        return_envmap: bool = False,
        mask_root: str = None,
        mask_area_min_rate: float = 0.002,
        epoch_bias: int = 0,
        epoch_cycle: int = 1000,
        preload_envmap: bool = False,
        return_cache=True,
        refmap_cache_root: Optional[str] = None,
    ):
        super().__init__(size, transform_func=transform_func, clamp_before_exp=clamp_before_exp)

        assert split in ["train", "val", "test"]
        self.split = split
        self.root = Path(data_root)
        self.data_name = self.root.name
        assert self.data_name in ["LavalIndoor2kxMERL", "HDRIHAVEN_4k", "LavalIndoor+PolyHaven_2k"]
        self.t = "train" if split in ["train", "val"] else "test"
        with open(f"data/datalists/{self.data_name}/envs_{split}.txt", "r") as f:
            self.envs = f.read().splitlines()

        if mask_root is not None:
            self.with_mask = True
            self.mask_root = Path(mask_root)
            self.mask_name = self.mask_root.name
            with open(f"data/datalists/{self.mask_name}/sparsemaskannotations_{split}.txt", "r") as f:
                self.mask_annotations = f.read().splitlines()
            self.mask_len = len(self.mask_annotations)
            self.mask_area_min_rate = mask_area_min_rate
        else:
            self.with_mask = False

        self.zdim = zdim
        self.return_envmap = return_envmap

        self.generator = torch.Generator()
        self.current_epoch = 0

        self.model: Union[DRMNet, ObsNetDiffuion] = None

        self.return_cache = return_cache
        self.epoch_bias = epoch_bias
        self.epoch_cycle = epoch_cycle
        self.preload_envmap = preload_envmap
        if self.return_envmap and preload_envmap:
            self.envmaps = {}
            for env in self.envs:
                env_name = env[:-4]
                self.envmaps[env_name] = load_exr(self.root / f"{env_name}.exr", as_torch=True)

        self.refmap_cache_root = Path(refmap_cache_root) if refmap_cache_root is not None else None
        if self.return_cache:
            assert self.refmap_cache_root is not None, "specify refmap_cache_root to return cache"

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

        mask_idx = torch.rand((), generator=self.generator).item()
        if self.with_mask:
            mask_idx = int(mask_idx * self.mask_len)
            while True:
                mask = cv2.imread(str(self.mask_root / self.t / self.mask_annotations[mask_idx]), -1)
                height, width = mask.shape[:2]
                # don't use the masks with too small region
                if mask.astype(bool).sum() >= height * width * self.mask_area_min_rate:
                    break
                else:
                    mask_idx = (mask_idx + 1) % self.mask_len
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            data["mask"] = mask / 255

        # load rendered cata cache
        if self.model is not None and self.return_cache:
            brdf_param_names = self.model.renderer.brdf_param_names or self.model.brdf_param_names
            size = self.model.renderer.refmap_res
            spp = self.model.renderer.spp
            denoise_suffix = f"_{self.model.renderer.denoise}denoise" if self.model.renderer.denoise else ""
            pieces_cache_dir = self.refmap_cache_root / f'{"-".join(brdf_param_names)}/{size}x{size}_spp{spp}{denoise_suffix}/'
            torch.set_printoptions(precision=4, sci_mode=True)

            def get_cache(z) -> torch.Tensor:
                pieces_key = "b" + str(z)[7:-1] + "v" + str(view_from)[7:-1]
                pieces_key = pieces_key.replace("\n", "").replace(" ", "")
                filename = pieces_key + ".pt"
                cache_file_path = pieces_cache_dir / env_name / filename
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

            rK = get_cache(zK)
            data["LrK"] = rK[1]

            if (z0 := getattr(self.model, "_z0", None)) is not None:  # to train DRMNet
                K, k, zk, zkm1 = self.model.get_schedule(zK, z0=z0, normalized_k=normalized_k, return_zkm1=True)
                # zk and zkm1 is the shape of [batch, zdim]
                data["K"] = K
                data["k"] = k
                rk = get_cache(zk)
                data["zk"] = zk
                data["Lrk"] = rk[1]
                if K > 0:
                    rkm1 = get_cache(zkm1)
                    data["zkm1"] = zkm1
                    data["Lrkm1"] = rkm1[1]
                else:
                    data["zkm1"] = torch.full_like(zkm1, torch.nan)
                    data["Lrkm1"] = torch.full_like(rk[1], torch.nan)
                r0 = get_cache(z0)
                data["r0"] = r0[1]
                seem_need_envmap = not (rK[0] and rk[0] and (rkm1[0] or r0[0]))
            else:  # to train ObsNet
                seem_need_envmap = not rK[0]
        else:
            seem_need_envmap = True

        if self.return_envmap and seem_need_envmap:
            if self.preload_envmap:
                envmap = self.envmaps[env_name]
            else:
                envmap = load_exr(self.root / f"{env_name}.exr", as_torch=True)
            data["envmap"] = envmap
        elif self.return_envmap:
            if self.preload_envmap:
                envmap = self.envmaps[env_name]
            else:
                # skip loading environment map.
                try:
                    envmap_size = self.model.renderer.envmap_size
                except Exception:
                    envmap_size = (1000, 2000)
                envmap = torch.full((*envmap_size, 3), torch.nan, dtype=torch.float)
            data["envmap"] = envmap

        data["tag"] = env_name

        return data
