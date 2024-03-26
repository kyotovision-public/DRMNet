from typing import Dict, List, Optional, Union

import torch
import torchvision


class BaseDataset(torch.utils.data.Dataset):
    """base class of dataset"""

    def __init__(
        self,
        size: int,
        transform_func: str = "log",
        clamp_before_exp: float = 0.0,
    ):
        """

        :param int size: the resolution of height/width
        :param str transform_func: specifies transform function, must be written in the order of mathematical notation ( f(g()) -> f_g ), defaults to "log"
        :param float clamp_before_exp: if >0, clamp data with min value before exponential, defaults to 0.0.
        """
        # transform_func must be written in the order of mathematical notation ( f(g()) -> f_g )
        self.size = size
        self.transform_func_str = transform_func
        self.clamp_before_exp = 10 if isinstance(clamp_before_exp, bool) and not clamp_before_exp else clamp_before_exp
        self.transform_funcs = [self.get_tranfrom_func(func_name) for func_name in transform_func.split("_")[::-1]]
        self.rescale_funcs = [self.get_rescale_func(func_name) for func_name in transform_func.split("_")]

    def transform(self, x: torch.Tensor, dynamic_normalize: bool = False, mask: torch.Tensor = None):
        # x: [(batch, channel), height, width]
        assert x.size(-1) >= self.size
        for func in self.transform_funcs:
            x = func(x, dynamic_normalize=dynamic_normalize, mask=mask)
        return x

    def rescale(self, x: torch.Tensor):
        for func in self.rescale_funcs:
            x = func(x)
        return x

    def get_tranfrom_func(self, func_name: str):
        # x: [(batch, channel), height, width]
        assert "_" not in func_name
        if func_name.startswith("resize"):
            if len(func_name) > 6:
                InterpolationMode = getattr(torchvision.transforms.InterpolationMode, func_name[6:].replace("-", "_"))
            else:
                InterpolationMode = torchvision.transforms.InterpolationMode.BILINEAR
            return lambda x, **kwargs: torchvision.transforms.functional.resize(
                x, size=(self.size, self.size), interpolation=InterpolationMode, antialias=True
            )
        elif func_name == "log":
            return lambda x, **kwargs: torch.log10(x + 1e-1) + 1
        elif func_name == "log10":
            return lambda x, **kwargs: torch.log10(x)
        elif func_name.startswith("lowerbound"):
            bottom = float(func_name[10:])
            return lambda x, **kwargs: torch.clip(x, bottom)
        elif func_name == "0p1tom1p1":
            return lambda x, **kwargs: x * 2 - 1
        elif func_name == "normalizedLogarithmic":

            def func(x: torch.Tensor, mask: torch.Tensor, dynamic_normalize: bool, **kwargs):
                if dynamic_normalize:
                    assert mask is not None
                    linearmax = (x * mask).amax(dim=(-1, -2, -3), keepdim=True)
                    log10max = torch.log10(linearmax)
                    log10min = torch.log10((x * mask + (1 - mask.float()) * linearmax).amin(dim=(-1, -2, -3), keepdim=True))
                    self.Logarithmic_params = [log10min, log10max]
                log10min, log10max = self.Logarithmic_params
                assert x.ndim == log10min.ndim == log10max.ndim, f"{x.ndim}, {log10min.ndim}, {log10max.ndim}"
                log10min, log10max = log10min.to(x.device), log10max.to(x.device)
                x = (torch.log10(x) - log10min) / (log10max - log10min)
                return x

            return func
        else:
            raise NotImplementedError(func_name)

    def get_rescale_func(self, func_name: str):
        do_nothing = lambda x, **kwargs: x
        # x: [(batch, channel), height, width]
        assert "_" not in func_name
        if func_name.startswith("resize"):
            return do_nothing
        elif func_name == "log":
            if self.clamp_before_exp:
                return lambda x, **kwargs: torch.pow(10, torch.clamp(x - 1, max=self.clamp_before_exp)) - 1e-1
            else:
                return lambda x, **kwargs: torch.pow(10, x - 1) - 1e-1
        elif func_name == "log10":
            if self.clamp_before_exp:
                return lambda x, **kwargs: torch.pow(10, torch.clamp(x, max=self.clamp_before_exp))
            else:
                return lambda x, **kwargs: torch.pow(10, x)
        elif func_name.startswith("lowerbound"):
            return do_nothing
        elif func_name == "0p1tom1p1":
            return lambda x, **kwargs: (x + 1) / 2
        elif func_name == "normalizedLogarithmic":
            log10 = self.get_rescale_func("log10")

            def func(x: torch.Tensor, **kwargs):
                log10min, log10max = self.Logarithmic_params
                log10min, log10max = log10min.to(x.device), log10max.to(x.device)
                assert x.ndim == log10min.ndim == log10max.ndim, f"{x.ndim=}, {log10min.ndim=}, {log10max.ndim=}"

                return log10(x * (log10max - log10min) + log10min, **kwargs)

            return func
        else:
            raise NotImplementedError(func_name)
