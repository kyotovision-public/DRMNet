"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import mitsuba as mi
import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf
from torchvision.utils import make_grid
from tqdm import tqdm

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import default, instantiate_from_config
from utils.file_io import load_exr
from utils.img2refmap import refmap_mask_make

if mi.variant() is not None:
    from utils.mitsuba3_utils import MitsubaOrthoRenderer, MitsubaRefMapRenderer


class ObsNetDiffusion(LatentDiffusion):
    """inpainting class"""

    def __init__(
        self,
        renderer_config: OmegaConf = None,
        img_renderer_config: OmegaConf = None,
        num_timesteps_cond=None,
        cond_stage_key="image",
        padding_mode="noise",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        ddim_steps: Optional[int] = None,
        ddim_eta: float = 1.0,
        masked_loss: bool = True,
        noisy_observe: float = 0.0,
        obj_img_key: bool = None,
        init_from_ckpt_verbose=True,
        cache_data: bool = False,
        refmap_cache_root: str = None,
        objimg_cache_root: str = None,
        envmap_dir: str = None,
        first_stage_config: OmegaConf = OmegaConf.create({"target": "ldm.models.autoencoder.IdentityFirstStage"}),
        cond_stage_config: OmegaConf = "__is_first_stage__",
        *args,
        **kwargs,
    ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(
            first_stage_config,
            cond_stage_config,
            num_timesteps_cond=num_timesteps_cond,
            cond_stage_key=cond_stage_key,
            cond_stage_trainable=cond_stage_trainable,
            concat_mode=concat_mode,
            cond_stage_forward=cond_stage_forward,
            conditioning_key=conditioning_key,
            scale_factor=scale_factor,
            scale_by_std=scale_by_std,
            *args,
            **kwargs,
        )
        if renderer_config is not None:
            self.renderer: MitsubaRefMapRenderer = instantiate_from_config(renderer_config)
        if img_renderer_config is not None:
            assert renderer_config is not None
            self.refmap_renderer = self.renderer
            self.img_renderer: MitsubaOrthoRenderer = instantiate_from_config(img_renderer_config)
        self.padding_mode = padding_mode

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, verbose=init_from_ckpt_verbose)
            self.restarted_from_ckpt = True

        self.masked_loss = masked_loss
        self.noisy_observe = noisy_observe
        self.obj_img_key = obj_img_key

        # for prediction
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta

        # for caching data
        self.cache_data = cache_data
        self.refmap_cache_root = Path(refmap_cache_root) if refmap_cache_root is not None else None
        self.objimg_cache_root = Path(objimg_cache_root) if objimg_cache_root is not None else None
        self.envmap_dir = Path(envmap_dir) if envmap_dir is not None else None

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False, verbose=True):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0 and verbose:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0 and verbose:
            print(f"Unexpected Keys: {unexpected}")

    def on_train_epoch_start(self) -> None:
        self.ds = getattr(self.trainer.datamodule, f"train_ds", None)

    def on_validation_epoch_start(self) -> None:
        self.ds = getattr(self.trainer.datamodule, f"val_ds", None)

    def on_test_epoch_start(self) -> None:
        self.ds = getattr(self.trainer.datamodule, f"test_ds", None)

    def on_predict_epoch_start(self) -> None:
        self.ds = getattr(self.trainer.datamodule, f"predict_ds", None)

    @torch.no_grad()
    def get_input(
        self, batch, k, return_first_stage_outputs=False, force_c_encode=False, cond_key=None, return_original_cond=False, bs=None
    ):
        zK = batch["zK"][:bs]
        self.batch_size = len(zK)
        envmap_name: List[str] = batch["envmap_name"][:bs]

        view_from: torch.Tensor = batch.get("view_from")[:bs]

        if k in batch:
            LrK = batch[k][:bs]
        else:
            LrK = torch.empty(len(zK), 3, self.renderer.image_size, dtype=torch.float, device=self.device)
            LrK[:, 0, 0, 0] = torch.nan
        not_cached = torch.isnan(LrK[:, 0, 0, 0])  # [batch]

        # load envmap for rendering
        if not_cached.any():
            assert self.envmap_dir is not None, "envmap_dir is needed, but not set"
            if "envmap" in batch:
                envmap = batch["envmap"][:bs]
            else:
                envmap = torch.empty(len(not_cached), *self.renderer.envmap_size, 3, dtype=torch.float, device=self.device)
                envmap[:, 0, 0, 0] = torch.nan
            need_to_load_envmap_idxs = torch.nonzero(torch.logical_and(not_cached.any(dim=0), torch.isnan(envmap[:, 0, 0, 0])))
            need_to_load_envmap_names = [envmap_name[i] for i in need_to_load_envmap_idxs]
            if len(need_to_load_envmap_idxs) > 0:
                with ThreadPoolExecutor(max_workers=20) as executer:
                    envmaps = executer.map(lambda name: load_exr(self.envmap_dir / f"{name}.exr", as_torch=True), need_to_load_envmap_names)
                    for idx, em in zip(need_to_load_envmap_idxs, envmaps):
                        envmap[idx] = em.cuda()
        else:
            envmap = None

        # rendering refmap which is not cached
        for (batch_idx,) in torch.nonzero(not_cached):
            LrK[batch_idx] = self.renderer.rendering(
                zK[batch_idx],
                None,
                envmap=envmap[batch_idx],
                view_from=view_from[batch_idx],
                channel_first=True,
            )

        if not_cached.any() and self.cache_data:
            assert self.refmap_cache_root is not None, "cache_data is True, but refmap_cache_root is not specified"
            size = self.renderer.refmap_res
            spp = self.renderer.spp
            brdf_param_names = self.renderer.brdf_param_names
            denoise_suffix = f"_{self.renderer.denoise}denoise" if self.renderer.denoise else ""
            refmap_cache_dir = self.refmap_cache_root / "-".join(brdf_param_names) / f"{size}x{size}_spp{spp}{denoise_suffix}/"
            refmap_cache_dir.mkdir(exist_ok=True, parents=True)
            zK_cpu = zK.cpu()
            view_from_cpu = view_from.cpu()
            torch.set_printoptions(precision=4, sci_mode=True)

            def thread_func(batch_idx: int) -> None:
                pieces_key = "b" + str(zK_cpu[batch_idx])[7:-1] + "v" + str(view_from_cpu[batch_idx])[7:-1]
                pieces_key = pieces_key.replace("\n", "").replace(" ", "")
                filename = pieces_key + ".pt"
                piece_cache_file_path = refmap_cache_dir / envmap_name[batch_idx] / filename
                if piece_cache_file_path.exists():
                    return
                piece_cache_file_path.parent.mkdir(exist_ok=True)
                data = {
                    "key": pieces_key,
                    "envmap_name": envmap_name[batch_idx],
                    "brdf_param_names": brdf_param_names,
                    "zk": zK_cpu[batch_idx].clone(),
                    "view_from": view_from_cpu[batch_idx].clone(),
                    "filename": filename,
                    "rendering_results": LrK[batch_idx].cpu(),
                    "envmap_size": self.renderer.envmap_size,
                    "refmap_res": size,
                    "spp": spp,
                }
                torch.save(data, piece_cache_file_path)

            with ThreadPoolExecutor(max_workers=20) as executer:
                for (batch_idx,) in torch.nonzero(not_cached):
                    executer.submit(thread_func, batch_idx)

        cond_key = cond_key if cond_key is not None else self.cond_stage_key
        if cond_key == "masked_LrK":
            mask = batch["mask"][:bs, None].float()  # [BS, 1, H, W]
            LrK = self.ds.transform(LrK, dynamic_normalize=True, mask=mask)
        elif cond_key == " raw_refmap":
            raw_refmap: torch.Tensor = batch["raw_refmap"][:bs]
            raw_refmask: torch.Tensor = batch["raw_refmask"][:bs]
            not_cached_raw_refmap = torch.isnan(raw_refmap[:, 0, 0, 0])  # [batch]
            if not_cached_raw_refmap.any():
                obj_img: torch.Tensor = batch[self.obj_img_key][:bs]
                obj_normal: torch.Tensor = batch["img_normal"][:bs]
                obj_depth: torch.Tensor = batch["img_depth"][:bs]
                not_cached = torch.isnan(obj_img[:, 0, 0, 0])  # [batch]
                size = self.img_renderer.image_size
                spp = self.img_renderer.spp
                brdf_param_names = self.img_renderer.brdf_param_names
                denoise_suffix = f"_{self.img_renderer.denoise}denoise" if self.img_renderer.denoise else ""
                envmap_name: List[str] = batch["envmap_name"][:bs]
                obj_name: List[str] = batch["obj_name"][:bs]
                zK: torch.Tensor = batch["zK"][:bs]
                zK_cpu = zK.cpu()
                view_from: torch.Tensor = batch["view_from"][:bs]
                view_from_cpu = view_from.cpu()

                # load envmap for rendering
                if not_cached.any():
                    assert self.envmap_dir is not None, "envmap_dir is needed, but not set"
                    if envmap is None and "envmap" not in batch:
                        envmap = torch.empty(len(not_cached), *self.renderer.envmap_size, 3, dtype=torch.float, device=self.device)
                        envmap[:, 0, 0, 0] = torch.nan
                    need_to_load_envmap_idxs = torch.nonzero(torch.logical_and(not_cached.any(dim=0), torch.isnan(envmap[:, 0, 0, 0])))
                    need_to_load_envmap_names = [envmap_name[i] for i in need_to_load_envmap_idxs]
                    if len(need_to_load_envmap_idxs) > 0:
                        with ThreadPoolExecutor(max_workers=20) as executer:
                            envmaps = executer.map(
                                lambda name: load_exr(self.envmap_dir / f"{name}.exr", as_torch=True), need_to_load_envmap_names
                            )
                            for idx, em in zip(need_to_load_envmap_idxs, envmaps):
                                envmap[idx] = em.cuda()
                else:
                    envmap = None

                # rendering object images which is not cached
                if not_cached.any():
                    obj_dict: List[Dict[str, torch.Tensor]] = batch["obj_shape"][:bs]
                    for (batch_idx,) in torch.nonzero(not_cached):
                        obj_img[batch_idx], obj_normal[batch_idx], obj_depth[batch_idx] = self.img_renderer.rendering(
                            zK[batch_idx],
                            brdf_param_names,
                            envmap=envmap[batch_idx],
                            view_from=view_from[batch_idx],
                            obj=obj_dict[batch_idx],
                            channel_first=True,
                        )

                    # cache object images which is not cached
                    if self.cache_data:
                        assert self.objimg_cache_root is not None, "cache_data is True, but objimg_cache_root is not specified"
                        obj_dict: List[Dict[str, torch.Tensor]] = batch["obj_shape"][:bs]

                        objimg_cache_dir = (
                            self.objimg_cache_root / "-".join(brdf_param_names) / f"{size[0]}x{size[1]}_spp{spp}{denoise_suffix}/"
                        )
                        objimg_cache_dir.mkdir(exist_ok=True, parents=True)

                        torch.set_printoptions(precision=4, sci_mode=True)

                        def thread_func(batch_idx: int) -> None:
                            pieces_key = "b" + str(zK_cpu[batch_idx])[7:-1] + "v" + str(view_from_cpu[batch_idx])[7:-1]
                            pieces_key = pieces_key.replace("\n", "").replace(" ", "")
                            filename = pieces_key + ".pt"
                            piece_cache_file_path = objimg_cache_dir / envmap_name[batch_idx] / obj_name[batch_idx] / filename
                            if piece_cache_file_path.exists():
                                return
                            piece_cache_file_path.parent.mkdir(parents=True, exist_ok=True)
                            data = {
                                "key": pieces_key,
                                "envmap_name": envmap_name[batch_idx],
                                "obj_name": obj_name[batch_idx],
                                "brdf_param_names": brdf_param_names,
                                "zk": zK_cpu[batch_idx].clone(),
                                "view_from": view_from_cpu[batch_idx].clone(),
                                "filename": filename,
                                "rendering_results_image": obj_img[batch_idx].cpu(),
                                "rendering_results_normal": obj_normal[batch_idx].cpu(),
                                "rendering_results_depth": obj_depth[batch_idx].cpu(),
                                "envmap_size": self.img_renderer.envmap_size,
                                "image_size": size,
                                "spp": spp,
                            }
                            torch.save(data, piece_cache_file_path)

                        with ThreadPoolExecutor(max_workers=20) as executer:
                            for (batch_idx,) in torch.nonzero(not_cached):
                                executer.submit(thread_func, batch_idx)

                # make raw refmaps which is not cached
                for (batch_idx,) in torch.nonzero(not_cached_raw_refmap):
                    img = obj_img[batch_idx]
                    normal = obj_normal[batch_idx]
                    obj_mask = torch.linalg.norm(normal, dim=0) > 0.5
                    rawmap, rawmask = refmap_mask_make(
                        img[:, obj_mask].permute(1, 0),
                        normal[:, obj_mask].permute(1, 0),
                        self.renderer.refmap_res,
                        angle_threshold=np.pi / 128 / 2,
                    )
                    raw_refmap[batch_idx], raw_refmask[batch_idx] = rawmap.permute(2, 0, 1), rawmask

                # cache raw refmaps
                if not_cached.any() and self.cache_data:
                    assert self.objimg_cache_root is not None, "cache_data is True, but objimg_cache_root is not specified"
                    rawrefmap_cache_dir = (
                        self.objimg_cache_root / "-".join(brdf_param_names) / f"{size[0]}x{size[1]}_spp{spp}{denoise_suffix}_rawrefmap/"
                    )
                    rawrefmap_cache_dir.mkdir(exist_ok=True, parents=True)
                    torch.set_printoptions(precision=4, sci_mode=True)

                    def thread_func(batch_idx: int) -> None:
                        pieces_key = "b" + str(zK_cpu[batch_idx])[7:-1] + "v" + str(view_from_cpu[batch_idx])[7:-1]
                        pieces_key = pieces_key.replace("\n", "").replace(" ", "")
                        filename = pieces_key + ".pt"
                        piece_cache_file_path = rawrefmap_cache_dir / envmap_name[batch_idx] / obj_name[batch_idx] / filename
                        if piece_cache_file_path.exists():
                            return
                        piece_cache_file_path.parent.mkdir(parents=True, exist_ok=True)
                        if torch.isnan(raw_refmap[batch_idx][0, 0, 0]):
                            breakpoint()
                        data = {
                            "key": pieces_key,
                            "envmap_name": envmap_name[batch_idx],
                            "obj_name": obj_name[batch_idx],
                            "brdf_param_names": brdf_param_names,
                            "zk": zK_cpu[batch_idx].clone(),
                            "view_from": view_from_cpu[batch_idx].clone(),
                            "filename": filename,
                            "raw_refmap": raw_refmap[batch_idx].cpu(),
                            "raw_refmask": raw_refmask[batch_idx].cpu(),
                            "image_size": size,
                            "spp": spp,
                        }
                        torch.save(data, piece_cache_file_path)

                    with ThreadPoolExecutor(max_workers=20) as executer:
                        for (batch_idx,) in torch.nonzero(not_cached_raw_refmap):
                            executer.submit(thread_func, batch_idx)

            mask = raw_refmask[:, None]
            # normalize based on raw refmap.
            raw_refmap = self.ds.transform(raw_refmap, dynamic_normalize=True, mask=mask)
            LrK = self.ds.transform(LrK, dynamic_normalize=False, mask=mask)
        else:
            raise NotImplementedError()

        LrK_z = self.get_first_stage_encoding(self.encode_first_stage(LrK)).detach()

        if self.model.conditioning_key is not None:
            if cond_key == "masked_LrK":
                cond = mask * LrK
            elif cond_key == " raw_refmap":
                cond = mask * raw_refmap
            else:
                raise NotImplementedError()
            if self.noisy_observe > 0:
                cond = self.noisy_observe * torch.randn_like(cond) + cond

            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(cond, dict) or isinstance(cond, list):
                    # import pudb; pudb.set_trace()
                    c = self.get_learned_conditioning(cond)
                else:
                    c = self.get_learned_conditioning(cond.to(self.device))
            else:
                c = cond

            mask = torch.nn.functional.interpolate(mask, size=(self.image_size, self.image_size))
            if self.padding_mode == "noise":
                cond += (1 - mask) * torch.randn_like(cond)
            elif self.padding_mode == "zeros":
                pass
            else:
                raise NotImplementedError()
        else:
            cond = None
            c = None

        out = [LrK_z, c, mask]
        if return_first_stage_outputs:
            LrK_rec = self.decode_first_stage(LrK_z)
            out.extend([LrK, LrK_rec])
        if return_original_cond:
            out.append(cond)
        return out

    def shared_step(self, batch, **kwargs):
        x, c, mask = self.get_input(batch, self.first_stage_key)
        loss = self(x, c, mask)
        return loss

    def forward(self, x, c, mask, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, mask, t, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        self.log(
            "global_step",
            self.global_step,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr_abs", lr, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True, batch_size=self.batch_size)

        return loss

    def p_losses(self, x_start, cond, mask, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # x_noisy = (1 - mask) * x_noisy + mask * x_start
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        if self.masked_loss:
            invmask = 1 - mask
            loss_simple = (self.get_loss(model_output, target, mean=False) * invmask).sum(dim=(1, 2, 3)) / (
                invmask.sum(dim=(1, 2, 3)) * model_output.size(1)
            )
        else:
            loss_simple = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        if self.masked_loss:
            loss_vlb = (self.get_loss(model_output, target, mean=False) * invmask).sum(dim=(1, 2, 3)) / (
                invmask.sum(dim=(1, 2, 3)) * model_output.size(1)
            )
        else:
            loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    @torch.no_grad()
    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        start_T=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = {"x_inter": [img], "pred_x0": []}

        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc="Sampling t", total=timesteps) if verbose else reversed(range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            if mask is not None:
                img_orig = x0.clone() if i == 0 else self.q_sample(x0, ts - 1)
                img = img_orig * mask + (1.0 - mask) * img

            img, pred_x0 = self.p_sample(
                img, cond, ts, clip_denoised=self.clip_denoised, quantize_denoised=quantize_denoised, return_x0=True
            )

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        if return_intermediates:
            return pred_x0, intermediates
        return pred_x0

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                log_every_t=kwargs.pop("log_every_t", None) or max(self.log_every_t * ddim_steps // self.num_timesteps, 1),
                **kwargs,
            )
        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size, return_intermediates=True, **kwargs)

        return samples, intermediates

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=10,
        n_row=10,
        sample=True,
        ddim_steps=None,
        ddim_eta=None,
        return_keys=None,
        plot_denoise_rows=True,
        plot_progressive_rows=True,
        plot_diffusion_rows=True,
        **kwargs,
    ):
        ddim_steps = ddim_steps or self.ddim_steps
        ddim_eta = ddim_eta or self.ddim_eta
        use_ddim = ddim_steps is not None
        log = dict()
        z, c, mask = self.get_input(batch, self.first_stage_key, force_c_encode=True, bs=N)
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["inputs"] = torch.concat([z, mask], dim=1)
        if self.model.conditioning_key is not None:
            log["conditionings"] = c

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.ds.rescale(self.decode_first_stage(z_noisy)))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, "n b c h w -> b n c h w")
            diffusion_grid = rearrange(diffusion_grid, "b n c h w -> (b n) c h w")
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid.permute(1, 2, 0).cpu().numpy()

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, intermediate = self.sample_log(
                    cond=c,
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    log_every_t=max(self.log_every_t * ddim_steps // self.num_timesteps, 1) if use_ddim else self.log_every_t,
                )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(intermediate["x_inter"])
                log["denoise_row"] = denoise_grid.permute(1, 2, 0).cpu().numpy()
            if plot_progressive_rows:
                denoise_grid = self._get_denoise_row_from_list(intermediate["pred_x0"])
                log["progressive_row"] = denoise_grid.permute(1, 2, 0).cpu().numpy()

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    @torch.no_grad()
    def get_cond_for_predict(
        self,
        batch: Dict[str, Union[torch.Tensor, str]],
        bs: Optional[int] = None,
        force_c_encode: bool = False,
    ):
        if self.model.conditioning_key is not None:
            if self.cond_stage_key == "masked_LrK":
                if "masked_LrK" in batch:
                    cond = batch["masked_LrK"][:bs]
                else:
                    LrK = batch[self.first_stage_key][:bs]
                    LrK: torch.Tensor = self.ds.transform(LrK, dynamic_normalize=True, mask=mask)
                    mask = batch["mask"][:bs, None].float()
                    cond = mask * LrK
            elif self.cond_stage_key == "raw_refmap":
                mask = batch["raw_refmask"][:bs, None].float()
                raw_refmap: torch.Tensor = self.ds.transform(batch["raw_refmap"][:bs], dynamic_normalize=True, mask=mask)
                cond = raw_refmap * mask
            else:
                raise NotImplementedError()
            if self.noisy_observe > 0:
                cond = self.noisy_observe * torch.randn_like(cond) + cond

            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(cond, dict) or isinstance(cond, list):
                    c = self.get_learned_conditioning(cond)
                else:
                    c = self.get_learned_conditioning(cond.to(self.device))
            else:
                c = cond

            mask = torch.nn.functional.interpolate(mask, size=(self.image_size, self.image_size))
            if self.padding_mode == "noise":
                cond += (1 - mask) * torch.randn_like(cond)
            elif self.padding_mode == "zeros":
                pass
            else:
                raise NotImplementedError()
        else:
            mask = batch.get("mask")
            if mask is not None:
                mask = mask[:bs, None].float()
            cond = None
            c = None

        tag: List[str] = batch["tag"][:bs]
        return c, mask, tag
