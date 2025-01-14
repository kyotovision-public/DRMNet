import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mitsuba as mi
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid
from tqdm import tqdm

from ldm.models.diffusion.ddpm import DiffusionWrapper, LatentDiffusion, disabled_train
from ldm.modules.diffusionmodules.util import linear
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.ema import LitEma
from ldm.util import count_params, instantiate_from_config
from utils.file_io import load_exr
from utils.transform import mirmap2envmap

if mi.variant() is not None:
    from utils.mitsuba3_utils import MitsubaRefMapRenderer, get_bsdf, visualize_bsdf


class ZEmbDiffusionWrapper(DiffusionWrapper):
    def __init__(self, model_config, conditioning_key, z_dim, emb_z: bool = True, emb_z_crossattn: bool = False):
        super().__init__(model_config, conditioning_key)
        self.emb_z = emb_z
        self.emb_z_crossattn = emb_z_crossattn
        if emb_z:
            model_channels = self.diffusion_model.model_channels
            self.z_emb_layer = nn.Sequential(
                linear(z_dim, model_channels // 2),
                nn.SiLU(),
                linear(model_channels // 2, model_channels // 2),
                nn.SiLU(),
                linear(model_channels // 2, model_channels),
                nn.SiLU(),
            )
        if emb_z_crossattn:
            if self.conditioning_key in ["crossattn", "hybrid", "adm"]:
                raise NotImplementedError()
            self.emb_z_context_layer = nn.Sequential(
                linear(model_channels, self.diffusion_model.context_dim),
                nn.SiLU(),
            )

    def forward(self, x, z_emb, c_concat: list = None, c_crossattn: list = None):
        z_emb = self.z_emb_layer(z_emb) if self.emb_z else None
        context = self.emb_z_context_layer(z_emb)[:, None] if self.emb_z_crossattn else None
        if self.conditioning_key is None or self.conditioning_key == "none":
            out = self.diffusion_model(x, t_emb=z_emb, context=context)
        elif self.conditioning_key == "concat":
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t_emb=z_emb, context=context)
        elif self.conditioning_key == "crossattn":
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t_emb=z_emb, context=cc)
        elif self.conditioning_key == "hybrid":
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t_emb=z_emb, context=cc)
        elif self.conditioning_key == "adm":
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t_emb=z_emb, y=cc)
        else:
            raise NotImplementedError()

        return out


class DRMNet(pl.LightningModule):
    def __init__(
        self,
        illnet_config: OmegaConf,
        refnet_config: OmegaConf,
        renderer_config: OmegaConf,
        max_timesteps: int = 250,
        loss_type: str = "l1",
        ckpt_path: str = None,
        init_from_ckpt_verbose: bool = True,
        ignore_keys: List[str] = [],
        monitor: str = "val/loss",
        use_ema: bool = True,
        input_key: str = "LrK",
        sigma_for_cond_xK: float = 0.0,
        image_size: int = 128,
        channels: int = 3,
        log_every_k: int = 5,
        parameterization: str = "residual",
        scheduler_config: OmegaConf = None,
        cond_stage_trainable: bool = False,
        concat_mode: bool = False,
        cond_stage_forward: Optional[str] = None,
        conditioning_key: Optional[str] = None,
        scale_factor: float = 1.0,
        scale_by_std: bool = False,
        l_refmap_weight: float = 1.0,
        l_refcode_weight: float = 1.0,
        sigma: float = 0.01,
        delta: float = 0.0125,
        gamma: float = 0.9,
        epsilon: float = 0.001,
        train_with_zk_gt: bool = False,
        train_with_zk_gt_switch_epoch: Optional[int] = None,
        brdf_param_names: List[str] = ["specular"],
        z0: List[float] = [1.0],
        model_emb_z: bool = True,
        emb_z_crossattn: bool = False,
        refmap_input_scaler: Optional[float] = None,
        first_stage_config: OmegaConf = OmegaConf.create({"target": "ldm.models.autoencoder.IdentityFirstStage"}),
        cond_stage_config: OmegaConf = "__is_first_stage__",
        cache_refmap: bool = False,
        refmap_cache_root: Optional[str] = None,
        envmap_dir: Optional[str] = None,
    ):
        """

        :param OmegaConf illnet_config: the config of unet for illnet
        :param OmegaConf refnet_config: the config of encoder for refnet
        :param OmegaConf renderer_config: the config of renderer for reflectance maps.
        :param int max_timesteps: the maximum timestep when sampling, defaults to 250
        :param str loss_type: the loss type of reflectance maps and reflectances, defaults to "l1"
        :param str ckpt_path: the path of checkpoint for load pretrained model, defaults to None
        :param bool init_from_ckpt_verbose: if True, outputs missing parameters and others. defaults to True
        :param list ignore_keys: the ignored keys when loading model parameters, defaults to []
        :param str monitor: the monitored indicator, defaults to "val/loss"
        :param bool use_ema: if True, use exponential moving average model parameter, defaults to True
        :param str input_key: the key of input reflectance maps, defaults to "LrK"
        :param float sigma_for_cond_xK: _description_, defaults to 0.0
        :param int image_size: the input/output image size of illnet/refnet, defaults to 128
        :param int channels: the input/output channel of illnet/refnet, defaults to 3
        :param int log_every_k: the interval for logging intermediates when sampling, defaults to 5
        :param str parameterization: the parameterization mode, defaults to "residual"
        :param OmegaConf scheduler_config: the config of training scheduler, defaults to None
        :param bool cond_stage_trainable: if True, the encoder for condition is trained, defaults to False
        :param bool concat_mode: if True, the conditioning mode is concat, defaults to False
        :param Optional[str] cond_stage_forward: the encoding function of the condition encoder, defaults to None
        :param Optional[str] conditioning_key: the conditioning mode is concat, defaults to None
        :param float scale_factor: the factor scaling input, defaults to 1.0
        :param bool scale_by_std: if True, calculates the scaling factor based on first batch, defaults to False
        :param float l_refmap_weight: loss weight for reflectance maps, defaults to 1.0
        :param float l_refcode_weight: loss weight for reflectances, defaults to 1.0
        :param float sigma: the standard deviation on forward process, defaults to 0.01
        :param float delta: the standard deviation on reverse process, defaults to 0.0125
        :param float gamma: the base of the reflectance transition, defaults to 0.9
        :param float epsilon: The threshold range considered to be identical to z0, defaults to 0.001
        :param bool train_with_zk_gt: if True, inputs ground truth reflectance to illnet, defaults to False
        :param Optional[int] train_with_zk_gt_switch_epoch: when train_with_zk_gt is switched, defaults to None
        :param List[str] brdf_param_names: the names of brdf parameter, defaults to ["specular"]
        :param List[float] z0: the brdf parameter corresponds to fully mirror reflection, defaults to [1.0]
        :param bool model_emb_z: if True, embed the brdf parameter in illnet, defaults to True
        :param bool emb_z_crossattn: if True, embed the brdf parameter as cross attention, defaults to False
        :param Optional[float] refmap_input_scaler: scaling reflectance maps based on refmap_input_scaler and mean luminance, defaults to None
        :param OmegaConf first_stage_config: the config for autoencoder, defaults to ldm.models.autoencoder.IdentityFirstStage.
        :param OmegaConf cond_stage_config: the config for encoding condition, defaults to __is_first_stage__.
        :param bool cache_refmap: if True, saves rendered reflectance maps to refmap_cache_root, defaults to False
        :param Optional[str] refmap_cache_root: the root directory where rendered reflectance maps are saved, defaults to None
        :param Optional[str] envmap_dir: the directory where environment maps are stored, defaults to None
        """

        super().__init__()
        assert parameterization in ["residual"], 'currently only supporting "residual"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")

        self.log_every_k = log_every_k
        self.input_key = input_key
        self.sigma_for_cond_xK = sigma_for_cond_xK
        self.image_size = image_size
        self.channels = channels
        self.max_timesteps = max_timesteps

        self.brdf_param_names = brdf_param_names
        self.gamma = gamma
        self.epsilon = epsilon
        self._z0 = torch.tensor(z0, dtype=torch.float32)
        self.zdim = len(self._z0)
        self.z0: torch.Tensor
        self.register_buffer("z0", self._z0)
        self.instantiate_brdf_model(renderer_config)

        assert concat_mode, "This model only supports concat mode"
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None
        self.illnet_model = ZEmbDiffusionWrapper(illnet_config, conditioning_key, self.zdim, model_emb_z, emb_z_crossattn)
        count_params(self.illnet_model, verbose=True)
        self.refnet_model = DiffusionWrapper(refnet_config, conditioning_key)
        count_params(self.refnet_model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.illnet_model_ema = LitEma(self.illnet_model)
            self.refnet_model_ema = LitEma(self.refnet_model)
            print(f"Keeping EMAs of {len(list(self.illnet_model_ema.buffers()))} and {len(list(self.refnet_model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.l_refmap_weight = l_refmap_weight
        self.l_refcode_weight = l_refcode_weight

        if monitor is not None:
            self.monitor = monitor

        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.scale_by_std = scale_by_std
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, verbose=init_from_ckpt_verbose)

        self.loss_type = loss_type

        self.sigma = sigma
        self.delta = delta

        self.train_with_zk_gt = train_with_zk_gt
        self.train_with_zk_gt_switch_epoch = train_with_zk_gt_switch_epoch

        self.refmap_input_scaler = refmap_input_scaler
        self.cache_refmap = cache_refmap
        self.refmap_cache_root = Path(refmap_cache_root) if refmap_cache_root is not None else None
        self.envmap_dir = Path(envmap_dir) if envmap_dir is not None else None

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.illnet_model_ema.store(self.illnet_model.parameters())
            self.refnet_model_ema.store(self.refnet_model.parameters())
            self.illnet_model_ema.copy_to(self.illnet_model)
            self.refnet_model_ema.copy_to(self.refnet_model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.illnet_model_ema.restore(self.illnet_model.parameters())
                self.refnet_model_ema.restore(self.refnet_model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

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
        missing, unexpected = (
            self.load_state_dict(sd, strict=False) if not only_model else self.illnet_model.load_state_dict(sd, strict=False)
        )
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0 and verbose:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0 and verbose:
            print(f"Unexpected Keys: {unexpected}")

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1.0, "rather not use custom rescaling and std-rescaling simultaneously"
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.input_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    @torch.no_grad()
    def on_train_epoch_start(self):
        if self.train_with_zk_gt_switch_epoch is not None and self.current_epoch == self.train_with_zk_gt_switch_epoch:
            self.train_with_zk_gt = not self.train_with_zk_gt

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != "__is_first_stage__"
            assert config != "__is_unconditional__"
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def instantiate_brdf_model(self, config):
        self.renderer: MitsubaRefMapRenderer = instantiate_from_config(config)
        white_env = torch.ones(*self.renderer.envmap_size, 3)
        sensor_dict = {
            "type": "refmapsensor",
            "to_world": mi.ScalarTransform4f.look_at(origin=[0, 0, 1.1], target=[0, 0, 0], up=[0, 1, 0]),
            "film": {"type": "hdrfilm", "width": self.image_size, "height": self.image_size, "rfilter": {"type": "box"}},
            "sampler": {"type": "stratified", "sample_count": self.renderer.spp},
        }
        basis_sensor = mi.load_dict(sensor_dict)
        basis_r0 = self.renderer.rendering(
            self.z0,
            self.brdf_param_names,
            envmap=white_env,
            sensor=basis_sensor,
        )
        basis_r0 = basis_r0.permute(2, 0, 1).contiguous()  # [3, Height, Width]
        self.basis_r0: torch.Tensor
        self.register_buffer("basis_r0", basis_r0, persistent=False)
        return None

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        return LatentDiffusion.decode_first_stage(self, z, predict_cids, force_not_quantize)

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def apply_model(self, model: DiffusionWrapper, input_refmap: torch.Tensor, k: Union[torch.Tensor, int], cond):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = "c_concat" if model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}
        if isinstance(k, int):
            k = torch.full((input_refmap.size(0),), k, device=input_refmap.device)

        return model(input_refmap, k, **cond)

    def get_brdf_out(self, brdf_model_out: torch.Tensor, reversed_k: Optional[torch.Tensor] = None) -> torch.Tensor:
        zK = brdf_model_out
        _, _, zk = self.get_schedule(zK, reversed_k=reversed_k)
        if not self.training:
            zk = zk.clamp(0, 1) if zk is not None else zk
            zK = zK.clamp(0, 1) if zK is not None else zK
        return zk, zK

    def get_loss(self, pred: torch.Tensor, target: torch.Tensor, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, Lr_k, Lr_km1, z_k, z_K, K, k, illnet_cond, refnet_cond):
        reversed_k = K - k - 1
        if self.sigma > 0:  # forward noise
            noise = torch.randn_like(Lr_k)
            Lr_k = Lr_k + self.sigma * noise

        if self.parameterization == "residual":
            with torch.no_grad():
                Lr_target = Lr_km1 - Lr_k
        else:
            raise NotImplementedError()

        if self.training and self.train_with_zk_gt:
            z_out = self.apply_model(self.refnet_model, Lr_k, reversed_k, refnet_cond)
            Delta = z_k - self.z0
            model_out = self.apply_model(self.illnet_model, Lr_k, Delta, illnet_cond)
        else:
            model_out, z_out = self(Lr_k, illnet_cond, refnet_cond, reversed_k)

        zk_out, zK_out = self.get_brdf_out(z_out, reversed_k)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        loss_refmap = self.get_loss(model_out[K != 0], Lr_target[K != 0], mean=True)

        loss_dict.update({f"{prefix}/loss_refmap": loss_refmap})

        loss_refcode_zk = self.get_loss(zk_out, z_k, mean=True)
        loss_refcode_zK = self.get_loss(zK_out, z_K, mean=True)
        loss_refcode = (loss_refcode_zk + loss_refcode_zK) / 2

        loss_dict.update({f"{prefix}/loss_refcode": loss_refcode})

        loss = self.l_refmap_weight * loss_refmap + self.l_refcode_weight * loss_refcode

        loss_dict.update({f"{prefix}/loss": loss})
        return loss, loss_dict

    def forward(self, Lr_k, illnet_cond, refnet_cond, reversed_k) -> torch.Tensor:
        z_out = self.apply_model(self.refnet_model, Lr_k, reversed_k, refnet_cond)
        zk, _ = self.get_brdf_out(z_out, reversed_k=reversed_k)
        Delta = zk - self.z0
        return self.apply_model(self.illnet_model, Lr_k, Delta, illnet_cond), z_out

    def get_schedule(
        self,
        zK: torch.Tensor,
        z0: Optional[torch.Tensor] = None,
        reversed_k: Optional[Union[torch.Tensor, int]] = None,
        normalized_k: Optional[torch.Tensor] = None,
        return_zkm1: bool = False,
        power_precision: torch.dtype = torch.double,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """get scheduled z_k and K from zK, normalized_k/reversed_k and z0

        :param torch.Tensor zK: (*, N_param) reflectance parameter  corresponds to an object.
        :param Optional[torch.Tensor] z0: (N_param) mirror reflectance parameter. Defaults to None, wich means self.z0.
        :param Optional[Union[torch.Tensor, int]] reversed_k: {0, ..., K} corresponding to {K, ..., 0}. Either it or normalized_k must be specified,
            and they are exclusive. Mainly be used when sampling. Defaults to None.
        :param Optional[torch.Tensor] normalized_k: (*,) continuous k normalized in [0, 1) corresponding to [0, K). Either it or reversed_k must be specified,
            and they are exclusive. Mainly used by dataset. Defaults to None.
        :param bool return_zkm1: return z_km1 too or not, Defaults to False.
        :param torch.dtype power_precision: the precision of power to ensure consistency of computational results between CPU and GPU. Defaults to torch.double.
        :return Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]: the max timestep K (*,), the sampled timestep k (*,)(*,), reflectance parameter z_k (*, N_param), and optional previous reflectance parameter z_km1 (*, N_param) if return_zkm1 is True.
        """

        z0 = self.z0 if z0 is None else z0.to(zK.device)
        Delta_K = zK - z0  # [Bs, num_param]
        log_gamma = math.log(self.gamma)
        distance = torch.linalg.norm(Delta_K, dim=-1)
        K = (torch.log(self.epsilon / distance) / math.log(self.gamma)).int() + 2  # take one more step after zk enter epsilon
        assert (normalized_k is None) ^ (reversed_k is None), "normalized_k and reversed_k are exclusive"
        if normalized_k is not None:
            K = K.clip(min=1).int()
            k = (normalized_k * K).int()
            reversed_k = K - k - 1
        elif reversed_k is not None:
            k = K - reversed_k - 1
            if isinstance(reversed_k, int):
                reversed_k = torch.tensor([reversed_k], device=zK.device)
        reversed_k = reversed_k.to(power_precision)
        Delta_k = torch.exp(reversed_k.unsqueeze(-1) * log_gamma).float() * Delta_K
        zk = Delta_k + z0
        if return_zkm1:
            Delta_km1 = torch.exp((reversed_k + 1).unsqueeze(-1) * math.log(self.gamma)).float() * Delta_K
            zkm1 = Delta_km1 + z0
            return K, k, zk, zkm1
        return K, k, zk

    @torch.no_grad()
    def get_input(
        self,
        batch,
        return_Lr_zero=False,
        return_envmap=False,
        return_envmap_name=False,
        return_view_from=False,
        bs: Optional[int] = None,
    ):
        zK = batch["zK"]
        bs = min(len(zK), bs) if bs is not None else len(zK)
        self.batch_size = bs
        zK = zK[:bs]
        envmap_name: List[str] = batch["envmap_name"][:bs]

        view_from: torch.Tensor = batch.get("view_from")[:bs]

        K, k, zk, zkm1 = batch["K"][:bs], batch["k"][:bs], batch["zk"][:bs], batch["zkm1"][:bs]

        Lr_z_pairs = [("LrK", zK), ("Lrk", zk), ("Lrkm1", zkm1)]
        stacked_z = torch.stack([zK, zk, zkm1])
        if return_Lr_zero:
            Lr_z_pairs.append(("r0", self.z0[None].expand(bs, -1)))
            stacked_z = torch.concat([stacked_z, self.z0[None].expand(bs, -1)[None]])

        stacked_Lr: List[torch.Tensor] = []
        not_cached_r = []
        for refmap_key, z_batch in Lr_z_pairs:
            if refmap_key in batch:
                r = batch[refmap_key][:bs]
            else:
                r = torch.empty(bs, 3, self.renderer.image_size, dtype=torch.float, device=self.device)
                r[:, 0, 0, 0] = torch.nan
            stacked_Lr.append(r)
            not_cached_r.append(torch.isnan(r[:, 0, 0, 0]))  # [batch]
        not_cached_r = torch.stack(not_cached_r)  # [stack, batch]

        # load envmap for rendering
        if not_cached_r.any():
            assert self.envmap_dir is not None, "envmap_dir was needed, but was not specified"
            if "envmap" in batch:
                envmap = batch["envmap"][:bs]
            else:
                envmap = torch.empty(bs, *self.renderer.envmap_size, 3, dtype=torch.float, device=self.device)
                envmap[:, 0, 0, 0] = torch.nan
            need_to_load_envmap_idxs = torch.nonzero(torch.logical_and(not_cached_r.any(dim=0), torch.isnan(envmap[:, 0, 0, 0])))
            need_to_load_envmap_names = [envmap_name[i] for i in need_to_load_envmap_idxs]
            if len(need_to_load_envmap_idxs) > 0:
                with ThreadPoolExecutor(max_workers=20) as executer:
                    envmaps = executer.map(lambda name: load_exr(self.envmap_dir / f"{name}.exr", as_torch=True), need_to_load_envmap_names)
                    for idx, em in zip(need_to_load_envmap_idxs, envmaps):
                        envmap[idx] = em.cuda()
        else:
            envmap = None

        # rendering refmap which is not cached
        pre_batch_idx = -1  # for efficiency of calculation pdf of envmaps
        for batch_idx, stack_idx in torch.nonzero(not_cached_r.T):
            stacked_Lr[stack_idx][batch_idx] = self.renderer.rendering(
                stacked_z[stack_idx, batch_idx],
                self.brdf_param_names,
                envmap=envmap[batch_idx] if pre_batch_idx != batch_idx else None,
                view_from=view_from[batch_idx] if pre_batch_idx != batch_idx else None,
                channel_first=True,
            )
            pre_batch_idx = batch_idx

        # cache rendered results
        if not_cached_r.any() and self.cache_refmap:
            assert self.refmap_cache_root is not None, "cache_refmap is True, but refmap_cache_root is not specified"
            size = self.renderer.refmap_res
            denoise_suffix = f"_{self.renderer.denoise}denoise" if self.renderer.denoise else ""
            refmap_cache_dir = Path(
                f'{self.refmap_cache_root}/{"-".join(self.brdf_param_names)}/{size}x{size}_spp{self.renderer.spp}{denoise_suffix}/'
            )
            refmap_cache_dir.mkdir(exist_ok=True, parents=True)
            stacked_z_cpu = stacked_z.cpu()
            view_from_cpu = view_from.cpu()
            torch.set_printoptions(precision=4, sci_mode=True)

            def thread_func(stack_idx: int, batch_idx: int) -> None:
                pieces_key = "b" + str(stacked_z_cpu[stack_idx, batch_idx])[7:-1] + "v" + str(view_from_cpu[batch_idx])[7:-1]
                pieces_key = pieces_key.replace("\n", "").replace(" ", "")
                filename = pieces_key + ".pt"
                piece_cache_file_path = refmap_cache_dir / envmap_name[batch_idx] / filename
                if piece_cache_file_path.exists() and (not not_cached_r[stack_idx, batch_idx] or self.rendering_denoise):
                    return
                piece_cache_file_path.parent.mkdir(exist_ok=True)
                data = {
                    "key": pieces_key,
                    "envmap_name": envmap_name[batch_idx],
                    "brdf_param_names": self.brdf_param_names,
                    "zk": stacked_z_cpu[stack_idx, batch_idx].clone(),
                    "view_from": view_from_cpu[batch_idx].clone(),
                    "filename": filename,
                    "rendering_results": stacked_Lr[stack_idx][batch_idx].cpu(),
                    "envmap_size": self.renderer.envmap_size,
                    "refmap_res": self.renderer.refmap_res,
                    "spp": self.renderer.spp,
                }
                torch.save(data, piece_cache_file_path)

            with ThreadPoolExecutor(max_workers=20) as executer:
                for stack_idx, batch_idx in torch.nonzero(not_cached_r):
                    executer.submit(thread_func, stack_idx, batch_idx)

        if self.refmap_input_scaler is not None:
            LrK_orig = stacked_Lr[0]
            L = 0.212671 * LrK_orig[:, 0] + 0.715160 * LrK_orig[:, 1] + 0.072169 * LrK_orig[:, 2]
            L_mask = L > 0
            L_mean = torch.exp((torch.log(L.clip(1e-5)) * L_mask).sum(dim=(1, 2)) / L_mask.sum(dim=(1, 2)))
            self.normalizing_scale = self.refmap_input_scaler / L_mean
            for idx, Lr in enumerate(stacked_Lr):
                stacked_Lr[idx] = Lr * self.normalizing_scale[:, None, None, None]

        for idx, Lr in enumerate(stacked_Lr):
            stacked_Lr[idx] = self.ds.transform(Lr)

        if return_Lr_zero:
            Lr_K, Lr_k, Lr_km1, Lr_0 = list(stacked_Lr)
        else:
            Lr_K, Lr_k, Lr_km1 = list(stacked_Lr)
            Lr_0 = None

        Lr_K = self.get_first_stage_encoding(self.encode_first_stage(Lr_K))
        Lr_k = self.get_first_stage_encoding(self.encode_first_stage(Lr_k))
        Lr_km1 = self.get_first_stage_encoding(self.encode_first_stage(Lr_km1))
        if Lr_0 is not None:
            Lr_0 = self.get_first_stage_encoding(self.encode_first_stage(Lr_0))

        illnet_c = list()
        cond_LrK = Lr_K
        if self.sigma_for_cond_xK > 0:
            cond_LrK = self.sigma_for_cond_xK * torch.randn_like(Lr_K) + cond_LrK
        illnet_c.append(cond_LrK)

        refnet_c = illnet_c

        out = [K, k, Lr_K, Lr_k, Lr_km1, zK, zk, illnet_c, refnet_c]
        if return_Lr_zero:
            out.append(Lr_0)
        if return_envmap:
            out.append(envmap)
        if return_envmap_name:
            out.append(batch["envmap_name"][:bs])
        if return_view_from:
            out.append(view_from)
        return out

    # dataset has transform and rescale function.
    def on_train_epoch_start(self) -> None:
        self.ds = getattr(self.trainer.datamodule, f"train_ds", None)

    def on_validation_epoch_start(self) -> None:
        self.ds = getattr(self.trainer.datamodule, f"val_ds", None)

    def on_test_epoch_start(self) -> None:
        self.ds = getattr(self.trainer.datamodule, f"test_ds", None)

    def on_predict_epoch_start(self) -> None:
        self.ds = getattr(self.trainer.datamodule, f"predict_ds", None)

    @torch.no_grad()
    def rendering_refmaps(
        self,
        envmaps: Union[List[str], torch.Tensor],
        z: torch.Tensor,
        brdf_param_names: List[str] = None,
        transform: bool = True,
        view_from: torch.Tensor = None,
        new_scene: bool = False,
    ) -> torch.Tensor:
        """
        envmaps: List[envmap names] or Tensor[B, H, W, C]
        z: [list, BS, NumParams]
        Params = [color_scalar, roughness, specular]
        out: [list, BS, channel, height, width]
        """
        assert len(envmaps) == z.size(1)
        BS = len(envmaps)
        num_per_batch = z.size(0)
        if isinstance(envmaps[0], str):
            assert self.envmap_dir is not None, "envmap_dir was needed, but was not set"
            envmaps = list()
            for name in envmaps:
                envmaps.append(load_exr(self.envmap_dir / f"{name}.exr", as_torch=True).cuda())

        brdf_param_names = brdf_param_names or self.brdf_param_names

        rendered_img = torch.empty(num_per_batch, BS, 3, *self.renderer.image_size, dtype=torch.float32, device=z.device)
        for batch_idx, z_list in enumerate(z.transpose(0, 1)):
            for list_idx, z_item in enumerate(z_list):
                rendered_img[list_idx, batch_idx] = self.renderer.rendering(
                    z_item,
                    brdf_param_names,
                    envmaps[batch_idx] if list_idx == 0 or new_scene else None,
                    view_from=view_from[batch_idx] if view_from is not None and (list_idx == 0 or new_scene) else None,
                    new_scene=new_scene,
                    channel_first=True,
                )

        return rendered_img

    def shared_step(self, batch):
        K, k, Lr_K, Lr_k, Lr_km1, zK, zk, illnet_c, refnet_c = self.get_input(batch)
        loss, loss_dict = self.p_losses(Lr_k, Lr_km1, zk, zK, K, k, illnet_c, refnet_c)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.log(
            "global_step",
            float(self.global_step),
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

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + "_ema": loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(
            loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size
        )
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.illnet_model_ema(self.illnet_model)
            self.refnet_model_ema(self.refnet_model)

    def check_convergence(self, zk: torch.Tensor):
        Deltak = (zk - self.z0).abs()
        distance = torch.linalg.norm(Deltak, dim=-1)
        return torch.logical_or(distance < self.epsilon, distance == 0)

    def p_mean_variance(
        self,
        Lr_k: torch.Tensor,
        illnet_cond: torch.Tensor,
        refnet_cond: torch.Tensor,
        reversed_k: Union[int, torch.Tensor],
        return_model_out: bool = False,
    ):
        model_out, z_out = self(Lr_k, illnet_cond, refnet_cond, reversed_k)

        if self.parameterization == "residual":
            model_mean = Lr_k + model_out
        else:
            raise NotImplementedError()

        if return_model_out:
            return model_mean, self.delta, z_out, model_out
        else:
            return model_mean, self.delta, z_out

    def p_sample(
        self,
        Lr_k: torch.Tensor,
        illnet_cond: torch.Tensor,
        refnet_cond: torch.Tensor,
        reversed_k: Union[int, torch.Tensor],
        return_model_out: bool = False,
    ):
        raise NotImplementedError("")

    @torch.no_grad()
    def p_sample_loop(
        self,
        Lr_K: torch.Tensor,
        illnet_cond: torch.Tensor,
        refnet_cond: torch.Tensor,
        return_intermediates=False,
        verbose=True,
        log_every_k=None,
    ):
        log_every_k = log_every_k or self.log_every_k
        device = self.device
        batch_size = Lr_K.size(0)

        Lr_K = Lr_K + self.delta * torch.randn_like(Lr_K)
        Lr_k = Lr_K.clone()

        if return_intermediates:
            intermediates = {"Lrk_inter": [Lr_K], "zk_inter": []}
        iterator = range(self.max_timesteps)
        if verbose:
            iterator = tqdm(iterator, desc="Sampling Lr_0")

        subbatch_mask = torch.ones(batch_size, device=device, dtype=bool)
        K = torch.full((batch_size,), fill_value=self.max_timesteps, device=device, dtype=torch.int32)
        zK = torch.full((batch_size, self.zdim), fill_value=torch.nan, device=device, dtype=torch.float32)

        for i in iterator:
            subbatch_illnet_c = [c[subbatch_mask] for c in illnet_cond]
            subbatch_refnet_c = [c[subbatch_mask] for c in refnet_cond]
            sub_model_mean, sub_model_variance, subz = self.p_mean_variance(
                Lr_k[subbatch_mask],
                subbatch_illnet_c,
                subbatch_refnet_c,
                reversed_k=i,
            )

            subzk, subzK = self.get_brdf_out(subz, reversed_k=i)
            not_converge = ~self.check_convergence(subzk)

            sub_Lr_k = sub_model_mean
            sub_Lr_k[not_converge] += torch.randn_like(sub_model_mean[not_converge]) * sub_model_variance

            Lr_k[subbatch_mask] = sub_Lr_k

            if i % log_every_k == 0 and return_intermediates:
                z_log = torch.full((Lr_K.size(0), self.zdim), torch.nan, device=device)
                z_log[subbatch_mask] = subzk
                intermediates["zk_inter"].append(z_log)

                Lr_k_log = torch.zeros_like(Lr_k)
                Lr_k_log[subbatch_mask] = sub_Lr_k
                intermediates["Lrk_inter"].append(Lr_k_log)

            converged_mask = torch.where(subbatch_mask)[0][~not_converge]
            K[converged_mask] = i + 1
            zK[converged_mask] = subzK[~not_converge]
            subbatch_mask[converged_mask] = False

            if not torch.any(subbatch_mask):
                break

        if return_intermediates:
            return Lr_k, zK, K, intermediates
        else:
            return Lr_k, zK, K

    @torch.no_grad()
    def progressive_denoising(
        self,
        Lr_K: torch.Tensor,
        illnet_cond: torch.Tensor,
        refnet_cond: torch.Tensor,
        return_intermediates=False,
        verbose=True,
        log_every_k=None,
    ):
        raise NotImplementedError("this is not supported")

    def _get_grid_from_list(self, samples: List[torch.Tensor], verbose: bool = True):
        iterator = tqdm(samples, desc="making denoise grid from list") if verbose else samples
        denoise_row = []
        for zd in iterator:
            denoise_row.append(self.ds.rescale(self.decode_first_stage(zd)))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_setp, n_row, C, ...
        denoise_row = rearrange(denoise_row, "n b c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_row, nrow=n_imgs_per_row)
        return denoise_grid

    def get_diffusion_grid(
        self,
        envmaps: Union[List[str], torch.Tensor],
        zKs: torch.Tensor,
        Ks: torch.Tensor,
        view_froms: Optional[torch.Tensor] = None,
        log_every_k: Optional[int] = None,
        verbose: bool = True,
    ) -> np.ndarray:
        log_every_k = log_every_k or self.log_every_k
        max_log_length = (Ks.max().item() + log_every_k) // log_every_k
        view_froms = [None] * len(envmaps) if view_froms == None else view_froms
        z0 = self.ds.z0.to(zKs.device) if self.ds.z0 is not None else self.z0

        iterator = zip(envmaps, zKs, Ks, view_froms)
        if verbose:
            iterator = tqdm(iterator, desc="Making diffusion grid", total=len(envmaps))
        diffusion_grid = torch.zeros(len(envmaps), max_log_length, 3, *self.renderer.image_size, dtype=torch.float32)
        for idx, (envmap, zK, K, view_from) in enumerate(iterator):
            zks = [z0]
            ks = [0]
            for k in range(1, K + 1):
                if k % log_every_k == 0:
                    _, _, zk = self.get_schedule(zK, z0=z0, reversed_k=K - k)
                    zks.append(zk)
                    ks.append(k)
            xks = self.rendering_refmaps([envmap], torch.stack(zks)[:, None], self.brdf_param_names, view_from=[view_from])[:, 0]

            if self.refmap_input_scaler is not None:
                LrK_orig = xks[0]
                L = 0.212671 * LrK_orig[0] + 0.715160 * LrK_orig[1] + 0.072169 * LrK_orig[2]
                L_mask = L > 0
                L_mean = torch.exp((torch.log(L.clip(1e-5)) * L_mask).sum(dim=(1, 2)) / L_mask.sum(dim=(1, 2)))
                normalizing_scale = self.refmap_input_scaler / L_mean
                xks *= normalizing_scale
            xks = self.get_first_stage_encoding(self.encode_first_stage(self.ds.transform(xks)))
            if self.nondeterministic:
                xks[1:] += self.sigma * torch.randn_like(xks[1:])
            xks = self.ds.rescale(self.decode_first_stage(xks))
            diffusion_grid[idx, : xks.size(0)] = xks
        diffusion_grid = rearrange(diffusion_grid, "b n c h w -> (b n) c h w")
        diffusion_grid = make_grid(diffusion_grid, nrow=max_log_length)
        return diffusion_grid.permute(1, 2, 0).cpu().numpy()

    def get_visualized_brdf_grid(
        self,
        zs: torch.Tensor,
        brdf_param_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        z : [BS, params]
        return: [H, W, 3]
        """
        rows = []
        for z in zs:
            bsdf = get_bsdf(z, brdf_param_names=brdf_param_names or self.brdf_param_names)
            rows.append(visualize_bsdf(bsdf, imsize=(128, 128))[0])
        return np.concatenate(rows, axis=0)

    def r0toenvmap(self, r0: torch.Tensor, envshape: Optional[Tuple[int]] = None) -> torch.Tensor:
        """
        r0 : rescaled [BS, Channel, Height, Width]
        envshape: (Height, Width)
        out : [BS, Height, Width, 3]
        """
        if envshape is None:
            envshape = (self.renderer.image_size[0], self.renderer.image_size[1] * 2)
        r0 = r0 / self.basis_r0
        envmap = mirmap2envmap(r0, envshape)
        return envmap.permute(0, 2, 3, 1)

    def reconstruct(self, Lr_0: torch.Tensor, z: torch.Tensor, brdf_param_names=None, transform: bool = True) -> torch.Tensor:
        """
        Lr_0: transformed [BS, channel, height, width]
        z:  [BS, params]
        return: x (scaled r)
        """
        r0 = self.ds.rescale(Lr_0)
        envmap = self.r0toenvmap(r0, (self.image_size, self.image_size * 2))
        recon = self.rendering_refmaps(envmap, z[None], new_scene=True, brdf_param_names=brdf_param_names, transform=transform)[0]
        return recon

    @torch.no_grad()
    def log_images(
        self,
        batch,
        split,
        N=10,
        sample=True,
        plot_denoised_rows=True,
        plot_diffusion_rows=False,
        plot_progressive_rows=False,
        return_zk_sequence: bool = True,
        reconstruction: bool = True,
    ):
        log = dict()
        other_logs = dict()
        K, _, Lr_K, _, _, zK, _, illnet_c, refnet_c, Lr_0, envmap, envmap_name, view_from = self.get_input(
            batch,
            return_Lr_zero=True,
            return_envmap=True,
            return_envmap_name=True,
            return_view_from=True,
            bs=N,
        )

        if self.refmap_input_scaler is not None:
            other_logs["scale"] = self.normalizing_scale
        log["LrK"] = Lr_K
        log["Lr0_gt"] = Lr_0
        log["brdf_gt"] = self.get_visualized_brdf_grid(zK, self.brdf_param_names)
        other_logs["zK_gt"] = zK

        if plot_diffusion_rows:
            envmap = envmap if isinstance(envmap, torch.Tensor) and ~torch.isnan(envmap[:, 0, 0, 0]).any() else envmap_name
            diffusion_grid = self.get_diffusion_grid(envmap, zK, K, view_from, self.log_every_k)
            log["diffusion_row"] = diffusion_grid

        if sample:
            with self.ema_scope("Plotting"):
                Lr0_est, zK_est, K_est, intermediate = self.p_sample_loop(Lr_K, illnet_c, refnet_c, return_intermediates=True)
            Lr0_est = self.decode_first_stage(Lr0_est)
            log["Lr0_est"] = Lr0_est
            other_logs["zK"] = zK_est  # [BS, zK]
            other_logs["K"] = K_est  # [BS]
            log["brdf_est"] = self.get_visualized_brdf_grid(zK_est, self.brdf_param_names)
            if plot_denoised_rows:
                denoise_grid = self._get_grid_from_list(intermediate["Lrk_inter"])
                log["denoise_row"] = denoise_grid.permute(1, 2, 0).cpu().numpy()
            if return_zk_sequence and "zk_inter" in intermediate:
                other_logs["zk_sequence"] = torch.stack(intermediate["zk_inter"], dim=1)  # [BS, seq, zk]
            if reconstruction:
                log["reconstruction"] = self.reconstruct(Lr0_est, zK_est, self.brdf_param_names)

        other_logs["envmap_name"] = envmap_name
        other_logs["view_from"] = view_from

        return log, other_logs

    @torch.no_grad()
    def get_input_for_predict(
        self,
        batch,
        bs: Optional[int] = None,
    ):
        LrK = batch[self.input_key]
        bs = min(len(LrK), bs) if bs is not None else len(LrK)
        LrK = LrK[:bs]
        if self.refmap_input_scaler is not None:
            L = 0.212671 * LrK[:, 0] + 0.715160 * LrK[:, 1] + 0.072169 * LrK[:, 2]
            self.normalizing_scale = self.refmap_input_scaler / torch.exp(
                (torch.log(L.clip(1e-5)) * (L > 0)).sum(dim=(1, 2)) / (L > 0).sum(dim=(1, 2))
            )
        if self.refmap_input_scaler is not None:
            LrK = LrK * self.normalizing_scale[:, None, None, None]
        LrK = self.ds.transform(LrK)
        Lr0 = batch.get("Lr0")
        if Lr0 is not None:
            if self.refmap_input_scaler is not None:
                Lr0 = Lr0[:bs] * self.normalizing_scale[:, None, None, None]
            Lr0 = self.ds.transform(Lr0)

        LrK = self.get_first_stage_encoding(self.encode_first_stage(LrK))
        tag: List[str] = batch["tag"][:bs]

        illnet_c = list()
        cond_LrK = LrK
        if self.sigma_for_cond_xK > 0:
            cond_LrK = self.sigma_for_cond_xK * torch.randn_like(LrK) + cond_LrK
        illnet_c.append(cond_LrK)

        refnet_c = illnet_c

        return LrK, Lr0, illnet_c, refnet_c, tag

    def configure_optimizers(self):
        lr: float = self.learning_rate
        params = list(self.illnet_model.parameters()) + list(self.refnet_model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert "target" in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [{"scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule), "interval": "step", "frequency": 1}]
            return [opt], scheduler
        return opt
