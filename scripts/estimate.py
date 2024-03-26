"""collect and reshape environment maps"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import mitsuba as mi
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

mi.set_variant("cuda_ad_rgb")

sys.path.append(str(Path(__file__).parent.parent))

from dataset.basedataset import BaseDataset
from ldm.util import instantiate_from_config
from models.drmnet import DRMNet
from models.obsnet import ObsNetDiffuion
from utils.file_io import load_exr, load_png, save_png
from utils.img2refmap import refmap_mask_make
from utils.mitsuba3_utils import get_bsdf, visualize_bsdf
from utils.tonemap import hdr2ldr


def estimate(
    DRMNet_model: DRMNet,
    ObsNet_model: ObsNetDiffuion,
    input_img: torch.Tensor,
    input_normal: torch.Tensor,
    mask: torch.Tensor,
    tag: str = "sample",
    erode_kernel_size: int = 5,
):
    refmap_res = DRMNet_model.ds.size

    torch.cuda.synchronize()

    # edge removing
    if erode_kernel_size > 0:
        inv_mask = ~mask
        kernel = torch.stack(torch.meshgrid(*torch.arange(erode_kernel_size, device="cuda").expand(2, -1), indexing="ij"))
        kernel = kernel + 0.5
        kernel = torch.linalg.norm(kernel - erode_kernel_size / 2, axis=0) <= erode_kernel_size / 2
        kernel = kernel[None, None].float()
        inv_mask = torch.nn.functional.conv2d(inv_mask[None, None].float(), kernel, padding="same").bool()[0, 0]
        mask = torch.logical_and(mask, ~inv_mask)

    print("Making refmap from object image...", flush=True)
    refmap_est, refmask = refmap_mask_make(
        input_img[mask],  # [N, 3]
        input_normal[mask],
        res=refmap_res,
        angle_threshold=np.pi / 128 / 2,
    )

    torch.cuda.synchronize()
    print("Inpainting refmap ...", flush=True)

    batch = {
        "tag": [tag],
        "raw_refmap": refmap_est.permute(2, 0, 1)[None],
        "raw_refmask": refmask[None],
    }
    c, _, _ = ObsNet_model.get_cond_for_predict(batch)

    # get denoise row
    use_ddim = ObsNet_model.ddim_steps is not None
    with ObsNet_model.ema_scope("Plotting"):
        samples, _ = ObsNet_model.sample_log(
            cond=c,
            batch_size=len(batch["tag"]),
            ddim=use_ddim,
            ddim_steps=ObsNet_model.ddim_steps,
            eta=ObsNet_model.ddim_eta,
        )
    inpaint_sample: torch.Tensor = ObsNet_model.ds.rescale(ObsNet_model.decode_first_stage(samples))[0]

    torch.cuda.synchronize()
    print("Inverse Rendering ...", flush=True)

    batch = {
        "tag": [tag],
        "LrK": inpaint_sample[None],
    }
    LrK, _, illnet_c, refnet_c, _ = DRMNet_model.get_input_for_predict(batch)

    torch.cuda.synchronize()
    with DRMNet_model.ema_scope():
        samples, zK_est, _ = DRMNet_model.p_sample_loop(LrK, illnet_c, refnet_c, verbose=False)

    Lr0_sample: torch.Tensor = DRMNet_model.ds.rescale(DRMNet_model.decode_first_stage(samples))[0].clip(0)
    zK_est = zK_est[0]

    torch.cuda.synchronize()
    if DRMNet_model.refmap_input_scaler is not None:
        Lr0_sample /= DRMNet_model.normalizing_scale[0]

    return Lr0_sample, zK_est


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_img", type=Path, help="The path of HDR image for an object (.exr, .hdr)")
    parser.add_argument("input_normal", type=Path, help="The path of normal map for an object (.npy)")
    parser.add_argument("input_mask", type=Path, help="The path of mask for an object (.png)", default=None)
    parser.add_argument(
        "--obsnet_base_path", type=Path, help="the config path for obsnet", default=Path("./configs/obsnet/eval_obsnet.yaml")
    )
    parser.add_argument(
        "--drmnet_base_path", type=Path, help="the config path for drmnet", default=Path("./configs/drmnet/eval_drmnet.yaml")
    )
    parser.add_argument("--output_dir", type=Path, help="the output directory", default=Path("./outputs/"))
    args = parser.parse_args()

    # load models
    obsnet_base_config = OmegaConf.load(args.obsnet_base_path)
    obsnet_model: ObsNetDiffuion = instantiate_from_config(obsnet_base_config.model).cuda()
    obsnet_model.ds: BaseDataset = instantiate_from_config(obsnet_base_config.data.params.predict)
    drmnet_base_config = OmegaConf.load(args.drmnet_base_path)
    drmnet_model: DRMNet = instantiate_from_config(drmnet_base_config.model).cuda()
    drmnet_model.ds: BaseDataset = instantiate_from_config(drmnet_base_config.data.params.predict)

    # load input
    input_img = load_exr(args.input_img, as_torch=True).cuda()
    input_normal = torch.from_numpy(np.load(args.input_normal)).cuda()
    normal_mask = torch.linalg.norm(input_normal, dim=-1) > 0.5
    if args.input_mask is not None:
        input_mask = load_png(args.input_mask, as_torch=True).cuda()
        if input_mask.ndim == 3:
            input_mask = input_mask[:, :, 0]
        mask = torch.logical_and(input_mask, normal_mask)
    else:
        mask = normal_mask

    Lr0_sample, zK_est = estimate(drmnet_model, obsnet_model, input_img, input_normal, mask)

    envmap_est = drmnet_model.r0toenvmap(Lr0_sample[None], (drmnet_model.image_size, drmnet_model.image_size * 2))[0]  # [H, W, 3]

    output_dir: Path = args.output_dir
    output_dir.mkdir(exist_ok=True)
    envmap_est_ldr = hdr2ldr(envmap_est.cpu().numpy())
    save_png(output_dir / f"sample_env.png", envmap_est_ldr)
    vis_ref, vis_ref_mask = visualize_bsdf(get_bsdf(zK_est, drmnet_model.brdf_param_names))
    vis_ref = hdr2ldr(vis_ref)
    save_png(output_dir / f"sample_brdf.png", vis_ref, mask=vis_ref_mask)
