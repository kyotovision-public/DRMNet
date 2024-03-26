import torch

from .transform import xyz2thetaphi


def refmap_mask_make(
    colors: torch.Tensor,  # [n, 3]
    normals: torch.Tensor,  # [n, 3]
    res: int,
    angle_threshold: float = None,
    min_points=0,
    refmap_batch_size=512,
):
    device = colors.device
    Height = Width = res
    theta = (torch.arange(Height, device=device) + 0.5) * (torch.pi / Height)
    phi = (torch.arange(Width, device=device) + 0.5) * (torch.pi / Width)
    thetaphi = torch.stack(torch.meshgrid(theta, phi, indexing="ij"), -1)
    # refmap_normal = thetaphi2xyz(thetaphi, normal=[0, 1, 0], tangent=[-1, 0, 0])
    thetaphi_normals = xyz2thetaphi(normals, normal=[0, 1, 0], tangent=[-1, 0, 0])  # [n, 2(tp)]
    refmap = torch.zeros((res * res), colors.size(-1), device=device, dtype=colors.dtype)
    refmask = torch.zeros((res * res), device=device, dtype=torch.bool)
    for i in range((res * res - 1) // refmap_batch_size + 1):
        batch_slice = slice(i * refmap_batch_size, min(res * res, (i + 1) * refmap_batch_size))
        refmap_thetaphi_batch = thetaphi.view(-1, 2)[batch_slice]  # [bs, 2(tp)]
        angles = (refmap_thetaphi_batch[:, None] - thetaphi_normals[None]).abs().amax(dim=-1)  # [bs, n]
        angle_mask = angles > angle_threshold
        angle_mask.masked_fill_((~angle_mask).sum(-1, keepdim=True) < min_points, True)

        expanded_colors = colors.sum(-1).expand(refmap_thetaphi_batch.size(0), -1).masked_fill(angle_mask, torch.nan)  # [bs, n]
        medians, indices = torch.nanmedian(expanded_colors, dim=-1)  # [bs]
        median_mask = torch.isnan(medians)
        if not median_mask.all():
            refmap[batch_slice][~median_mask] = colors[indices[~median_mask]]
            refmask[batch_slice][~median_mask] = True

    return refmap.view(res, res, colors.size(-1)), refmask.view(res, res)
