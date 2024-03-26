import numpy as np


def hdr2ldr(x: np.ndarray, mask: np.ndarray = None, alpha=0.18, gamma=2.2) -> np.ndarray:
    L = 0.212671 * x[:, :, 0] + 0.715160 * x[:, :, 1] + 0.072169 * x[:, :, 2]
    mask = np.logical_and(mask, L > 5e-5) if mask is not None else L > 5e-5
    assert mask.ndim == 2
    coeff = alpha / np.exp((np.log(L.clip(0) + 1e-7) * mask).sum() / mask.sum())
    return (x * coeff).clip(0, 1) ** (1 / gamma)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from file_io import load_exr, save_exr

    work_dir = Path(sys.argv[1])

    for path in work_dir.glob("*.hdr"):
        img = load_exr(path)
        img = hdr2ldr(img)
        save_exr(path.with_suffix(".png"), img)
