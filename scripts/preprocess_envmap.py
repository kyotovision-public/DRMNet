"""collect and reshape environment maps"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from utils.file_io import load_exr, save_exr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dirs", nargs="?", type=Path, help="The path of directories containing environment maps")
    parser.add_argument("--output_dir", "-o", type=Path, help="The output directory", default=Path("./data/LavalIndoor+PolyHaven_2k/"))
    parser.add_argument("--resolution", type=str, help="The target resolution as HxW", default="2000x1000")
    args = parser.parse_args()

    source_dirs: List[Path] = args.source_dirs
    output_dir: Path = args.output_dir
    output_dir.mkdir(exist_ok=True)
    resolution: Tuple[int, int] = (int(i) for i in args.resolution.split("x")[::-1])

    for source_dir in source_dirs:
        _suffix = [".hdr", ".exr"]
        for envmap_path in source_dir.glob("*.*"):
            if envmap_path.suffix not in _suffix:
                continue
            envmap = load_exr(envmap_path)
            envmap = cv2.resize(envmap)
            envmap = cv2.resize(envmap, resolution, interpolation=cv2.INTER_AREA)
            save_exr(output_dir.joinpath(envmap_path.name).with_suffix(".exr"), envmap)
