"""subprocess the shapes of [Xu et al.](https://cseweb.ucsd.edu/~viscomp/projects/SIG18Relighting/)"""

import argparse
import sys
from pathlib import Path

import mitsuba as mi
import numpy as np
import torch
from tqdm import tqdm

mi.set_variant("cuda_ad_rgb")

sys.path.append(str(Path(__file__).parent.parent))

from utils.mitsuba3_utils import load_mesh

breakpoint()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path, help="The root path of the original shape dataset Shapes_Multi_5000")
    parser.add_argument("--output_dir", "-o", type=Path, help="The output directory", default=Path("./data/DeepRelighting_shape5000/"))
    args = parser.parse_args()

    source_root: Path = args.source_root
    output_dir: Path = args.output_dir
    output_dir.mkdir(exist_ok=True)

    mesh = mi.load_dict({"type": "obj", "filename": "./data/sample.obj"})
    params = mi.traverse(mesh)

    num_shape = len(sorted(source_root.glob("Shape__*")))
    for i in tqdm(range(num_shape)):
        obj_path = source_root / f"Shape__{i}/object.obj"
        obj_dict = load_mesh(obj_path)

        vertex_positions = obj_dict["vertex_positions"].torch().view(-1, 3)
        # normalize the size of mesh
        vertex_positions *= 0.9 / torch.linalg.vector_norm(vertex_positions, dim=-1).max()
        # ensure there is no overflow on torch.int32
        assert len(vertex_positions) < 2**31

        obj_dict["vertex_positions"] = vertex_positions.cpu()
        obj_dict["vertex_normals"] = obj_dict["vertex_normals"].torch().view(-1, 3).cpu()
        obj_dict["faces"] = torch.from_numpy(obj_dict["faces"].numpy().astype(np.int32)).view(-1, 3)

        torch.save(obj_dict, output_dir / f"Shape__{i}.pt")
