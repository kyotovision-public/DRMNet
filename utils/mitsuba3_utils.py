from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import drjit as dr
import mitsuba as mi
import numpy as np
import torch

mi.register_sensor("refmapsensor", lambda props: RefMapSensor(props))


class RefMapSensor(mi.Sensor):
    """sensor for reflectance map"""

    def __init__(self, props: mi.Properties):
        if props.has_property("to_world"):
            self.m_to_world = mi.Transform4f(props["to_world"].matrix)
            props.mark_queried("to_world")
        super().__init__(props)

        if self.film().rfilter().radius() > 0.5 + mi.math.RayEpsilon:
            pass

        if props.has_property("flip"):
            self.flip = mi.Bool(props["flip"])
        else:
            self.flip = mi.Bool(False)

        self.set_transforms()
        self.world_transform()
        self.m_needs_sample_3 = False

    def set_transforms(self):
        m_film: mi.Film = self.film()
        m_size: mi.ScalarVector2u = m_film.size()
        self.m_sample_to_camera = mi.Transform3f.translate(1.0) @ mi.Transform3f.scale(-2.0)
        if self.flip.all_():
            self.m_sample_to_camera = mi.Transform3f.scale([-1.0, 1.0]) @ self.m_sample_to_camera

    def sample_ray(self, time, wavelength_sample, position_sample: mi.Point2f, aperture_sample, active):
        m_film: mi.Film = self.film()
        wavelengths, wav_weight = self.sample_wavelengths(dr.zeros(mi.SurfaceInteraction3f), wavelength_sample, active)
        ray = mi.Ray3f()
        ray.time = time
        ray.wavelengths = wavelengths

        m_size: mi.ScalarVector2u = m_film.size()
        position_sample = self.m_sample_to_camera @ position_sample
        y = dr.sin(position_sample.y / 2.0 * dr.pi)
        x = dr.cos(position_sample.y / 2.0 * dr.pi) * dr.sin(position_sample.x / 2.0 * dr.pi)
        near_p: mi.Point3f = mi.Point3f(x, y, 0.0)
        ray.o = self.m_to_world @ near_p
        ray.d = self.m_to_world @ mi.Vector3f(0.0, 0.0, 1.0)
        ray.o += ray.d * mi.math.RayEpsilon

        return (ray, wav_weight)

    def sample_ray_differential(self, time, wavelength_sample, position_sample: mi.Point2f, aperture_sample, active):
        ray, wav_weight = self.sample_ray(time, wavelength_sample, position_sample, aperture_sample, active)
        return (ray, wav_weight)

    def bbox(self):
        return mi.ScalarBoundingBox3f()

    def parameters_changed(self, keys: list[str] = ...) -> None:
        super().parameters_changed(keys)
        if "flip" in keys or "film" in keys:
            self.set_transforms()

    def traverse(self, cb: mi.TraversalCallback):
        cb.put_parameter("flip", self.flip, mi.ParamFlags.NonDifferentiable)
        cb.put_parameter("to_world", self.m_to_world, mi.ParamFlags.NonDifferentiable)
        return super().traverse(cb)

    def to_string(self):
        indent = lambda x, i: str(x).replace("\n", "\n" + " " * i)
        return (
            f"RefMapSensor[\n"
            f"  film = {indent(self.film(), 2)},\n"
            f"  sampler = {indent(self.sampler(), 2)},\n"
            f"  flip = {self.flip},\n"
            f"  to_world = {indent(self.m_to_world, 13)},\n"
            f"]"
        )

    def world_transform(self):
        return self.m_to_world


class MitsubaBaseRenderer:
    def __init__(
        self,
        image_size: tuple[int, int],
        spp: int = 1024,
        envmap_size: tuple[int, int] = (1000, 2000),
        denoise: str = None,
        return_normal: bool = False,
        return_depth: bool = False,
        init_view_from: list[float] = [0, 0, 1.1],
        brdf_param_names: list[str] = None,
    ) -> None:
        self.image_size = image_size
        self.spp = spp
        self.envmap_size = envmap_size
        self.scene_dict = {
            "type": "scene",
            "integrator": {
                "type": None,
            },
            "emitter": {"type": "envmap", "bitmap": mi.Bitmap(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, envmap_size)},
            "sensor": {
                "type": None,
                "to_world": mi.ScalarTransform4f.look_at(origin=init_view_from, target=[0, 0, 0], up=[0, 1, 0]),
                "film": {"type": "hdrfilm", "width": image_size[1], "height": image_size[0], "rfilter": {"type": "box"}},
                "sampler": {"type": "stratified", "sample_count": spp},
            },
        }
        self._env_key = "emitter.data"
        self._bsdf_base_key = None
        self.initialize_scene_dict()
        if self.scene_dict["integrator"]["type"] is None:
            raise Exception(f"{self.__class__.__name__} must specify integrator type in initialize_scene_dict")
        if self.scene_dict["sensor"]["type"] is None:
            raise Exception(f"{self.__class__.__name__} must specify integrator type in initialize_scene_dict")
        if self._bsdf_base_key is None:
            raise Exception(f"{self.__class__.__name__} must specify _bsdf_base_key in initialize_scene_dict")

        self.return_normal = return_normal
        self.return_depth = return_depth

        aovs: Dict[Tuple[str]] = dict()  # aovs[label] = [attribute name, name]
        self.albedo_ch_name: str = ""
        self.normal_ch_name: str = ""
        self.depth_ch_name: str = ""
        if self.return_normal:
            aovs["sh_normal"] = ["normal_ch_name", "normal"]
        if self.return_depth:
            aovs["depth"] = ["depth_ch_name", "depth"]
        self.denoise = denoise
        if denoise:
            assert denoise in ["simple", "informative"], f"{denoise} denoise mode isn't supported"
            if denoise == "simple":
                self._denoiser = mi.OptixDenoiser(image_size)
            elif denoise == "informative":
                self._denoiser = mi.OptixDenoiser(image_size, albedo=True, normals=True)
                aovs["albedo"] = ["albedo_ch_name", "albedo"]
                aovs["sh_normal"] = ["normal_ch_name", "normal"]
            else:
                raise NotImplementedError()

        if len(aovs) > 0:
            if self.scene_dict["integrator"]["type"] != "aov":
                self.scene_dict["integrator"] = {
                    "type": "aov",
                    "aovs": ",".join([f"{aovs[key][1]}:{key}" for key in aovs]),
                    "integrator": self.scene_dict["integrator"],
                }
                for aov_type, (attr_name, aov_name) in aovs.items():
                    setattr(self, attr_name, aov_name)
            else:
                aovs_str = self.scene_dict["integrator"]["aovs"]
                for aov_type, (attr_name, aov_name) in aovs.items():
                    for aov in aovs_str.split(","):
                        name, type_name = aov.split(":")
                        if type_name == aov_type:
                            setattr(self, attr_name, name)
                            break
                    else:
                        while f",{aov_name}:" in aovs_str:
                            aov_name += "_"
                        aovs_str += f",{aov_name}:{aov_type}"
                        setattr(self, attr_name, aov_name)
        self.scene = mi.load_dict(self.scene_dict)
        self.params = mi.traverse(self.scene)
        self.brdf_param_names = brdf_param_names

    def initialize_scene_dict(self):
        # set initial scene dict
        pass

    def _get_results(self, result: mi.TensorXf, sensor: mi.Sensor, channel_first: bool) -> mi.TensorXf:
        noisy_multichannel = sensor.film().bitmap()
        bitmap_dict = dict(noisy_multichannel.split())
        if self.denoise:
            img = mi.TensorXf(
                self._denoiser(
                    noisy_multichannel,
                    albedo_ch=self.albedo_ch_name,
                    normals_ch=self.normal_ch_name,
                    to_sensor=sensor.world_transform().inverse(),
                )
            ).torch()
        else:
            img = mi.TensorXf(bitmap_dict["<root>"]).torch()  # result.torch()
        if channel_first:
            img = img.permute(2, 0, 1)
        if not self.return_normal and not self.return_depth:
            return img
        else:
            outputs = [img]
        if self.return_normal:
            R = sensor.world_transform().matrix.torch()[..., :3, :3]
            # [x, y, z] : mitsuba3 [left, up, forward] -> [right, up, backward]
            R *= torch.tensor([-1, 1, -1], dtype=R.dtype, device=R.device)[:, None] / torch.linalg.det(R) ** (1 / 3)
            normal = mi.TensorXf(bitmap_dict[self.normal_ch_name]).torch() @ R
            if channel_first:
                normal = normal.permute(2, 0, 1)
            outputs.append(normal)
        if self.return_depth:
            depth = mi.TensorXf(bitmap_dict[self.depth_ch_name]).torch()
            depth = depth[None] if channel_first else depth[..., None]
            outputs.append(depth)
        return outputs

    def _render_scene(
        self,
        scene,
        z: torch.Tensor,
        brdf_param_names: list[str],
        params: mi.SceneParameters = None,
        envmap: torch.Tensor = None,
        view_from: torch.Tensor = None,
        channel_first: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if params is None:
            params = mi.traverse(params)
        z = z.cuda()
        brdf_param_names = brdf_param_names or self.brdf_param_names
        if envmap is not None:
            params[self._env_key] = envmap.cuda()
        if view_from is not None:
            view_from = view_from * (1.1 / torch.norm(view_from))
            params["sensor.to_world"] = mi.ScalarTransform4f.look_at(origin=list(view_from), target=[0, 0, 0], up=[0, 1, 0])
        if "base_color.value.R" in brdf_param_names:
            rgb_index = [brdf_param_names.index("base_color.value." + c) for c in "RGB"]
            params[self._bsdf_base_key + ".base_color.value"] = mi.Color3f(z[rgb_index].clip(0, 1)[None])
        for idx, key in enumerate(brdf_param_names):
            if not key.startswith("base_color.value"):
                params[f"{self._bsdf_base_key}.{key}"] = z[idx].clip(0, 1).reshape(1)
        params.update()
        result = mi.render(scene, params, **kwargs)
        sensor: mi.Sensor = scene.sensors()[kwargs["sensor"]] if isinstance(kwargs["sensor"], int) else kwargs["sensor"]
        return self._get_results(result, sensor, channel_first)

    def _render_orig_scene(
        self,
        z: torch.Tensor,
        brdf_param_names: list[str],
        envmap: torch.Tensor = None,
        view_from: torch.Tensor = None,
        channel_first: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        return self._render_scene(
            self.scene,
            z,
            brdf_param_names,
            params=self.params,
            envmap=envmap,
            view_from=view_from,
            channel_first=channel_first,
            **kwargs,
        )

    def _render_new_scene(
        self,
        z: torch.Tensor,
        brdf_param_names: list[str],
        envmap: torch.Tensor,
        view_from: torch.Tensor = None,
        channel_first: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        scene_dict = self.scene_dict.copy()
        scene_dict["emitter"] = {
            "type": "envmap",
            "bitmap": mi.Bitmap(mi.TensorXf(envmap.cuda())),
        }
        if view_from is not None:
            view_from = view_from * (1.1 / torch.norm(view_from))
            scene_dict["sensor"]["to_world"] = mi.ScalarTransform4f.look_at(origin=list(view_from), target=[0, 0, 0], up=[0, 1, 0])

        scene = mi.load_dict(scene_dict)
        params = mi.traverse(scene)
        return self._render_scene(
            scene,
            z,
            brdf_param_names,
            params=params,
            envmap=envmap,
            channel_first=channel_first,
            **kwargs,
        )

    def rendering(
        self,
        z: torch.Tensor,
        brdf_param_names: list[str],
        envmap: torch.Tensor = None,
        view_from: torch.Tensor = None,
        sensor: int | mi.Sensor = 0,
        spp: int = 0,
        new_scene: bool = False,
        channel_first: bool = False,
    ):
        if envmap is not None:
            assert isinstance(envmap, torch.Tensor) and envmap.dim() == 3 and not torch.isnan(envmap[0, 0, 0]), f"envmap [{envmap.shape}]"
        if new_scene:
            return self._render_new_scene(z, brdf_param_names, envmap, view_from, channel_first=channel_first, sensor=sensor, spp=spp)
        else:
            return self._render_orig_scene(z, brdf_param_names, envmap, view_from, channel_first=channel_first, sensor=sensor, spp=spp)


class MitsubaRefMapRenderer(MitsubaBaseRenderer):
    def __init__(
        self,
        refmap_res: int,
        spp: int = 1024,
        envmap_size: tuple = (1000, 2000),
        denoise: str = None,
        return_normal: bool = False,
        return_depth: bool = False,
        init_view_from: list[float] = [0, 0, 1.1],
        brdf_param_names: List[str] = None,
    ) -> None:
        self.refmap_res = refmap_res
        super().__init__(
            (refmap_res, refmap_res),
            spp=spp,
            envmap_size=envmap_size,
            denoise=denoise,
            return_normal=return_normal,
            return_depth=return_depth,
            init_view_from=init_view_from,
            brdf_param_names=brdf_param_names,
        )

    def initialize_scene_dict(self):
        # set initial scene dict
        self.scene_dict["integrator"]["type"] = "direct"
        self.scene_dict["sensor"]["type"] = "refmapsensor"
        self.scene_dict["sphere"] = {
            "type": "sphere",
            "radius": 1.0000001,
            "bsdf": {
                "type": "principled",
                "base_color": {"type": "rgb", "value": [0.0, 0.0, 0.0]},
                "metallic": 0.0,
                "specular": 1.0,
                "roughness": 0.0,
                "spec_tint": 0.0,
                "anisotropic": 0.0,
                "sheen": 0.0,
                "sheen_tint": 0.0,
                "clearcoat": 0.0,
                "clearcoat_gloss": 0.0,
                "spec_trans": 0.0,
            },
        }
        self._bsdf_base_key = "sphere.bsdf"

    def _render_orig_scene(
        self,
        z: torch.Tensor,
        brdf_param_names: list[str],
        envmap: torch.Tensor = None,
        view_from: torch.Tensor = None,
        flip: bool = None,
        channel_first: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if flip is not None:
            self.params["sensor.flip"] = flip
        return super()._render_orig_scene(z, brdf_param_names, envmap=envmap, view_from=view_from, channel_first=channel_first, **kwargs)

    def _render_new_scene(
        self,
        z: torch.Tensor,
        brdf_param_names: list[str],
        envmap: torch.Tensor,
        view_from: torch.Tensor = None,
        flip: bool = None,
        channel_first: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        scene_dict = self.scene_dict.copy()
        scene_dict["emitter"] = {
            "type": "envmap",
            "bitmap": mi.Bitmap(mi.TensorXf(envmap.cuda())),
        }
        if view_from is not None:
            view_from = view_from * (1.1 / torch.norm(view_from))
            scene_dict["sensor"]["to_world"] = mi.ScalarTransform4f.look_at(origin=list(view_from), target=[0, 0, 0], up=[0, 1, 0])
        if flip is not None:
            scene_dict["sensor"]["flip"] = flip
        scene = mi.load_dict(scene_dict)
        params = mi.traverse(scene)
        return self._render_scene(
            scene,
            z,
            brdf_param_names,
            params=params,
            envmap=envmap,
            channel_first=channel_first,
            **kwargs,
        )

    def rendering(
        self,
        z: torch.Tensor,
        brdf_param_names: list[str],
        envmap: torch.Tensor = None,
        view_from: torch.Tensor = None,
        flip: bool = None,
        sensor: int | mi.Sensor = 0,
        spp: int = 0,
        new_scene: bool = False,
        channel_first: bool = False,
    ):
        if envmap is not None:
            assert isinstance(envmap, torch.Tensor) and envmap.dim() == 3 and not torch.isnan(envmap[0, 0, 0]), f"envmap [{envmap.shape}]"
        if new_scene:
            return self._render_new_scene(z, brdf_param_names, envmap, view_from, flip, channel_first=channel_first, sensor=sensor, spp=spp)
        else:
            return self._render_orig_scene(
                z, brdf_param_names, envmap, view_from, flip, channel_first=channel_first, sensor=sensor, spp=spp
            )


class MitsubaOrthoRenderer(MitsubaBaseRenderer):
    def initialize_scene_dict(self):
        # set initial scene dict
        self.scene_dict["integrator"]["type"] = "path"
        self.scene_dict["sensor"]["type"] = "orthographic"
        self.scene_dict["object"] = {
            "type": "obj",
            "filename": "./data/sample.obj",
            "face_normals": False,
            "bsdf": {
                "type": "principled",
                "base_color": {"type": "rgb", "value": [0.0, 0.0, 0.0]},
                "metallic": 0.0,
                "specular": 1.0,
                "roughness": 0.0,
                "spec_tint": 0.0,
                "anisotropic": 0.0,
                "sheen": 0.0,
                "sheen_tint": 0.0,
                "clearcoat": 0.0,
                "clearcoat_gloss": 0.0,
                "spec_trans": 0.0,
            },
        }

        self._bsdf_base_key = "object.bsdf"
        self._object_base_key = "object"

    def _set_obj_params(
        self,
        params: mi.SceneParameters,
        vertex_positions: Union[torch.Tensor, "mi.Float"] = None,
        vertex_normals: Union[torch.Tensor, "mi.Float"] = None,
        faces: Union[torch.Tensor, "mi.UInt"] = None,
    ):
        if vertex_positions is not None:
            if isinstance(vertex_positions, torch.Tensor):
                vertex_count = vertex_positions.size(0)
                vertex_positions = vertex_positions.flatten()
            if isinstance(vertex_positions, mi.Float):
                vertex_count = len(vertex_positions) // 3
            params[f"{self._object_base_key}.vertex_positions"] = vertex_positions
            try:
                params[f"{self._object_base_key}.vertex_count"] = vertex_count
            except KeyError:
                pass
        if faces is not None:
            if isinstance(faces, torch.Tensor):
                face_count = faces.size(0)
                faces = mi.UInt((mi.Int(faces.to(torch.int32).flatten())))
            elif isinstance(faces, mi.UInt):
                face_count = len(faces) // 3
            params[f"{self._object_base_key}.faces"] = faces
            try:
                params[f"{self._object_base_key}.face_count"] = face_count
            except KeyError:
                pass
        if vertex_normals is not None:
            if isinstance(vertex_normals, torch.Tensor):
                vertex_normals = vertex_normals.flatten()
            params[f"{self._object_base_key}.vertex_normals"] = vertex_normals

    def _render_orig_scene(
        self,
        z: torch.Tensor,
        brdf_param_names: list[str],
        envmap: torch.Tensor = None,
        view_from: torch.Tensor = None,
        vertex_positions: torch.Tensor = None,
        vertex_normals: torch.Tensor = None,
        faces: torch.Tensor = None,
        channel_first: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        self._set_obj_params(self.params, vertex_positions, vertex_normals, faces)
        return super()._render_orig_scene(z, brdf_param_names, envmap=envmap, view_from=view_from, channel_first=channel_first, **kwargs)

    def _render_new_scene(
        self,
        z: torch.Tensor,
        brdf_param_names: list[str],
        envmap: torch.Tensor,
        view_from: torch.Tensor = None,
        vertex_positions: torch.Tensor = None,
        vertex_normals: torch.Tensor = None,
        faces: torch.Tensor = None,
        channel_first: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        scene_dict = self.scene_dict.copy()
        scene_dict["emitter"] = {
            "type": "envmap",
            "bitmap": mi.Bitmap(mi.TensorXf(envmap.cuda())),
        }
        if view_from is not None:
            view_from = view_from * (1.1 / torch.norm(view_from))
            scene_dict["sensor"]["to_world"] = mi.ScalarTransform4f.look_at(origin=list(view_from), target=[0, 0, 0], up=[0, 1, 0])
        scene = mi.load_dict(scene_dict)
        params = mi.traverse(scene)
        self._set_obj_params(params, vertex_positions, vertex_normals, faces)
        return self._render_scene(
            scene,
            z,
            brdf_param_names,
            params=params,
            envmap=envmap,
            channel_first=channel_first,
            **kwargs,
        )

    def rendering(
        self,
        z: torch.Tensor,
        brdf_param_names: list[str],
        envmap: torch.Tensor = None,
        view_from: torch.Tensor = None,
        obj: Dict[str, torch.Tensor] = {},
        sensor: int | mi.Sensor = 0,
        spp: int = 0,
        new_scene: bool = False,
        channel_first: bool = False,
    ):
        if envmap is not None:
            assert isinstance(envmap, torch.Tensor) and envmap.dim() == 3 and not torch.isnan(envmap[0, 0, 0]), f"envmap [{envmap.shape}]"
        if new_scene:
            return self._render_new_scene(
                z, brdf_param_names, envmap, view_from, **obj, channel_first=channel_first, sensor=sensor, spp=spp
            )
        else:
            return self._render_orig_scene(
                z, brdf_param_names, envmap, view_from, **obj, channel_first=channel_first, sensor=sensor, spp=spp
            )


def get_bsdf(z: torch.Tensor, brdf_param_names: List[str]) -> "mi.BSDF":
    bsdf_dict = {
        "type": "principled",
        "base_color": {"type": "rgb", "value": [0.0, 0.0, 0.0]},
        "metallic": 0.0,
        "specular": 1.0,
        "roughness": 0.0,
        "spec_tint": 0.0,
        "anisotropic": 0.0,
        "sheen": 0.0,
        "sheen_tint": 0.0,
        "clearcoat": 0.0,
        "clearcoat_gloss": 0.0,
        "spec_trans": 0.0,
    }
    brdf_param_names = brdf_param_names
    if "base_color.value" in brdf_param_names:
        bsdf_dict["base_color"]["value"] = z[brdf_param_names.index("base_color.value")].clip(0, 1).expand(3).tolist()
    elif "base_color.value.R" in brdf_param_names:
        rgb_index = [brdf_param_names.index("base_color.value." + c) for c in "RGB"]
        bsdf_dict["base_color"]["value"] = z[rgb_index].clip(0, 1).tolist()
    for idx, key in enumerate(brdf_param_names):
        if not key.startswith("base_color.value"):
            bsdf_dict[key.split(".")[0]] = z[idx].clip(0, 1).item()
    bsdf: mi.BSDF = mi.load_dict(bsdf_dict)
    return bsdf


def sph_to_dir(theta, phi):
    """Map spherical to Euclidean coordinates"""
    st, ct = dr.sincos(theta)
    sp, cp = dr.sincos(phi)
    return mi.Vector3f(cp * st, sp * st, ct)


def bsdf_to_merl(bsdf: mi.BSDF, color_scale: bool = False) -> np.ndarray:
    # Create a (dummy) surface interaction to use for the evaluation of the BSDF
    si = dr.zeros(mi.SurfaceInteraction3f)
    # Create grid in MERL coodrinates
    BRDFSamplingResThetaH = 90
    BRDFSamplingResThetaD = 90
    BRDFSamplingResPhiD = 360
    theta_h = dr.linspace(mi.Float, 0, 1.0, BRDFSamplingResThetaH, endpoint=False)
    theta_h = theta_h**2.0 * (dr.pi / 2.0)
    theta_d = dr.linspace(mi.Float, 0, dr.pi / 2.0, BRDFSamplingResThetaD, endpoint=False)
    phi_d = dr.linspace(mi.Float, 0, dr.pi, BRDFSamplingResPhiD // 2, endpoint=False)
    theta_h, theta_d, phi_d = dr.meshgrid(theta_h, theta_d, phi_d, indexing="ij")
    # Frame3f based on surface normal
    sth, cth = dr.sincos(theta_h)
    sh_frame = mi.Frame3f(mi.Vector3f(-sth, 0, cth))
    # path outgoing direction on half vector frame
    wo = sph_to_dir(theta_d, phi_d)
    # path incident direction on half vector frame
    wi = wo * mi.Vector3f(-1, -1, 1)
    # path outgoing direction on shading frame
    si.wi = sh_frame.to_local(wo)
    # path incident direction on shading frame
    wo = sh_frame.to_local(wi)
    # Evaluate the whole array (90 * 90 * 180) at once
    brdf_x_cos = bsdf.eval(mi.BSDFContext(), si, wo)
    # BRDF * cos -> BRDF
    cos_theta_out = mi.Frame3f().cos_theta(wo)
    MERL_brdf = brdf_x_cos / cos_theta_out
    # MERL_brdf *= (wi.numpy()[:, 2:] > 0) * (wo.numpy()[:, 2:] > 0)
    if color_scale:
        RED_SCALE = 1.0 / 1500.0
        GREEN_SCALE = 1.15 / 1500.0
        BLUE_SCALE = 1.66 / 1500.0
        MERL_brdf = MERL_brdf / mi.Color3f(RED_SCALE, GREEN_SCALE, BLUE_SCALE)
    MERL_brdf = MERL_brdf.numpy()
    MERL_brdf[np.logical_or(si.wi.numpy()[:, 2] < 0, wo.numpy()[:, 2] < 0)] = -1
    return MERL_brdf.reshape(BRDFSamplingResThetaH, BRDFSamplingResThetaD, BRDFSamplingResPhiD // 2, 3)


def eval_bsdf(
    bsdf: mi.BSDF,
    normal: Union[np.ndarray, torch.Tensor, List],
    wo: Union[np.ndarray, torch.Tensor, List],
    wi: Union[np.ndarray, torch.Tensor, List],
):
    if isinstance(normal, (torch.Tensor, np.ndarray)) and len(normal.shape) > 2:
        raise Exception()
    if isinstance(wo, (torch.Tensor, np.ndarray)) and len(wo.shape) > 2:
        raise Exception()
    if isinstance(wi, (torch.Tensor, np.ndarray)) and len(wi.shape) > 2:
        raise Exception()
    sh_frame = mi.Frame3f(mi.Vector3f(normal))
    wo = sh_frame.to_local(mi.Vector3f(wo))
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wi = sh_frame.to_local(mi.Vector3f(wi))
    return bsdf.eval(mi.BSDFContext(), si, wo)


def visualize_bsdf(bsdf: mi.BSDF, increment_deg: float = 30, imsize: tuple = (512, 512), angle_start: float = 0):
    width, height = imsize
    # Calculate wo, normal map
    angles_deg = range(angle_start, 180, increment_deg)
    wo_map = np.zeros((height, (width * (len(angles_deg) + 1)) // 2, 3), dtype=np.float32)
    normal_map = np.zeros((height, (width * (len(angles_deg) + 1)) // 2, 3), dtype=np.float32)
    mask_map = np.zeros((height, (width * (len(angles_deg) + 1)) // 2), dtype=bool)
    for i, angle_deg in enumerate(angles_deg):
        # Calculate for each sphere
        u, v = np.meshgrid(range(width), range(height))
        x = 2.0 * (u + 0.5) / (imsize[0]) - 1.0
        y = 2.0 * (v + 0.5) / (imsize[0]) - 1.0
        r = np.sqrt(x**2 + y**2)
        mask = r <= 1.0
        z = -np.sqrt(np.clip(1 - r**2, 0, None))
        normal = np.stack([x, y, z], axis=-1)
        offset = (i * width) // 2
        angle = np.radians(angle_deg)
        wo_map[:, offset : offset + width][mask] = np.array([-np.sin(angle), 0.0, -np.cos(angle)])
        normal_map[:, offset : offset + width][mask] = normal[mask]
        mask_map[:, offset : offset + width][mask] = True

    # Evaluate at once
    result = eval_bsdf(bsdf, normal_map[mask_map], wo_map[mask_map], [0, 0, -1])
    # Make a figure
    fig = np.zeros_like(normal_map)
    fig[mask_map] = result
    return fig, mask_map


def load_mesh(path: Path) -> Dict[str, Union["mi.UInt", "mi.Float"]]:
    assert path.suffix in [".obj", ".ply"]
    mesh = mi.load_dict({"type": path.suffix[1:], "filename": str(path)})
    params = mi.traverse(mesh)
    obj_dict = {
        "vertex_positions": params.get("vertex_positions"),
        "vertex_normals": params.get("vertex_normals"),
        "faces": params.get("faces"),
    }
    return obj_dict
