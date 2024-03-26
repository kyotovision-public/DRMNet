import argparse
import datetime
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mitsuba as mi
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from omegaconf import OmegaConf
from packaging import version
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from torch.utils.data import default_collate

from ldm.util import get_obj_from_str, instantiate_from_config
from utils.file_io import save_exr, save_png

mi.set_variant("cuda_ad_rgb")


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=Path,
        const=True,
        default=None,
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        type=Path,
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=Path,
        default=Path("logs"),
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--sampling_postfix",
        type=str,
        default="",
        help="postfix for default sampling result name",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = Path(logdir)
        self.ckptdir = Path(ckptdir)
        self.cfgdir = Path(cfgdir)
        self.config = config
        self.lightning_config = lightning_config

    def on_exception(self, trainer, pl_module, exception):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            print("raised Error:", exception)
            ckpt_path = self.ckptdir / "last_exception.ckpt"
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            self.logdir.mkdir(exist_ok=True)
            self.ckptdir.mkdir(exist_ok=True)
            self.cfgdir.mkdir(exist_ok=True)

            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config, self.cfgdir / f"{self.now}-project.yaml")

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}), self.cfgdir / f"{self.now}-lightning.yaml")

        else:
            while not self.logdir.is_dir():
                time.sleep(0.5)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device)
        torch.cuda.synchronize(trainer.strategy.root_device)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_images,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
        logdir_postfix="",
    ):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.logdir_postfix = logdir_postfix
        if self.logdir_postfix:
            self.logdir_postfix = "_" + self.logdir_postfix

    @rank_zero_only
    def log_local(
        self,
        save_dir,
        split: str,
        images,
        global_rank,
        global_step,
        current_epoch,
        batch_idx,
        other_logs=dict(),
        add_batch_idx: bool = False,
        max_nrow: int = 10,
    ):
        root = Path(save_dir) / "images" / (split + self.logdir_postfix)
        base_filename = f"{current_epoch:03d}_{global_step:06d}_{global_rank:03d}"
        if add_batch_idx:
            base_filename += f"_{batch_idx:03d}"
        for k in images:
            mask_exist = False
            if isinstance(images[k], torch.Tensor):
                image = images[k]
                if image.size(-3) == 4:  # if the image has alpha/mask channel
                    mask_exist = True
                    mask = image[..., -1:, :, :]
                    row_grid_mask = torchvision.utils.make_grid(mask, nrow=max_nrow, pad_value=1.0)[0, :, :, None].numpy()
                    image = image[..., :-1, :, :]
                if self.rescale:
                    image = self.ds.rescale(image)
                grid = torchvision.utils.make_grid(image, nrow=max_nrow)
                grid = grid.permute(1, 2, 0).numpy()
            elif isinstance(images[k], np.ndarray):  # when the image is numpy, it is assumed to be already image shape
                image = images[k]
                if image.shape[-1] == 4:  # if it has alpha/mask channel
                    mask_exist = True
                    mask = image[:, :, -1:]
                    image = image[:, :, :-1]
                grid = images[k]
            else:
                raise NotImplementedError()

            filename = base_filename + f"_{k}"
            path_nosuffix = root / filename

            root.mkdir(exist_ok=True, parents=True)
            ldr = grid.clip(0, 1) ** (1 / 2.2)
            if mask_exist:
                ldr = np.concatenate([ldr, row_grid_mask], axis=-1)
            save_png(path_nosuffix.with_suffix(".png"), ldr)
            if mask_exist:
                grid = np.concatenate([grid, row_grid_mask], axis=-1)
            save_exr(path_nosuffix.with_suffix(".exr"), grid)

        if len(other_logs) > 0:
            root = Path(save_dir) / "other_logs" / (split + self.logdir_postfix)
            root.mkdir(exist_ok=True, parents=True)
            torch.save(other_logs, root / (base_filename + ".pt"))

    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if split == "train":
            check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        else:  # use current epoch when validation
            check_idx = pl_module.current_epoch + 1
        if split == "train" or split == "val":
            if not (
                self.check_frequency(check_idx)
                and hasattr(pl_module, "log_images")  # batch_idx % self.batch_freq == 0
                and callable(pl_module.log_images)
                and self.max_images > 0
            ):
                return

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            images = pl_module.log_images(batch, split=split, N=self.max_images, **self.log_images_kwargs)

        if isinstance(images, tuple):
            images, other_logs = images[0], images[1]
        else:
            other_logs = dict()
            assert isinstance(images, dict)

        for k in images:
            assert not isinstance(images[k], list), f"images[{k}] is list"
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
        for k in other_logs:
            if isinstance(other_logs[k], torch.Tensor):
                other_logs[k] = other_logs[k].detach().cpu()

        self.log_local(
            pl_module.logger.save_dir,
            split,
            images,
            pl_module.global_rank,
            pl_module.global_step,
            pl_module.current_epoch,
            batch_idx,
            other_logs,
        )

        if is_train:
            pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (check_idx > 0 or self.log_first_step):
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.ds = getattr(trainer.datamodule, f"train_ds")
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self.last_val_batch = batch
        self.last_val_batch_idx = batch_idx

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.ds = getattr(trainer.datamodule, f"val_ds")
        if not self.disabled and pl_module.global_step >= 0:
            self.log_img(pl_module, self.last_val_batch, self.last_val_batch_idx, split="val")

    def on_test_epoch_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        self.ds = getattr(trainer.datamodule, f"test_ds")


class CustomRandomSampler(torch.utils.data.Sampler):
    def __init__(self, ds, generator: Optional[torch.Generator] = None, same_sampling: bool = False):
        self.ds = ds
        self.num_samples = len(ds)
        self.generator = generator
        self.same_sampling = same_sampling
        if self.same_sampling:
            if self.generator is None:
                self.generator = torch.Generator(torch.empty((), dtype=torch.int64).random_().item())
            self.g_state = self.generator.get_state()

    def __iter__(self):
        if self.same_sampling:
            self.generator.set_state(self.g_state)
        yield from iter(torch.randperm(self.num_samples, generator=self.generator).tolist())

    def __len__(self):
        return self.num_samples


# don't stack mesh data
def my_collate_fn(batch: List[Dict]):
    obj_shape = [item.pop("obj_shape", None) for item in batch]
    batch = default_collate(batch)
    if not all(r is None for r in obj_shape):
        batch["obj_shape"] = obj_shape
    return batch


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        predict=None,
        num_workers=None,
        train_sampler=None,
        val_sampler=None,
        test_sampler=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = {}
        self.num_workers = num_workers if num_workers is not None else batch_size
        if train is not None:
            self.dataset_configs["train"] = train
            train_ds = instantiate_from_config(self.dataset_configs["train"])
            generator = torch.Generator()
            generator.manual_seed(10)
            if train_sampler is not None:
                self.train_sampler = get_obj_from_str(train_sampler["target"])(train_ds, generator, **train_sampler.get("params", dict()))
            else:
                self.train_sampler = None
            self.train_ds = train_ds
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            val_ds = instantiate_from_config(self.dataset_configs["validation"])
            generator = torch.Generator()
            generator.manual_seed(5)
            if val_sampler is not None:
                self.val_sampler = get_obj_from_str(val_sampler["target"])(val_ds, generator, **val_sampler.get("params", dict()))
            else:
                self.val_sampler = None
            self.val_ds = val_ds
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            test_ds = instantiate_from_config(self.dataset_configs["test"])
            generator = torch.Generator()
            generator.manual_seed(7)
            if test_sampler is not None:
                self.test_sampler = get_obj_from_str(test_sampler["target"])(test_ds, generator, **test_sampler.get("params", dict()))
            else:
                self.test_sampler = None
            self.test_ds = test_ds
            self.test_dataloader = self._test_dataloader
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_ds = instantiate_from_config(self.dataset_configs["predict"])
            self.predict_dataloader = self._predict_dataloader

    def state_dict(self):
        state = {}
        for k in ["train", "val"]:
            if hasattr(self, f"{k}_sampler") and self.train_sampler is not None:
                state["{k}_sampler_generator_state"] = self.train_sampler.generator.get_state()
            if hasattr(self, f"{k}_ds") and hasattr(self.train_ds, "generator"):
                state[f"{k}_ds_generator_state"] = self.train_ds.generator.get_state()
        return state

    def load_state_dict(self, state_dict) -> None:
        for k in ["train", "val"]:
            if hasattr(self, f"{k}_sampler") and f"{k}_sampler_generator_state" in state_dict:
                self.train_sampler.generator.set_state(state_dict.get(f"{k}_sampler_generator_state"))
            if hasattr(self, f"{k}_ds") and hasattr(self.train_ds, "generator") and f"{k}_ds_generator_state" in state_dict:
                self.train_ds.generator.set_state(state_dict.get(f"{k}_ds_generator_state"))

    def _train_dataloader(self):
        if hasattr(self.train_ds, "current_epoch"):
            self.train_ds.set_current_epoch(self.trainer.current_epoch)
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=self.train_sampler is None,
            num_workers=self.num_workers,
            sampler=self.train_sampler,
            collate_fn=my_collate_fn,
        )

    def _val_dataloader(self):
        if hasattr(self.val_ds, "current_epoch"):
            self.val_ds.set_current_epoch(self.trainer.current_epoch)
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, sampler=self.val_sampler, collate_fn=my_collate_fn
        )

    def _test_dataloader(self):
        if hasattr(self.test_ds, "current_epoch"):
            self.test_ds.set_current_epoch(self.trainer.current_epoch)
        return torch.utils.data.DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, sampler=self.test_sampler, collate_fn=my_collate_fn
        )

    def _predict_dataloader(self):
        if hasattr(self.predict_ds, "current_epoch"):
            self.predict_ds.set_current_epoch(self.trainer.current_epoch)
        return torch.utils.data.DataLoader(
            self.predict_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=my_collate_fn
        )


# give current epoch to dataset
class DataModuleCallback(Callback):
    def _set_current_epoch(self, ds, current_epoch):
        if hasattr(ds, "current_epoch"):
            ds.set_current_epoch(current_epoch)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        self._set_current_epoch(trainer.datamodule.train_ds, trainer.current_epoch)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        self._set_current_epoch(trainer.datamodule.val_ds, trainer.current_epoch)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        self._set_current_epoch(trainer.datamodule.test_ds, trainer.current_epoch)

    def on_predict_epoch_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        self._set_current_epoch(trainer.datamodule.predict_ds, trainer.current_epoch)


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    resume: Path = opt.resume
    if opt.name and resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if resume:
        if not resume.exists():
            raise ValueError("Cannot find {}".format(resume))
        if resume.is_file():
            logdir = resume.parent.parent
            ckpt = resume
        else:
            assert resume.is_dir(), resume
            logdir = resume
            ckpt = logdir / "checkpoints" / "last.ckpt"

        opt.resume_from_checkpoint = str(ckpt)
        base_configs = sorted(logdir.glob("configs/*.yaml"))
        opt.base = base_configs + opt.base  # List[Path]
        nowname = logdir.name
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_name = opt.base[0].stem
            name = "_" + cfg_name
        else:
            name = ""
        nowname: str = now + name + opt.postfix
        logdir: Path = opt.logdir / nowname

    logdir.mkdir(exist_ok=True, parents=True)
    ckptdir = logdir / "checkpoints"
    cfgdir = logdir / "configs"
    seed_everything(opt.seed)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    config = OmegaConf.merge(*configs)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_config["strategy"] = "ddp"
    trainer_config["accelerator"] = "gpu"
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "devices" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["devices"].strip(",")
        trainer_config["devices"] = gpuinfo + ","
        print(f"Running on GPUs {gpuinfo}")
        cpu = False

    # no sanity check when resuming
    if opt.resume:
        trainer_config["num_sanity_val_steps"] = 0
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model: pl.LightningModule = instantiate_from_config(config.model)

    # trainer and callbacks
    trainer_kwargs = dict()

    # default logger
    trainer_kwargs["logger"] = TensorBoardLogger(logdir, name="tensorboard", log_graph=False)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": str(ckptdir),
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
        },
    }
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 3

    modelckpt_cfg = getattr(lightning_config, "modelcheckpoint", OmegaConf.create())
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    print(f"Merged modelckpt-cfg: \n{OmegaConf.to_yaml(modelckpt_cfg)}")
    if version.parse(pl.__version__) < version.parse("1.4.0"):
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "main.SetupCallback",
            "params": {
                "resume": str(opt.resume),
                "now": now,
                "logdir": str(logdir),
                "ckptdir": str(ckptdir),
                "cfgdir": str(cfgdir),
                "config": config,
                "lightning_config": lightning_config,
            },
        },
        "image_logger": {
            "target": "main.ImageLogger",
            "params": {"batch_frequency": 750, "max_images": 4, "clamp": True},
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                # "log_momentum": True
            },
        },
        "cuda_callback": {"target": "main.CUDACallback"},
        "datamodule_callback": {"target": "main.DataModuleCallback"},
    }
    if version.parse(pl.__version__) >= version.parse("1.4.0"):
        default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if "ignore_keys_callback" in callbacks_cfg and hasattr(trainer_opt, "resume_from_checkpoint"):
        callbacks_cfg.ignore_keys_callback.params["ckpt_path"] = trainer_opt.resume_from_checkpoint
    elif "ignore_keys_callback" in callbacks_cfg:
        del callbacks_cfg["ignore_keys_callback"]

    if callbacks_cfg["image_logger"]["target"] == "main.ImageLogger":
        callbacks_cfg["image_logger"]["params"]["logdir_postfix"] = opt.sampling_postfix

    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    trainer: Trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir  ###

    # data
    print("#### Data #####")
    data: DataModuleFromConfig = instantiate_from_config(config.data)
    for split in ["train", "val", "test", "predict"]:
        if ds := getattr(data, f"{split}_ds", None):
            # give the model to datasets to they can use it.
            ds.model = model
            print(f"{split}, {ds.__class__.__name__}, {len(ds)}")

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    ngpu = len(lightning_config.trainer.devices.strip(",").split(",")) if not cpu else 1
    accumulate_grad_batches = getattr(lightning_config.trainer, "accumulate_grad_batches", 1)
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_devices) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
            )
        )
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")

    # run
    if opt.train:
        trainer.fit(model, data)
    else:
        if opt.resume:
            model.init_from_ckpt(opt.resume_from_checkpoint)
        trainer.validate(model, data)
