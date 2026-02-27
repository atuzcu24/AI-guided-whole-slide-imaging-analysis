# -*- coding: utf-8 -*-
# CellVit Experiment Class for PanNuke
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import copy
import datetime
import os
import shutil
import uuid
from pathlib import Path
from typing import Callable, Tuple, Union

import albumentations as A
import torch
import torch.nn as nn
import wandb

os.environ["WANDB__SERVICE_WAIT"] = "300"

import yaml
from cellvit.models.cell_segmentation.cellvit import CellViT
from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from cellvit.models.cell_segmentation.cellvit_uni import CellViTUNI
from cellvit.models.cell_segmentation.cellvit_virchow import CellViTVirchow
from cellvit.models.cell_segmentation.cellvit_virchow2 import CellViTVirchow2
from cellvit.models.cell_segmentation.cellvit_sam_rosie_film import CellViTSAMRosieFiLM #Fusion
from cellvit.models.cell_segmentation.cellvit_sam_rosie_early_fusion import CellViTSAMRosieEarlyFusion
from cellvit.models.cell_segmentation.cellvit_virchow_rosie_film import CellViTVirchowRosieFiLM #Fusion
from cellvit.training.base_ml.base_early_stopping import EarlyStopping
from cellvit.training.base_ml.base_experiment import BaseExperiment
from cellvit.training.base_ml.base_optim import OPTI_DICT
from cellvit.training.base_ml.base_loss import retrieve_loss_fn
from cellvit.training.base_ml.base_trainer import BaseTrainer
from cellvit.training.datasets.base_cell_dataset import CellDataset
from cellvit.training.datasets.dataset_coordinator import select_dataset
from cellvit.training.trainer.trainer_cellvit import CellViTTrainer
from cellvit.utils.tools import close_logger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    ExponentialLR,
    SequentialLR,
    _LRScheduler,
)
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    Subset,
    WeightedRandomSampler,
)
from torchinfo import summary
from wandb.sdk.lib.runid import generate_id
import cv2


class ExperimentCellVitPanNuke(BaseExperiment):
    def __init__(
        self, default_conf: dict, checkpoint=None, just_load_model=False
    ) -> None:
        super().__init__(default_conf, checkpoint, just_load_model)
        self.load_dataset_setup(dataset_path=self.default_conf["data"]["dataset_path"])

    def run_experiment(self) -> tuple[Path, dict, nn.Module, dict]:
        """Main Experiment Code"""
        ### Setup
        # close loggers
        self.close_remaining_logger()

        # get the config for the current run
        self.run_conf = copy.deepcopy(self.default_conf)
        self.run_conf["dataset_config"] = self.dataset_config
        self.run_name = f"{datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')}_{self.run_conf['logging']['log_comment']}"

        wandb_run_id = generate_id()
        resume = None
        if self.checkpoint is not None and not self.just_load_model:
            wandb_run_id = self.checkpoint["wandb_id"]
            resume = "must"
            self.run_name = self.checkpoint["run_name"]

        # initialize wandb
        run = wandb.init(
            project=self.run_conf["logging"]["project"],
            tags=self.run_conf["logging"].get("tags", []),
            name=self.run_name,
            notes=self.run_conf["logging"]["notes"],
            dir=self.run_conf["logging"]["wandb_dir"],
            mode=self.run_conf["logging"]["mode"].lower(),
            group=self.run_conf["logging"].get("group", str(uuid.uuid4())),
            allow_val_change=True,
            id=wandb_run_id,
            resume=resume,
            settings=wandb.Settings(start_method="fork"),
        )

        # get ids
        self.run_conf["logging"]["run_id"] = run.id
        self.run_conf["logging"]["wandb_file"] = run.id

        # overwrite configuration with sweep values are leave them as they are
        if self.run_conf["run_sweep"] is True:
            self.run_conf["logging"]["sweep_id"] = run.sweep_id
            self.run_conf["logging"]["log_dir"] = str(
                Path(self.default_conf["logging"]["log_dir"])
                / f"sweep_{run.sweep_id}"
                / f"{self.run_name}_{self.run_conf['logging']['run_id']}"
            )
            self.overwrite_sweep_values(self.run_conf, run.config)
        else:
            self.run_conf["logging"]["log_dir"] = str(
                Path(self.default_conf["logging"]["log_dir"]) / self.run_name
            )

        # update wandb
        wandb.config.update(
            self.run_conf, allow_val_change=True
        )  # this may lead to the problem

        # create output folder, instantiate logger and store config
        self.create_output_dir(self.run_conf["logging"]["log_dir"])
        self.logger = self.instantiate_logger()
        self.logger.info("Instantiated Logger. WandB init and config update finished.")
        self.logger.info(f"Run ist stored here: {self.run_conf['logging']['log_dir']}")
        self.store_config()

        self.logger.info(
            f"Cuda devices: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}"
        )
        ### Machine Learning
        device = f"cuda:{self.run_conf['gpu']}"
        self.logger.info(f"Using GPU: {device}")
        self.logger.info(f"Using device: {device}")

        # loss functions
        loss_fn_dict = self.get_loss_fn(self.run_conf.get("loss", {}))
        self.logger.info("Loss functions:")
        self.logger.info(loss_fn_dict)

        # model
        model = self.get_train_model(
            pretrained_encoder=self.run_conf["model"].get("pretrained_encoder", None),
            pretrained_model=self.run_conf["model"].get("pretrained", None),
            backbone_type=self.run_conf["model"].get("backbone", "default"),
            regression_loss=self.run_conf["model"].get("regression_loss", False),
        )
        model.to(device)

        # Optional: enable FiLM gamma/beta stats logging for Virchow Rosie FiLM
        bb = self.run_conf.get("model", {}).get("backbone", "").lower()
        fusion_cfg = self.run_conf.get("fusion", {})
        log_film_every = fusion_cfg.get("log_film_stats_every", 0)
        if bb == "virchow-rosie-film" and log_film_every > 0 and hasattr(model, "film_blocks"):
            for block in model.film_blocks.values():
                block.log_film_stats = True
            self.logger.info(f"FiLM stats logging enabled every {log_film_every} steps")

        # Optional: one-time print of trainable params (guarded by debug.print_trainables)
        if self.run_conf.get("debug", {}).get("print_trainables", False):
            trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
            total = sum(c for _, c in trainable)
            names = [n for n, _ in trainable]
            self.logger.info(f"[debug] Trainable params: {total} across {len(names)} modules")
            self.logger.info(f"[debug] Trainable module names: {names[:20]}{'...' if len(names) > 20 else ''}")

        # optimizer
        optimizer = self.get_optimizer(
            model,
            self.run_conf["training"]["optimizer"],
            self.run_conf["training"]["optimizer_hyperparameter"],
        )

        # scheduler
        scheduler = self.get_scheduler(
            optimizer=optimizer,
            scheduler_type=self.run_conf["training"]["scheduler"]["scheduler_type"],
        )

        # early stopping (no early stopping for basic setup)
        early_stopping = None
        if "early_stopping_patience" in self.run_conf["training"]:
            if self.run_conf["training"]["early_stopping_patience"] is not None:
                early_stopping = EarlyStopping(
                    patience=self.run_conf["training"]["early_stopping_patience"],
                    strategy="maximize",
                )

        ### Data handling
        train_transforms, val_transforms = self.get_transforms(
            self.run_conf["transformations"],
            input_shape=self.run_conf["data"].get("input_shape", 256),
        )

        train_dataset, val_dataset = self.get_datasets(
            train_transforms=train_transforms,
            val_transforms=val_transforms,
        )

        # load sampler
        training_sampler = self.get_sampler(
            train_dataset=train_dataset,
            strategy=self.run_conf["training"].get("sampling_strategy", "random"),
            gamma=self.run_conf["training"].get("sampling_gamma", 1),
        )

        # define dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.run_conf["training"]["batch_size"],
            sampler=training_sampler,
            num_workers=16,
            pin_memory=False,
            worker_init_fn=self.seed_worker,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=128,
            num_workers=16,
            pin_memory=True,
            worker_init_fn=self.seed_worker,
        )

        # start Training
        self.logger.info("Instantiate Trainer")
        trainer_fn = self.get_trainer()
        trainer = trainer_fn(
            model=model,
            loss_fn_dict=loss_fn_dict,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=self.logger,
            logdir=self.run_conf["logging"]["log_dir"],
            num_classes=self.run_conf["data"]["num_nuclei_classes"],
            dataset_config=self.dataset_config,
            early_stopping=early_stopping,
            experiment_config=self.run_conf,
            log_images=self.run_conf["logging"].get("log_images", False),
            magnification=self.run_conf["data"].get("magnification", 40),
            mixed_precision=self.run_conf["training"].get("mixed_precision", False),
        )

        # Load checkpoint if provided
        if self.checkpoint is not None:
            self.logger.info("Checkpoint was provided. Restore ...")
            trainer.resume_checkpoint(self.checkpoint, self.just_load_model)

        # Call fit method
        self.logger.info("Calling Trainer Fit")
        unfreeze_epoch = self.run_conf["training"]["unfreeze_epoch"]
        bb = self.run_conf.get("model", {}).get("backbone", "").lower()
        fusion_cfg = self.run_conf.get("fusion", {})
        if bb in ("sam-h-rosie-film", "sam-h-proxy-film") and fusion_cfg.get("freeze_cellvit", True):
            uce = fusion_cfg.get("unfreeze_cellvit_epoch", 0)
            # Use fusion.unfreeze_cellvit_epoch when set; else training.unfreeze_epoch
            unfreeze_epoch = uce if uce > 0 else unfreeze_epoch
        trainer.fit(
            epochs=self.run_conf["training"]["epochs"],
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            metric_init=self.get_wandb_init_dict(),
            unfreeze_epoch=unfreeze_epoch,
            eval_every=self.run_conf["training"].get("eval_every", 1),
        )

        # --- Always save latest checkpoint safely, even if eval_every is large ---
        import os
        ckpt_dir = os.path.join(self.run_conf["logging"]["log_dir"], "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "latest_checkpoint.pth")

        torch.save({
            "epoch": self.run_conf["training"]["epochs"],
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler else None,
        }, ckpt_path)
        print(f"✅ Saved latest checkpoint → {ckpt_path}")

        # Select best model if not provided by early stopping
        checkpoint_dir = Path(self.run_conf["logging"]["log_dir"]) / "checkpoints"
        if not (checkpoint_dir / "model_best.pth").is_file():
            shutil.copy(
                checkpoint_dir / "latest_checkpoint.pth",
                checkpoint_dir / "model_best.pth",
            )

        # At the end close logger
        self.logger.info(f"Finished run {run.id}")
        close_logger(self.logger)

        return self.run_conf["logging"]["log_dir"]

    def load_dataset_setup(self, dataset_path: Union[Path, str]) -> None:
        """Load the configuration of the cell segmentation dataset.

        The dataset must have a dataset_config.yaml file in their dataset path with the following entries:
            * tissue_types: describing the present tissue types with corresponding integer
            * nuclei_types: describing the present nuclei types with corresponding integer

        Args:
            dataset_path (Union[Path, str]): Path to dataset folder
        """
        dataset_config_path = Path(dataset_path) / "dataset_config.yaml"
        with open(dataset_config_path, "r") as dataset_config_file:
            yaml_config = yaml.safe_load(dataset_config_file)
            self.dataset_config = dict(yaml_config)

    def get_loss_fn(self, loss_fn_settings: dict) -> dict:
        """Create a dictionary with loss functions for all branches

        Branches: "nuclei_binary_map", "hv_map", "nuclei_type_map", "tissue_types"

        Args:
            loss_fn_settings (dict): Dictionary with the loss function settings. Structure
            branch_name(str):
                loss_name(str):
                    loss_fn(str): String matching to the loss functions defined in the LOSS_DICT (base_ml.base_loss)
                    weight(float): Weighting factor as float value
                    (optional) args:  Optional parameters for initializing the loss function
                            arg_name: value

            If a branch is not provided, the defaults settings (described below) are used.

            For further information, please have a look at the file configs/examples/cell_segmentation/train_cellvit.yaml
            under the section "loss"

            Example:
                  nuclei_binary_map:
                    bce:
                        loss_fn: xentropy_loss
                        weight: 1
                    dice:
                        loss_fn: dice_loss
                        weight: 1

        Returns:
            dict: Dictionary with loss functions for each branch. Structure:
                branch_name(str):
                    loss_name(str):
                        "loss_fn": Callable loss function
                        "weight": weight of the loss since in the end all losses of all branches are added together for backward pass
                    loss_name(str):
                        "loss_fn": Callable loss function
                        "weight": weight of the loss since in the end all losses of all branches are added together for backward pass
                branch_name(str)
                ...

        Default loss dictionary:
            nuclei_binary_map:
                bce:
                    loss_fn: xentropy_loss
                    weight: 1
                dice:
                    loss_fn: dice_loss
                    weight: 1
            hv_map:
                mse:
                    loss_fn: mse_loss_maps
                    weight: 1
                msge:
                    loss_fn: msge_loss_maps
                    weight: 1
            nuclei_type_map
                bce:
                    loss_fn: xentropy_loss
                    weight: 1
                dice:
                    loss_fn: dice_loss
                    weight: 1
            tissue_types
                ce:
                    loss_fn: nn.CrossEntropyLoss()
                    weight: 1
        """
        loss_fn_dict = {}
        if "nuclei_binary_map" in loss_fn_settings.keys():
            loss_fn_dict["nuclei_binary_map"] = {}
            for loss_name, loss_sett in loss_fn_settings["nuclei_binary_map"].items():
                parameters = loss_sett.get("args", {})
                loss_fn_dict["nuclei_binary_map"][loss_name] = {
                    "loss_fn": retrieve_loss_fn(loss_sett["loss_fn"], **parameters),
                    "weight": loss_sett["weight"],
                }
        else:
            loss_fn_dict["nuclei_binary_map"] = {
                "bce": {"loss_fn": retrieve_loss_fn("xentropy_loss"), "weight": 1},
                "dice": {"loss_fn": retrieve_loss_fn("dice_loss"), "weight": 1},
            }
        if "hv_map" in loss_fn_settings.keys():
            loss_fn_dict["hv_map"] = {}
            for loss_name, loss_sett in loss_fn_settings["hv_map"].items():
                parameters = loss_sett.get("args", {})
                loss_fn_dict["hv_map"][loss_name] = {
                    "loss_fn": retrieve_loss_fn(loss_sett["loss_fn"], **parameters),
                    "weight": loss_sett["weight"],
                }
        else:
            loss_fn_dict["hv_map"] = {
                "mse": {"loss_fn": retrieve_loss_fn("mse_loss_maps"), "weight": 1},
                "msge": {"loss_fn": retrieve_loss_fn("msge_loss_maps"), "weight": 1},
            }
        if "nuclei_type_map" in loss_fn_settings.keys():
            loss_fn_dict["nuclei_type_map"] = {}
            for loss_name, loss_sett in loss_fn_settings["nuclei_type_map"].items():
                parameters = loss_sett.get("args", {})
                loss_fn_dict["nuclei_type_map"][loss_name] = {
                    "loss_fn": retrieve_loss_fn(loss_sett["loss_fn"], **parameters),
                    "weight": loss_sett["weight"],
                }
        else:
            loss_fn_dict["nuclei_type_map"] = {
                "bce": {"loss_fn": retrieve_loss_fn("xentropy_loss"), "weight": 1},
                "dice": {"loss_fn": retrieve_loss_fn("dice_loss"), "weight": 1},
            }
        if "tissue_types" in loss_fn_settings.keys():
            loss_fn_dict["tissue_types"] = {}
            for loss_name, loss_sett in loss_fn_settings["tissue_types"].items():
                parameters = loss_sett.get("args", {})
                loss_fn_dict["tissue_types"][loss_name] = {
                    "loss_fn": retrieve_loss_fn(loss_sett["loss_fn"], **parameters),
                    "weight": loss_sett["weight"],
                }
        else:
            loss_fn_dict["tissue_types"] = {
                "ce": {"loss_fn": nn.CrossEntropyLoss(), "weight": 1},
            }
        if "regression_loss" in loss_fn_settings.keys():
            loss_fn_dict["regression_map"] = {}
            for loss_name, loss_sett in loss_fn_settings["regression_loss"].items():
                parameters = loss_sett.get("args", {})
                loss_fn_dict["regression_map"][loss_name] = {
                    "loss_fn": retrieve_loss_fn(loss_sett["loss_fn"], **parameters),
                    "weight": loss_sett["weight"],
                }
        elif "regression_loss" in self.run_conf["model"].keys():
            loss_fn_dict["regression_map"] = {
                "mse": {"loss_fn": retrieve_loss_fn("mse_loss_maps"), "weight": 1},
            }
        return loss_fn_dict

    def get_optimizer(
        self, model: nn.Module, optimizer_name: str, hp: dict
    ) -> Optimizer:
        """Optimizer with optional FiLM-only LR scaling for sam-h-rosie-film."""
        bb = self.run_conf.get("model", {}).get("backbone", "").lower()
        fusion_cfg = self.run_conf.get("fusion", {})
        film_lr_mult = fusion_cfg.get("film_lr_mult", 1.0)

        if bb not in ("sam-h-rosie-film", "virchow-rosie-film") or film_lr_mult == 1.0:
            return super().get_optimizer(model, optimizer_name, hp)

        if optimizer_name not in OPTI_DICT:
            raise NotImplementedError("Optimizer not known")

        base_lr = hp.get("lr", 3e-5)
        weight_decay = hp.get("weight_decay", 0.0)
        film_params = []
        base_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "film_blocks." in name:
                film_params.append(p)
            else:
                base_params.append(p)

        param_groups = [
            {"params": base_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": film_params, "lr": base_lr * film_lr_mult, "weight_decay": weight_decay},
        ]
        hp_copy = {k: v for k, v in hp.items() if k not in ("lr", "weight_decay")}
        optimizer = OPTI_DICT[optimizer_name](param_groups, **hp_copy)

        self.logger.info(
            f"Loaded {optimizer_name} with film_lr_mult={film_lr_mult} | "
            f"base: {len(base_params)} params @ lr={base_lr} | "
            f"FiLM: {len(film_params)} params @ lr={base_lr * film_lr_mult}"
        )
        return optimizer

    def get_scheduler(self, scheduler_type: str, optimizer: Optimizer) -> _LRScheduler:
        """Get the learning rate scheduler for CellViT

        The configuration of the scheduler is given in the "training" -> "scheduler" section.
        Currenlty, "constant", "exponential" and "cosine" schedulers are implemented.

        Required parameters for implemented schedulers:
            - "constant": None
            - "exponential": gamma (optional, defaults to 0.95)
            - "cosine": eta_min (optional, defaults to 1-e5)

        Args:
            scheduler_type (str): Type of scheduler as a string. Currently implemented:
                - "constant" (lowering by a factor of ten after 25 epochs, increasing after 50, decreasimg again after 75)
                - "exponential" (ExponentialLR with given gamma, gamma defaults to 0.95)
                - "cosine" (CosineAnnealingLR, eta_min as parameter, defaults to 1-e5)
            optimizer (Optimizer): Optimizer

        Returns:
            _LRScheduler: PyTorch Scheduler
        """
        implemented_schedulers = ["constant", "exponential", "cosine"]
        if scheduler_type.lower() not in implemented_schedulers:
            self.logger.warning(
                f"Unknown Scheduler - No scheduler from the list {implemented_schedulers} select. Using default scheduling."
            )
        if scheduler_type.lower() == "constant":
            scheduler = SequentialLR(
                optimizer=optimizer,
                schedulers=[
                    ConstantLR(optimizer, factor=1, total_iters=25),
                    ConstantLR(optimizer, factor=0.1, total_iters=25),
                    ConstantLR(optimizer, factor=1, total_iters=25),
                    ConstantLR(optimizer, factor=0.1, total_iters=1000),
                ],
                milestones=[24, 49, 74],
            )
        elif scheduler_type.lower() == "exponential":
            scheduler = ExponentialLR(
                optimizer,
                gamma=self.run_conf["training"]["scheduler"].get("gamma", 0.95),
            )
        elif scheduler_type.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.run_conf["training"]["epochs"],
                eta_min=self.run_conf["training"]["scheduler"].get("eta_min", 1e-5),
            )
        else:
            scheduler = super().get_scheduler(optimizer)
        return scheduler

    def get_datasets(
        self,
        train_transforms: Callable = None,
        val_transforms: Callable = None,
    ) -> Tuple[Dataset, Dataset]:
        """Retrieve training dataset and validation dataset

        Args:
            train_transforms (Callable, optional): PyTorch transformations for train set. Defaults to None.
            val_transforms (Callable, optional): PyTorch transformations for validation set. Defaults to None.

        Returns:
            Tuple[Dataset, Dataset]: Training dataset and validation dataset
        """
        if (
            "val_split" in self.run_conf["data"]
            and "val_folds" in self.run_conf["data"]
        ):
            raise RuntimeError(
                "Provide either val_splits or val_folds in configuration file, not both."
            )
        if (
            "val_split" not in self.run_conf["data"]
            and "val_folds" not in self.run_conf["data"]
        ):
            raise RuntimeError(
                "Provide either val_split or val_folds in configuration file, one is necessary."
            )
        if (
            "val_split" not in self.run_conf["data"]
            and "val_folds" not in self.run_conf["data"]
        ):
            raise RuntimeError(
                "Provide either val_split or val_fold in configuration file, one is necessary."
            )
        if "regression_loss" in self.run_conf["model"].keys():
            self.run_conf["data"]["regression_loss"] = True

        full_dataset = select_dataset(
            dataset_name="pannuke",
            split="train",
            dataset_config=self.run_conf["data"],
            transforms=train_transforms,
        )
        if "val_split" in self.run_conf["data"]:
            generator_split = torch.Generator().manual_seed(
                self.default_conf["random_seed"]
            )
            val_splits = float(self.run_conf["data"]["val_split"])
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset,
                lengths=[1 - val_splits, val_splits],
                generator=generator_split,
            )
            val_dataset.dataset = copy.deepcopy(full_dataset)
            val_dataset.dataset.set_transforms(val_transforms)
        else:
            train_dataset = full_dataset
            val_dataset = select_dataset(
                dataset_name="pannuke",
                split="validation",
                dataset_config=self.run_conf["data"],
                transforms=val_transforms,
            )

        return train_dataset, val_dataset

    def get_train_model(
        self,
        pretrained_encoder: Union[Path, str] = None,
        pretrained_model: Union[Path, str] = None,
        backbone_type: str = "default",
        regression_loss: bool = False,
        **kwargs,
    ) -> CellViT:
        """Return the CellViT training model

        Args:
            pretrained_encoder (Union[Path, str]): Path to a pretrained encoder. Defaults to None.
            pretrained_model (Union[Path, str], optional): Path to a pretrained model. Defaults to None.
            backbone_type (str, optional): Backbone Type. Currently supported are default (None, ViT256, SAM-B, SAM-L, SAM-H). Defaults to None
            regression_loss (bool, optional): If regression loss is used. Defaults to False

        Returns:
            CellViT: CellViT training model with given setup
        """
        # reseed needed, due to subprocess seeding compatibility
        self.seed_run(self.default_conf["random_seed"])

        # check for backbones
        implemented_backbones = [
            "default",
            "vit256",
            "sam-b",
            "sam-l",
            "sam-h",
            "sam-h-rosie-film",
            "sam-h-proxy-film",
            "sam-h-rosie-earlyfusion-vec",
            "sam-h-rosie-earlyfusion-mapc",
            "uni",
            "virchow",
            "virchow2",
            "virchow-rosie-film",
        ]
        if backbone_type.lower() not in implemented_backbones:
            raise NotImplementedError(
                f"Unknown Backbone Type - Currently supported are: {implemented_backbones}"
            )
        if backbone_type.lower() == "default":
            model = CellViT(
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                embed_dim=self.run_conf["model"]["embed_dim"],
                input_channels=self.run_conf["model"].get("input_channels", 3),
                depth=self.run_conf["model"]["depth"],
                num_heads=self.run_conf["model"]["num_heads"],
                extract_layers=self.run_conf["model"]["extract_layers"],
                drop_rate=self.run_conf["training"].get("drop_rate", 0),
                attn_drop_rate=self.run_conf["training"].get("attn_drop_rate", 0),
                drop_path_rate=self.run_conf["training"].get("drop_path_rate", 0),
                regression_loss=regression_loss,
            )

            if pretrained_model is not None:
                self.logger.info(
                    f"Loading pretrained CellViT model from path: {pretrained_model}"
                )
                cellvit_pretrained = torch.load(pretrained_model)
                self.logger.info(model.load_state_dict(cellvit_pretrained, strict=True))
                self.logger.info("Loaded CellViT model")

        if backbone_type.lower() == "vit256":
            model = CellViT256(
                model256_path=pretrained_encoder,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                drop_rate=self.run_conf["training"].get("drop_rate", 0),
                attn_drop_rate=self.run_conf["training"].get("attn_drop_rate", 0),
                drop_path_rate=self.run_conf["training"].get("drop_path_rate", 0),
                regression_loss=regression_loss,
            )
            model.load_pretrained_encoder(model.model256_path)
            if pretrained_model is not None:
                self.logger.info(
                    f"Loading pretrained CellViT model from path: {pretrained_model}"
                )
                cellvit_pretrained = torch.load(pretrained_model, map_location="cpu")
                self.logger.info(model.load_state_dict(cellvit_pretrained, strict=True))
            model.freeze_encoder()
            self.logger.info("Loaded CellVit256 model")
        if backbone_type.lower() in ["sam-b", "sam-l", "sam-h"]:
            model_cfg = self.run_conf.get("model", {})
            in_ch = model_cfg.get("input_channels", 3)
            model = CellViTSAM(
                model_path=pretrained_encoder,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                vit_structure=backbone_type,
                drop_rate=self.run_conf["training"].get("drop_rate", 0),
                regression_loss=regression_loss,
                input_channels=in_ch,
            )
            model.load_pretrained_encoder(model.model_path)
            if in_ch > 3:
                from cellvit.models.cell_segmentation.cellvit_sam_rosie_early_fusion import expand_input_layer
                expand_input_layer(model.encoder, model.input_channels, "zeros")
                self.logger.info(f"Expanded encoder input to {in_ch} channels for proxy/early-fusion")
            if pretrained_model is not None:
                self.logger.info(
                    f"Loading pretrained CellViT model from path: {pretrained_model}"
                )
                cellvit_pretrained = torch.load(pretrained_model, map_location="cpu")
                self.logger.info(model.load_state_dict(cellvit_pretrained, strict=True))
            model.freeze_encoder()
            self.logger.info(f"Loaded CellViT-SAM model with backbone: {backbone_type}")
        
        if backbone_type.lower() == "sam-h-rosie-film":

            fusion_cfg = self.run_conf.get("fusion", {})
            model_cfg  = self.run_conf.get("model", {})

            # Prefer fusion.* (your YAML uses that), fallback to model.*
            freeze_cellvit = fusion_cfg.get("freeze_cellvit", model_cfg.get("freeze_cellvit", True))
            freeze_rosie   = fusion_cfg.get("freeze_rosie",   model_cfg.get("freeze_rosie", True))

            film_enabled   = fusion_cfg.get("film_enabled", True)
            film_layers    = fusion_cfg.get("film_layers", ["z4"])
            film_feat_dims = fusion_cfg.get("film_feat_dims", {})
            film_init      = fusion_cfg.get("film_init", "default")
            film_mode      = fusion_cfg.get("film_mode", "full")
            film_use_gating = fusion_cfg.get("film_use_gating", False)
            film_gating_init = fusion_cfg.get("film_gating_init", 0.0)
            film_gating_mode = fusion_cfg.get("film_gating_mode", "scalar")
            film_scale     = fusion_cfg.get("film_scale", 1.0)
            film_clamp_gamma = fusion_cfg.get("film_clamp_gamma")
            unfreeze_cellvit_epoch = fusion_cfg.get("unfreeze_cellvit_epoch", 0)
            unfreeze_last_n_blocks = fusion_cfg.get("unfreeze_last_n_blocks")
            unfreeze_full_encoder = fusion_cfg.get("unfreeze_full_encoder", False)
            debug_print_z_shapes = fusion_cfg.get("debug_print_z_shapes", False)
            rosie_marker_subset = fusion_cfg.get("rosie_marker_subset")
            rosie_marker_subset_indices = fusion_cfg.get("rosie_marker_subset_indices")
            conditioning_mode_train = fusion_cfg.get("conditioning_mode_train", "normal")
            conditioning_subset_indices = fusion_cfg.get("conditioning_subset_indices") or []
            conditioning_dropout = fusion_cfg.get("conditioning_dropout")

            model = CellViTSAMRosieFiLM(
                model_path=pretrained_encoder,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                vit_structure="sam-h",   # FIXED
                drop_rate=self.run_conf["training"].get("drop_rate", 0),
                regression_loss=regression_loss,
                rosie_hidden_dim=model_cfg.get("rosie_hidden_dim", 256),
                rosie_weights_path=model_cfg.get("rosie_weights_path", None),

                # YAML control behavior
                freeze_cellvit=freeze_cellvit,
                freeze_rosie=freeze_rosie,
                film_enabled=film_enabled,
                film_layers=tuple(film_layers),
                film_feat_dims=film_feat_dims,
                film_init=film_init,
                film_mode=film_mode,
                film_use_gating=film_use_gating,
                film_gating_init=film_gating_init,
                film_gating_mode=film_gating_mode,
                film_scale=film_scale,
                film_clamp_gamma=film_clamp_gamma,
                unfreeze_cellvit_epoch=unfreeze_cellvit_epoch,
                unfreeze_last_n_blocks=unfreeze_last_n_blocks,
                unfreeze_full_encoder=unfreeze_full_encoder,
                debug_print_z_shapes=debug_print_z_shapes,
                rosie_marker_subset=rosie_marker_subset,
                rosie_marker_subset_indices=rosie_marker_subset_indices,
                conditioning_mode_train=conditioning_mode_train,
                conditioning_subset_indices=conditioning_subset_indices,
                conditioning_dropout=conditioning_dropout,
            )

            model.load_pretrained_encoder(model.model_path)

            # Only freeze encoder if requested
            if freeze_cellvit:
                model.freeze_encoder()

            self.logger.info(
                f"Loaded sam-h-rosie-film | film_enabled={film_enabled} | film_layers={sorted(model.film_layers) if model.film_layers else []} | film_feat_dims={dict(model.film_feat_dims)}"
            )
            if conditioning_mode_train != "normal":
                self.logger.info(
                    f"Training conditioning override: conditioning_mode_train={conditioning_mode_train} | conditioning_subset_indices={conditioning_subset_indices}"
                )
            if conditioning_dropout is not None:
                self.logger.info(
                    f"Conditioning dropout enabled: {conditioning_dropout}"
                )

            # -------------------------
            # Debug: trainable params
            # -------------------------
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            self.logger.info(f"FiLM config: film_layers={sorted(model.film_layers)}, film_feat_dims={model.film_feat_dims}, trainable_params={trainable_params:,} / {total_params:,}")
            self.logger.info("---- Trainable Parameters ----")
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.logger.info(f"[TRAINABLE] {name}")

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            self.logger.info(f"Trainable params: {trainable_params:,} / {total_params:,}")

            # -------------------------
            # Debug: FiLM blocks (replaces z4_film)
            # -------------------------
            if getattr(model, "film_enabled", False) and len(getattr(model, "film_blocks", {})) > 0:
                self.logger.info("FiLM blocks:")
                for layer_name, block in model.film_blocks.items():
                    self.logger.info(f"  - {layer_name}: {block.__class__.__name__}")
                    for n, p in block.named_parameters():
                        self.logger.info(f"    {layer_name}.{n}: {tuple(p.shape)}")
            else:
                self.logger.info("FiLM disabled (baseline behavior)")

            # Rosie head
            self.logger.info("ROSIE classifier head:")
            self.logger.info(str(model.rosie_model.classifier[2]))

            # -------------------------
            # Optional: forward test
            # -------------------------
            try:
                dummy = torch.randn(1, 3, 128, 128).to(model.rosie_mean.device)
                with torch.no_grad():
                    out = model(dummy)
                self.logger.info(f"Dummy forward OK: {out['nuclei_type_map'].shape}")
            except Exception as e:
                self.logger.error(f"Forward pass failed: {e}")

            # -------------------------
            # Optional: FiLM stats
            # -------------------------
            if getattr(model, "film_enabled", False) and len(getattr(model, "film_blocks", {})) > 0:
                with torch.no_grad():
                    dummy_x = torch.randn(1, 3, 128, 128).to(model.rosie_mean.device)
                    x_rosie = model._rosie_preprocess(dummy_x)
                    rosie_full = model.rosie_model(x_rosie)
                    rosie_vec = rosie_full[:, model.rosie_marker_indices]

                    # pick first FiLM block to probe
                    first_layer = next(iter(model.film_blocks.keys()))
                    film_vec = model.film_blocks[first_layer].mlp(rosie_vec)

                    self.logger.info(f"ROSIE subset dim={model.rosie_dim_for_film} | vec mean {rosie_vec.mean().item():.4f}, std {rosie_vec.std().item():.4f}")
                    self.logger.info(f"FiLM vec mean {film_vec.mean().item():.4f}, std {film_vec.std().item():.4f}")

            # -------------------------
            # Finalize & return
            # -------------------------
            return model

        if backbone_type.lower() == "sam-h-proxy-film":
            from cellvit.models.cell_segmentation.cellvit_sam_proxy_film import (
                CellViTSAMProxyFiLM,
            )
            fusion_cfg = self.run_conf.get("fusion", {})
            model_cfg = self.run_conf.get("model", {})
            transform_cfg = self.run_conf.get("transformations", {})
            norm_cfg = transform_cfg.get("normalize", {})
            norm_mean = norm_cfg.get("mean", [0.5, 0.5, 0.5])
            norm_std = norm_cfg.get("std", [0.5, 0.5, 0.5])
            film_layers = fusion_cfg.get("film_layers", ["z4"])
            film_feat_dims = fusion_cfg.get("film_feat_dims", {})
            if not film_feat_dims and film_layers:
                film_feat_dims = {k: 1280 for k in film_layers}
            model = CellViTSAMProxyFiLM(
                model_path=pretrained_encoder,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                vit_structure="sam-h",
                drop_rate=self.run_conf["training"].get("drop_rate", 0),
                regression_loss=regression_loss,
                film_layers=tuple(film_layers),
                film_feat_dims=film_feat_dims,
                film_init=fusion_cfg.get("film_init", "default"),
                rosie_hidden_dim=model_cfg.get("rosie_hidden_dim", 256),
                conditioning_mode_train=fusion_cfg.get("conditioning_mode_train", "normal"),
                conditioning_mode_infer=fusion_cfg.get("conditioning_mode_infer", "normal"),
                normalize_mean=norm_mean,
                normalize_std=norm_std,
            )
            model.load_pretrained_encoder(model.model_path)
            model.freeze_encoder()
            self.logger.info(
                f"Loaded sam-h-proxy-film | film_layers={sorted(model.film_layers)} | "
                f"conditioning_mode_train={model.conditioning_mode_train} infer={model.conditioning_mode_infer}"
            )
            return model

        if backbone_type.lower() in ("sam-h-rosie-earlyfusion-vec", "sam-h-rosie-earlyfusion-mapc"):
            fusion_cfg = self.run_conf.get("fusion", {})
            model_cfg = self.run_conf.get("model", {})
            early_type = "vec_broadcast" if "vec" in backbone_type.lower() else "map_compress"
            early_compress = fusion_cfg.get("early_fusion_compress_out_channels", 8)

            model = CellViTSAMRosieEarlyFusion(
                model_path=pretrained_encoder,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                vit_structure="sam-h",
                drop_rate=self.run_conf["training"].get("drop_rate", 0),
                regression_loss=regression_loss,
                freeze_cellvit=fusion_cfg.get("freeze_cellvit", True),
                freeze_rosie=fusion_cfg.get("freeze_rosie", True),
                rosie_weights_path=model_cfg.get("rosie_weights_path", None),
                early_fusion_type=early_type,
                early_fusion_compress_out_channels=early_compress,
                rosie_marker_subset=fusion_cfg.get("rosie_marker_subset"),
                rosie_marker_subset_indices=fusion_cfg.get("rosie_marker_subset_indices"),
                early_fusion_detach_rosie=fusion_cfg.get("early_fusion_detach_rosie", True),
            )
            model.load_pretrained_encoder(model.model_path)
            if fusion_cfg.get("freeze_cellvit", True):
                model.freeze_encoder()
            self.logger.info(
                f"Loaded sam-h-rosie-earlyfusion | type={early_type} | "
                f"extra_channels={model.extra_channels} | rosie_markers={model.rosie_dim}"
            )
            try:
                dummy = torch.randn(1, 3, 256, 256).to(
                    model.rosie_mean.device if hasattr(model, "rosie_mean") else "cpu"
                )
                with torch.no_grad():
                    out = model(dummy)
                self.logger.info(f"Dummy forward OK: fused_x channels=3+{model.extra_channels} | nuclei_type_map {out['nuclei_type_map'].shape}")
            except Exception as e:
                self.logger.error(f"Forward pass failed: {e}")
            return model

        if backbone_type.lower() == "uni":
            model = CellViTUNI(
                model_uni_path=pretrained_encoder,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
            )
            if pretrained_model is not None:
                self.logger.info(
                    f"Loading pretrained CellViT model from path: {pretrained_model}"
                )
                cellvit_pretrained = torch.load(pretrained_model, map_location="cpu")
                self.logger.info(model.load_state_dict(cellvit_pretrained, strict=True))
            model.freeze_encoder()
            self.logger.info(f"Loaded CellViTUNI model with backbone: {backbone_type}")
        if backbone_type.lower() == "virchow":
            model = CellViTVirchow(
                model_virchow_path=pretrained_encoder,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
            )
            if pretrained_model is not None:
                self.logger.info(
                    f"Loading pretrained CellViT model from path: {pretrained_model}"
                )
                cellvit_pretrained = torch.load(pretrained_model, map_location="cpu")
                self.logger.info(model.load_state_dict(cellvit_pretrained, strict=True))
            model.freeze_encoder()
            self.logger.info(
                f"Loaded CellViTVirchow model with backbone: {backbone_type}"
            )
        if backbone_type.lower() == "virchow2":
            model = CellViTVirchow2(
                model_virchow_path=pretrained_encoder,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
            )
            if pretrained_model is not None:
                self.logger.info(
                    f"Loading pretrained CellViT model from path: {pretrained_model}"
                )
                cellvit_pretrained = torch.load(pretrained_model, map_location="cpu")
                self.logger.info(model.load_state_dict(cellvit_pretrained, strict=True))
            model.freeze_encoder()
            self.logger.info(
                f"Loaded CellViTVirchow2 model with backbone: {backbone_type}"
            )
        if backbone_type.lower() == "virchow-rosie-film":

            fusion_cfg = self.run_conf.get("fusion", {})
            model_cfg  = self.run_conf.get("model", {})

            # prefer fusion.*, fallback to model.*
            freeze_encoder = fusion_cfg.get("freeze_cellvit", model_cfg.get("freeze_encoder", True))
            freeze_rosie   = fusion_cfg.get("freeze_rosie", True)

            film_enabled   = fusion_cfg.get("film_enabled", True)
            film_layers    = fusion_cfg.get("film_layers", ["z4"])
            film_feat_dims = fusion_cfg.get("film_feat_dims", {})
            film_init      = fusion_cfg.get("film_init", "default")  # "identity" for gamma=1,beta=0 (recommended ablation)
            film_mode      = fusion_cfg.get("film_mode", "full")     # "full" or "beta_only"
            film_force_identity_train = fusion_cfg.get("film_force_identity_train", False)  # always gamma=1, beta=0 (train+eval)
            conditioning_mode_train = fusion_cfg.get("conditioning_mode_train", "normal")
            debug_print_z_shapes = fusion_cfg.get("debug_print_z_shapes", False)
            rosie_subset_indices = fusion_cfg.get("rosie_subset_indices")
            rosie_topk = fusion_cfg.get("rosie_topk")
            rosie_topk_method = fusion_cfg.get("rosie_topk_method", "energy")
            rosie_topk_cache_path = fusion_cfg.get("rosie_topk_cache_path")
            # Per-run cache path under log_dir to avoid race when multiple jobs share a file
            if rosie_topk is not None and rosie_topk > 0:
                log_dir = Path(self.run_conf["logging"]["log_dir"])
                rosie_topk_cache_path = str(
                    log_dir / f"rosie_topk_cache_{rosie_topk_method}_k{rosie_topk}.json"
                )
            rosie_topk_dataset_path = self.run_conf["data"].get("dataset_path")
            rosie_topk_seed = self.run_conf.get("random_seed")

            model = CellViTVirchowRosieFiLM(
                model_virchow_path=pretrained_encoder,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],

                # Rosie / FiLM settings
                rosie_hidden_dim=model_cfg.get("rosie_hidden_dim", 256),
                rosie_weights_path=model_cfg.get("rosie_weights_path", None),
                freeze_rosie=freeze_rosie,

                # FiLM controls (config-driven ablations)
                film_enabled=film_enabled,
                film_layers=tuple(film_layers),
                film_feat_dims=film_feat_dims,
                film_init=film_init,
                film_mode=film_mode,
                film_force_identity_train=film_force_identity_train,
                conditioning_mode_train=conditioning_mode_train,
                debug_print_z_shapes=debug_print_z_shapes,
                rosie_subset_indices=rosie_subset_indices,
                rosie_topk=rosie_topk,
                rosie_topk_method=rosie_topk_method,
                rosie_topk_cache_path=rosie_topk_cache_path,
                rosie_topk_dataset_path=rosie_topk_dataset_path,
                rosie_topk_seed=rosie_topk_seed,

                # NP spatial prior for L_sup / L_bd losses
                rosie_make_spatial_prior=fusion_cfg.get("rosie_make_spatial_prior", False),
                rosie_prior_from=fusion_cfg.get("rosie_prior_from", "rosie_backbone"),
                rosie_prior_channels=int(fusion_cfg.get("rosie_prior_channels", 50)),
            )

            model.load_pretrained_encoder(pretrained_encoder)

            if freeze_encoder:
                model.freeze_encoder()

            if conditioning_mode_train != "normal":
                self.logger.info(
                    f"Loaded virchow-rosie-film | film_enabled={film_enabled} | film_layers={film_layers} | conditioning_mode_train={conditioning_mode_train}"
                )
            else:
                self.logger.info(
                    f"Loaded virchow-rosie-film | film_enabled={film_enabled} | film_layers={film_layers}"
                )


        self.logger.info(f"\nModel: {model}")
        model = model.to("cpu")
        C = self.run_conf.get("model", {}).get("input_channels") or getattr(
            model, "input_channels", 3
        )
        H = W = self.run_conf.get("data", {}).get("input_shape", 256)
        try:
            self.logger.info(
                f"\n{summary(model, input_size=(1, C, H, W), device='cpu')}"
            )
        except Exception as e:
            self.logger.warning(
                f"torchinfo.summary failed (training will proceed): {e}"
            )

        return model

    def get_wandb_init_dict(self) -> dict:
        pass

    def get_transforms(
        self, transform_settings: dict, input_shape: int = 256
    ) -> Tuple[Callable, Callable]:
        """Get Transformations (Albumentation Transformations). Return both training and validation transformations.

        The transformation settings are given in the following format:
            key: dict with parameters
        Example:
            colorjitter:
                p: 0.1
                scale_setting: 0.5
                scale_color: 0.1

        For further information on how to setup the dictionary and default (recommended) values is given here:
        configs/examples/cell_segmentation/train_cellvit.yaml

        Training Transformations:
            Implemented are:
                - A.RandomRotate90: Key in transform_settings: randomrotate90, parameters: p
                - A.HorizontalFlip: Key in transform_settings: horizontalflip, parameters: p
                - A.VerticalFlip: Key in transform_settings: verticalflip, parameters: p
                - A.Downscale: Key in transform_settings: downscale, parameters: p, scale
                - A.Blur: Key in transform_settings: blur, parameters: p, blur_limit
                - A.GaussNoise: Key in transform_settings: gaussnoise, parameters: p, var_limit
                - A.ColorJitter: Key in transform_settings: colorjitter, parameters: p, scale_setting, scale_color
                - A.Superpixels: Key in transform_settings: superpixels, parameters: p
                - A.ZoomBlur: Key in transform_settings: zoomblur, parameters: p
                - A.RandomSizedCrop: Key in transform_settings: randomsizedcrop, parameters: p
                - A.ElasticTransform: Key in transform_settings: elastictransform, parameters: p
            Always implemented at the end of the pipeline:
                - A.Normalize with given mean (default: (0.5, 0.5, 0.5)) and std (default: (0.5, 0.5, 0.5))

        Validation Transformations:
            A.Normalize with given mean (default: (0.5, 0.5, 0.5)) and std (default: (0.5, 0.5, 0.5))

        Args:
            transform_settings (dict): dictionay with the transformation settings.
            input_shape (int, optional): Input shape of the images to used. Defaults to 256.

        Returns:
            Tuple[Callable, Callable]: Train Transformations, Validation Transformations

        """
        transform_list = []
        transform_settings = {k.lower(): v for k, v in transform_settings.items()}
        if "WhiteBorderAugmentation".lower() in transform_settings:
            p = transform_settings["whiteborderaugmentation"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(
                    A.RandomCrop(
                        height=int(input_shape / 2), width=int(input_shape / 3), p=p
                    )
                )
                transform_list.append(
                    A.PadIfNeeded(
                        min_height=input_shape,
                        min_width=input_shape,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=(255, 255, 255),
                        position="random",
                        always_apply=True,
                    )
                )
        if "RandomRotate90".lower() in transform_settings:
            p = transform_settings["randomrotate90"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.RandomRotate90(p=p))
        if "HorizontalFlip".lower() in transform_settings.keys():
            p = transform_settings["horizontalflip"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.HorizontalFlip(p=p))
        if "VerticalFlip".lower() in transform_settings:
            p = transform_settings["verticalflip"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.VerticalFlip(p=p))
        if "Downscale".lower() in transform_settings:
            p = transform_settings["downscale"]["p"]
            scale = transform_settings["downscale"]["scale"]
            if p > 0 and p <= 1:
                transform_list.append(
                    A.Downscale(p=p, scale_max=scale, scale_min=scale)
                )
        if "Blur".lower() in transform_settings:
            p = transform_settings["blur"]["p"]
            blur_limit = transform_settings["blur"]["blur_limit"]
            if p > 0 and p <= 1:
                transform_list.append(A.Blur(p=p, blur_limit=blur_limit))
        if "GaussNoise".lower() in transform_settings:
            p = transform_settings["gaussnoise"]["p"]
            var_limit = transform_settings["gaussnoise"]["var_limit"]
            if p > 0 and p <= 1:
                transform_list.append(A.GaussNoise(p=p, var_limit=var_limit))
        if "ColorJitter".lower() in transform_settings:
            p = transform_settings["colorjitter"]["p"]
            scale_setting = transform_settings["colorjitter"]["scale_setting"]
            scale_color = transform_settings["colorjitter"]["scale_color"]
            if p > 0 and p <= 1:
                transform_list.append(
                    A.ColorJitter(
                        p=p,
                        brightness=scale_setting,
                        contrast=scale_setting,
                        saturation=scale_color,
                        hue=scale_color / 2,
                    )
                )
        if "Superpixels".lower() in transform_settings:
            p = transform_settings["superpixels"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(
                    A.Superpixels(
                        p=p,
                        p_replace=0.1,
                        n_segments=200,
                        max_size=int(input_shape / 2),
                    )
                )
        if "ZoomBlur".lower() in transform_settings:
            p = transform_settings["zoomblur"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.ZoomBlur(p=p, max_factor=1.05))
        if "RandomSizedCrop".lower() in transform_settings:
            p = transform_settings["randomsizedcrop"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(
                    A.RandomSizedCrop(
                        min_max_height=(input_shape / 2, input_shape),
                        height=input_shape,
                        width=input_shape,
                        p=p,
                    )
                )
        if "ElasticTransform".lower() in transform_settings:
            p = transform_settings["elastictransform"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(
                    A.ElasticTransform(p=p, sigma=25, alpha=0.5, alpha_affine=15)
                )

        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        transform_list.append(A.Normalize(mean=mean, std=std))

        train_transforms = A.Compose(transform_list)
        val_transforms = A.Compose([A.Normalize(mean=mean, std=std)])

        return train_transforms, val_transforms

    def get_sampler(
        self, train_dataset: CellDataset, strategy: str = "random", gamma: float = 1
    ) -> Sampler:
        """Return the sampler (either RandomSampler or WeightedRandomSampler)

        Args:
            train_dataset (CellDataset): Dataset for training
            strategy (str, optional): Sampling strategy. Defaults to "random" (random sampling).
                Implemented are "random", "cell", "tissue", "cell+tissue".
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Raises:
            NotImplementedError: Not implemented sampler is selected

        Returns:
            Sampler: Sampler for training
        """
        if strategy.lower() == "random":
            sampling_generator = torch.Generator().manual_seed(
                self.default_conf["random_seed"]
            )
            sampler = RandomSampler(train_dataset, generator=sampling_generator)
            self.logger.info("Using RandomSampler")
        else:
            # this solution is not accurate when a subset is used since the weights are calculated on the whole training dataset
            if isinstance(train_dataset, Subset):
                ds = train_dataset.dataset
            else:
                ds = train_dataset
            ds.load_cell_count()
            if strategy.lower() == "cell":
                weights = ds.get_sampling_weights_cell(gamma)
            elif strategy.lower() == "tissue":
                weights = ds.get_sampling_weights_tissue(gamma)
            elif strategy.lower() == "cell+tissue":
                weights = ds.get_sampling_weights_cell_tissue(gamma)
            else:
                raise NotImplementedError(
                    "Unknown sampling strategy - Implemented are cell, tissue and cell+tissue"
                )

            if isinstance(train_dataset, Subset):
                weights = torch.Tensor([weights[i] for i in train_dataset.indices])

            sampling_generator = torch.Generator().manual_seed(
                self.default_conf["random_seed"]
            )
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(train_dataset),
                replacement=True,
                generator=sampling_generator,
            )

            self.logger.info(f"Using Weighted Sampling with strategy: {strategy}")
            self.logger.info(f"Unique-Weights: {torch.unique(weights)}")

        return sampler

    def get_trainer(self) -> BaseTrainer:
        """Return Trainer matching to this network

        Returns:
            BaseTrainer: Trainer
        """
        return CellViTTrainer
