import json
import shutil
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pytorch_lightning as pl
import scml
import torch
from scml import configparserx as cpx
from transformers import AutoConfig

__all__ = [
    "Task",
    "training_callbacks",
    "ParamType",
]


ParamType = Union[str, int, float, bool]
log = scml.get_logger(__name__)


def training_callbacks(
    patience: int,
    eval_every_n_steps: int,
    ckpt_filename: str,
    monitor: str = "val_loss",
    save_top_k: int = 1,
    verbose: bool = True,
) -> List[pl.callbacks.Callback]:
    return [
        pl.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, verbose=verbose, divergence_threshold=10
        ),
        pl.callbacks.ModelCheckpoint(
            monitor=monitor,
            verbose=verbose,
            save_top_k=save_top_k,
            save_on_train_epoch_end=False,
            every_n_train_steps=(
                eval_every_n_steps + 1 if eval_every_n_steps > 0 else None
            ),
            filename=ckpt_filename,
            auto_insert_metric_name=False,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval=None),
    ]


class DeprecatedTrainer(pl.Trainer):
    def save_checkpoint(
        self,
        filepath,
        weights_only: bool = False,
        storage_options: Optional[Any] = None,
    ) -> None:
        torch.distributed.barrier()
        if self.is_global_zero:
            # save oof predictions from best model
            v = getattr(self.lightning_module, "val_pred", None)
            if v is not None:
                self.lightning_module.best_val_pred = np.array(v, dtype=np.float16)  # type: ignore
        super().save_checkpoint(filepath, weights_only, storage_options)


class Task:
    name: str = "default_task_name"

    def __init__(self, conf: ConfigParser):
        self.full_conf = conf
        self.conf = conf[self.name]
        self.mc = conf[conf[self.name]["backbone"]]
        schedulers: List[str] = conf[self.name]["schedulers"].split()
        self.scheduler_conf = [conf[s] for s in schedulers]
        self.config = AutoConfig.from_pretrained(self.mc["directory"])
        self.validation_result: Optional[Mapping] = None

    def run(self) -> None:
        raise NotImplementedError

    def _save_job_config(self) -> None:
        filepath = Path(self.conf["job_dir"]) / "train.json"
        with open(str(filepath), "w") as f:
            d: Dict = {}
            if self.validation_result is not None:
                d.update(self.validation_result)
            d.update(cpx.as_dict(self.full_conf))
            json.dump(d, f)

    def _copy_tokenizer_files(self, src: Path) -> None:
        log.info("Copy tokenizer files...")
        with scml.Timer() as tim:
            dst = Path(self.conf["job_dir"])
            for f in src.glob("*.txt"):
                if not f.is_file():
                    continue
                shutil.copy(str(f), str(dst))
            for f in src.glob("*.json"):
                if not f.is_file():
                    continue
                if f.stem == "config":
                    continue
                shutil.copy(str(f), str(dst))
            for f in src.glob("*.model"):
                if not f.is_file():
                    continue
                shutil.copy(str(f), str(dst))
            for f in src.glob("*encoder.bin"):
                if not f.is_file():
                    continue
                shutil.copy(str(f), str(dst))
        log.info(f"Copy tokenizer files...DONE. Time taken {str(tim.elapsed)}")


from .ner import *

__all__ += ner.__all__  # type: ignore
