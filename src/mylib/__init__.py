import json
import os
import shutil
from configparser import ConfigParser, SectionProxy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pytorch_lightning as pl
import scml
from pytorch_lightning.strategies import DeepSpeedStrategy
from scml import configparserx as cpx
from transformers import AutoConfig, PreTrainedModel

__all__ = [
    "Task",
    "Trainer",
    "training_callbacks",
    "transformers_conf",
    "ParamType",
]


ParamType = Union[str, int, float, bool]
log = scml.get_logger(__name__)


def training_callbacks(
    patience: int,
    monitor: str = "val_loss",
    verbose: bool = True,
) -> List[pl.callbacks.Callback]:
    return [
        pl.callbacks.EarlyStopping(monitor=monitor, patience=patience, verbose=verbose),
        pl.callbacks.ModelCheckpoint(monitor=monitor, verbose=verbose, save_top_k=1),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]


def transformers_conf(conf: SectionProxy) -> Dict[str, ParamType]:
    res: Dict[str, ParamType] = {}
    if "gradient_checkpointing" in conf:
        res["gradient_checkpointing"] = conf.getboolean("gradient_checkpointing")
    if "hidden_dropout_prob" in conf:
        res["hidden_dropout_prob"] = conf.getfloat("hidden_dropout_prob")
    if "attention_probs_dropout_prob" in conf:
        res["attention_probs_dropout_prob"] = conf.getfloat(
            "attention_probs_dropout_prob"
        )
    if "max_position_embeddings" in conf:
        res["max_position_embeddings"] = conf.getint("max_position_embeddings")
    return res


class Trainer(pl.Trainer):
    def save_checkpoint(
        self,
        filepath,
        weights_only: bool = False,
        storage_options: Optional[Any] = None,
    ) -> None:
        if self.is_global_zero and not isinstance(self.strategy, DeepSpeedStrategy):
            # save oof predictions from best model
            v = getattr(self.lightning_module, "val_pred", None)
            if v is not None:
                self.lightning_module.best_val_pred = np.array(v, dtype=np.float16)  # type: ignore
            model = self.lightning_module.model_to_save
            white = ["weighted_layer_pooling", "log_vars"]
            for name, param in model.named_parameters():  # type: ignore
                for w in white:
                    if name.startswith(w):
                        log.info(f"{name}={param}")
            if isinstance(model, PreTrainedModel):
                dirpath = os.path.split(filepath)[0]
                model.save_pretrained(dirpath)  # type: ignore
                return
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
            dst = Path(self.conf["job_dir"]) / "lightning_logs/version_0/checkpoints"
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
