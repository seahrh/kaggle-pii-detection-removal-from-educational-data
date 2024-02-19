import gc
import json
import math
import random
from configparser import ConfigParser, SectionProxy
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import scml
import torch
from pytorch_lightning.loggers import CSVLogger
from scml import torchx

# from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

import mylib
from mylib import ParamType, Task, Trainer, training_callbacks

__all__ = [
    "NerDataset",
    "predict_ner",
    "NerModel",
    "NerTask",
]

log = scml.get_logger(__name__)


class NerDataset(Dataset):
    ID_TO_LABEL: Mapping[int, str] = {
        0: "O",
        1: "B-NAME_STUDENT",
        2: "I-NAME_STUDENT",
        3: "B-URL_PERSONAL",
        4: "I-URL_PERSONAL",
        5: "B-ID_NUM",
        6: "I-ID_NUM",
        7: "B-EMAIL",
        8: "I-EMAIL",
        9: "B-USERNAME",
        10: "I-USERNAME",
        11: "B-PHONE_NUM",
        12: "I-PHONE_NUM",
        13: "B-STREET_ADDRESS",
        14: "I-STREET_ADDRESS",
    }
    LABEL_TO_ID: Mapping[str, int] = {v: k for k, v in ID_TO_LABEL.items()}
    IGNORE_LABEL: int = -100

    @staticmethod
    def split(n: int, window_length: int, window_stride: int) -> List[Tuple[int, int]]:
        res = []
        i = 0
        while i < n:
            j = min(n, i + window_length)
            res.append((i, j))
            i = j - window_stride
        return res

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        window_length: int,
        window_stride: int,
        texts: List[List[str]],
        document_ids: List[str],
        labels: Optional[List[List[str]]] = None,
    ):
        self.tokenizer = tokenizer
        self.document_ids: List[str] = []
        self.word_ranges: List[Tuple[int, int]] = []
        self.texts: List[List[str]] = []
        self.labels: List[List[str]] = []
        for i in range(len(texts)):
            for j, k in self.split(
                n=len(texts[i]),
                window_length=window_length,
                window_stride=window_stride,
            ):
                self.document_ids.append(document_ids[i])
                self.word_ranges.append((j, k))
                self.texts.append(texts[i][j:k])
                if labels is not None:
                    self.labels.append(labels[i][j:k])

    def __getitem__(self, idx):
        res = {}
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            is_split_into_words=True,  # processing an array of string tokens (instead of a string)
            add_special_tokens=True,
            return_overflowing_tokens=False,
            return_offsets_mapping=False,
            return_special_tokens_mask=False,
        )
        for k, v in enc.items():
            t = torch.tensor(v)
            log.debug(f"{k} {t.shape}")
            res[k] = t
        if len(self.labels) != 0:
            word_labels = self.labels[idx]
            labels = []
            prev = None
            for wid in enc.word_ids():
                # Only label the first token of a given word.
                if wid is not None and wid != prev:
                    labels.append(self.LABEL_TO_ID[word_labels[wid]])
                else:
                    labels.append(self.IGNORE_LABEL)  # Set the special tokens to -100.
                prev = wid
            res["labels"] = torch.tensor(
                labels,
                dtype=torch.int32,
            )
        return res

    def __len__(self):
        return len(self.texts)


def predict_ner(
    ds: NerDataset,
    model: PreTrainedModel,
    batch_size: int,
    device: Optional[torch.device] = None,
    progress_bar: bool = False,
    dtype=np.float32,
) -> np.ndarray:
    if device is None:
        device = torch.device("cpu")
    torch.cuda.empty_cache()
    res = []
    batches = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()  # type: ignore
    model.to(device)  # type: ignore
    with torch.no_grad():
        for batch in tqdm(
            batches, desc="predict ner", disable=not progress_bar, mininterval=10
        ):
            for k, v in batch.items():
                batch[k] = v.to(device)
            outputs = model(**batch)  # type: ignore
            # (batch_size, sequence_length, config.num_labels)
            logits = outputs.logits.detach().cpu()
            log.debug(f"logits.size={logits.size()}\n{logits}")
            res += logits.tolist()
    return np.array(res, dtype=dtype)


# noinspection PyAbstractClass
class NerModel(pl.LightningModule):
    def __init__(
        self,
        pretrained_dir: Path,
        lr: float,
        scheduler_conf: Iterable[SectionProxy],
        swa_start_epoch: int = -1,
        conf: Optional[Mapping[str, ParamType]] = None,
    ):
        super().__init__()
        self.automatic_optimization = True
        self.lr = lr
        self.scheduler_conf = list(scheduler_conf)
        for s in self.scheduler_conf:
            if s["qualified_name"] == "torch.optim.swa_utils.SWALR":
                s["swa_lr"] = str(self.lr)
        # TODO pass instantiated model to constructor if custom model
        config = AutoConfig.from_pretrained(str(pretrained_dir))
        config.problem_type = "classification"
        config.id2label = NerDataset.ID_TO_LABEL
        config.label2id = NerDataset.LABEL_TO_ID
        if conf is not None:
            for k, v in conf.items():
                setattr(config, k, v)
        model = AutoModelForTokenClassification.from_pretrained(
            str(pretrained_dir),
            config=config,
            ignore_mismatched_sizes=True,
        )
        self.model = model
        self.model_to_save = self.model
        self.swa_start_epoch = swa_start_epoch
        self.swa_enable: bool = self.swa_start_epoch >= 0
        # only `AveragedModel.module` is used in its forward pass, so we save `AveragedModel.module`
        # see https://github.com/pytorch/pytorch/blob/master/torch/optim/swa_utils.py
        if self.swa_enable:
            self.automatic_optimization = False
            self.swa_model = torch.optim.swa_utils.AveragedModel(model=self.model)
            self.model_to_save = self.swa_model.module
        self._has_swa_started: bool = False
        # (samples, tokens, labels)
        self.val_pred: List[List[List[float]]] = []
        self.best_val_pred: Optional[np.ndarray] = None

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        if not self.automatic_optimization:
            opts = self.optimizers()
            if not isinstance(opts, list):
                opts = [opts]
            for opt in opts:
                opt.zero_grad()
                self.manual_backward(loss)
                opt.step()
        return loss

    def on_train_epoch_end(self):
        if self.automatic_optimization:
            return
        epoch = self.trainer.current_epoch
        schedulers = self.lr_schedulers()
        if self.swa_enable:
            if epoch >= self.swa_start_epoch:
                self._has_swa_started = True
                self.swa_model.update_parameters(self.model)
                for sch in schedulers:
                    if isinstance(sch, torch.optim.swa_utils.SWALR):
                        sch.step()
                return
            schedulers = [
                sch
                for sch in schedulers
                if not isinstance(sch, torch.optim.swa_utils.SWALR)
            ]
        if schedulers is None:
            return
        if not isinstance(schedulers, list):
            schedulers = [schedulers]
        for sch in schedulers:
            if not isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step()

    def validation_step(self, batch, batch_idx):
        model = self.model
        if self._has_swa_started:
            model = self.swa_model
        outputs = model(**batch)
        loss = outputs.loss
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if not self.automatic_optimization:
            for sch in self.lr_schedulers():
                if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sch.step(loss)
        if batch_idx == 0:
            self.val_pred = []
        logits = outputs.logits.detach()
        log.debug(f"{logits.shape} logits={logits}")
        self.val_pred += logits.tolist()

    def configure_optimizers(self):
        """

        :return: Two lists - The first list has multiple optimizers,
        and the second has multiple LR schedulers (or multiple lr_scheduler_config).
        """
        # The optimizer does not seem to reference any FSDP parameters.
        # HINT: Make sure to create the optimizer after setting up the model
        # by referencing `self.trainer.model.parameters()` in the `configure_optimizers()` hook.
        optimizers = [
            torch.optim.AdamW(
                self.trainer.model.parameters(),
                lr=self.lr,
                amsgrad=False,
            )
        ]
        schedulers = torchx.schedulers_by_config(
            optimizer=optimizers[0], sections=self.scheduler_conf
        )
        if self.swa_start_epoch >= 0:
            if len(schedulers) == 0:
                raise ValueError("For SWA, there must be at least one scheduler")
            if not isinstance(schedulers[0].scheduler, torch.optim.swa_utils.SWALR):
                raise ValueError(
                    "For SWA, the first scheduler must be of the type `torch.optim.swa_utils.SWALR`"
                )
        return optimizers, [s._asdict() for s in schedulers]


class NerTask(Task):
    name: str = "ner"

    def __init__(
        self,
        conf: ConfigParser,
    ):
        super().__init__(conf=conf)

    def _dataset(self) -> NerDataset:
        log.info("Prepare dataset...")
        with scml.Timer() as tim:
            with open(self.conf["train_data_file"]) as f:
                data = json.load(f)
            frac: float = self.conf.getfloat("train_data_sample_frac")
            if frac < 1:
                tmp = []
                for i in random.sample(range(len(data)), k=math.ceil(frac * len(data))):
                    tmp.append(data[i])
                data = tmp
            texts: List[List[str]] = []
            labels: List[List[str]] = []
            dids: List[str] = []
            for row in data:
                texts.append(row["tokens"])
                labels.append(row["labels"])
                dids.append(str(row["document"]))
            model_max_length: int = self.conf.getint("model_max_length")
            tokenizer = AutoTokenizer.from_pretrained(
                self.mc["directory"],
                model_max_length=model_max_length,
            )
            config = AutoConfig.from_pretrained(self.mc["directory"])
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = model_max_length
            if str(getattr(config, "model_type")).lower() == "llama":
                tokenizer.pad_token = tokenizer.eos_token
            ds = NerDataset(
                tokenizer=tokenizer,
                texts=texts,
                labels=labels,
                document_ids=dids,
                window_length=self.conf.getint("window_length"),
                window_stride=self.conf.getint("window_stride"),
            )
            log.info(f"train_data_sample_frac={frac}, len(ds)={len(ds)}\nds[0]={ds[0]}")
            del data
            gc.collect()
        log.info(f"Prepare dataset...DONE. Time taken {str(tim.elapsed)}")
        return ds

    def _train_final_model(
        self, ds: NerDataset, hps: Dict[str, ParamType], num_workers: int = 0
    ) -> None:
        log.info("Train final model on best Hps...")
        log.info(f"hps={hps}")
        gc.collect()
        torch.cuda.empty_cache()
        with scml.Timer() as tim:
            # TODO split beforehand and save versioned test set
            splitter = KFold(
                n_splits=int(100 / self.conf.getint("final_model_validation_percent"))
            )
            dummy = np.zeros(len(ds))
            y = [max(z) for z in ds.labels]
            for ti, vi in splitter.split(dummy, y=y):
                tra_ds = torch.utils.data.Subset(ds, ti)
                val_ds = torch.utils.data.Subset(ds, vi)
                break
            log.info(f"len(tra_ds)={len(tra_ds)}, len(val_ds)={len(val_ds)}")
            model = NerModel(
                pretrained_dir=Path(self.mc["directory"]),
                lr=float(hps["lr"]),
                swa_start_epoch=int(hps["swa_start_epoch"]),
                scheduler_conf=self.scheduler_conf,
                conf=mylib.transformers_conf(self.conf),
            )
            devices: Union[List[int], str, int] = "auto"
            accelerator: str = "auto"
            if torch.cuda.is_available():
                accelerator = "gpu"
                devices = scml.to_int_list(self.conf["gpus"])
            elif torch.backends.mps.is_available():
                accelerator = "mps"
                devices = 1
            trainer = Trainer(
                default_root_dir=self.conf["job_dir"],
                accelerator=accelerator,
                devices=devices,
                max_epochs=self.conf.getint("epochs"),
                callbacks=training_callbacks(patience=self.conf.getint("patience")),
                deterministic=False,
                logger=CSVLogger(save_dir=self.conf["job_dir"]),
                log_every_n_steps=100,
            )
            trainer.fit(
                model,
                train_dataloaders=DataLoader(
                    tra_ds,
                    batch_size=self.conf.getint("batch_size"),
                    shuffle=True,
                    num_workers=num_workers,
                ),
                val_dataloaders=DataLoader(
                    val_ds,
                    batch_size=self.conf.getint("batch_size"),
                    shuffle=False,
                    num_workers=num_workers,
                    # persistent_workers=True,
                ),
            )
        log.info(f"Train final model on best Hps...DONE. Time taken {str(tim.elapsed)}")

    def run(self) -> None:
        with scml.Timer() as tim:
            self._save_job_config()
            ds = self._dataset()
            self._train_final_model(
                ds=ds,
                hps={
                    "lr": self.conf.getfloat("lr"),
                    "swa_start_epoch": self.conf.getfloat("swa_start_epoch"),
                },
            )
            self._copy_tokenizer_files(src=Path(self.mc["directory"]))
        log.info(f"Total time taken {str(tim.elapsed)}. Saved {self.conf['job_dir']}")
