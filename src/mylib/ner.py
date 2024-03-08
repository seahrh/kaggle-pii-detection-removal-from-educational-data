import gc
import json
from configparser import ConfigParser, SectionProxy
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scml
import torch
from pytorch_lightning.loggers import CSVLogger
from scml import torchx
from sklearn.metrics import fbeta_score, precision_score, recall_score
from torch.nn import functional as F
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
from mylib import ParamType, Task, training_callbacks

__all__ = [
    "NerDataset",
    "blend_predictions",
    "predict_ner",
    "predict_ner_proba",
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
    EVALUATION_CLASSES = list(range(1, 15))  # exclude `Outside` class from evaluation

    @staticmethod
    def stratification_group(easy: bool, medium: bool, hard: bool) -> int:
        ba: List[str] = ["0", "0", "0"]
        if easy:
            ba[2] = "1"
        if medium:
            ba[1] = "1"
        if hard:
            ba[0] = "1"
        return int("".join(ba), base=2)

    @staticmethod
    def split(n: int, window_length: int, window_stride: int) -> List[Tuple[int, int]]:
        res = []
        i = 0
        while i < n:
            j = min(n, i + window_length)
            res.append((i, j))
            i = i + window_length - window_stride
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
        self.stratification_groups: List[int] = []
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
        self.word_ids: List[List[Optional[int]]] = [[] for _ in range(len(self.texts))]

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
        self.word_ids[idx] = enc.word_ids()
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
                # use int64 instead of int32 to prevent error on nvidia gpu
                # https://stackoverflow.com/questions/69742930/runtimeerror-nll-loss-forward-reduce-cuda-kernel-2d-index-not-implemented-for
                dtype=torch.int64,
            )
        return res

    def __len__(self):
        return len(self.texts)


def blend_predictions(
    weights: np.ndarray,
    dw_map: Mapping[
        Tuple[int, int],
        torch.Tensor,
    ],
    outside_label_threshold: float,
) -> pd.DataFrame:
    rows = []
    for k, v in dw_map.items():
        p = np.matmul(weights, np.array(v, dtype=np.float32)).flatten()
        indices = (-p).argsort()  # sort in descending order
        i = 0
        # get top-2 if outside label is top-1 but falls below threshold
        if indices[0] == 0 and p[indices[0]] < outside_label_threshold:
            i = indices[1]
        elif indices[0] > 0:
            i = indices[0]
        if i > 0:
            rows.append(
                {"document": k[0], "token": k[1], "label": NerDataset.ID_TO_LABEL[i]}
            )
    df = pd.DataFrame.from_records(rows)
    df["row_id"] = df.index
    return df


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


def predict_ner_proba(
    ds: NerDataset,
    model: PreTrainedModel,
    batch_size: int,
    device: Optional[torch.device] = None,
    progress_bar: bool = False,
    dtype=np.float32,
) -> np.ndarray:
    logits = predict_ner(
        ds=ds,
        model=model,
        batch_size=batch_size,
        device=device,
        progress_bar=progress_bar,
        dtype=dtype,
    )
    y_proba = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)
    return np.array(y_proba, dtype=dtype)


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
        config.problem_type = "single_label_classification"
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

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
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
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if not self.automatic_optimization:
            for sch in self.lr_schedulers():
                if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sch.step(loss)

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
        self.tra_ds: Optional[NerDataset] = None
        self.val_ds: Optional[NerDataset] = None
        self.model: Optional[NerModel] = None
        self.trainer: Optional[pl.Trainer] = None
        self.batch_size: int = self.conf.getint("batch_size")
        self.devices: Union[List[int], str, int] = "auto"
        self.accelerator: str = "auto"
        if torch.cuda.is_available():
            self.accelerator = "gpu"
            self.devices = scml.to_int_list(self.conf["gpus"])
        elif torch.backends.mps.is_available():
            self.accelerator = "mps"
            self.devices = 1

    @staticmethod
    def _dataset(
        filepath: str,
        tokenizer_directory: str,
        model_max_length: int,
        window_length: int,
        window_stride: int,
        first_n: int = 0,
    ) -> NerDataset:
        with open(filepath) as f:
            data = json.load(f)
        # do not random sample else DDP gets different sized batches on different processes
        if first_n > 0:
            data = data[:first_n]
        elif first_n < 0:
            data = data[first_n:]
        texts: List[List[str]] = []
        labels: List[List[str]] = []
        dids: List[str] = []
        for row in data:
            texts.append(row["tokens"])
            labels.append(row["labels"])
            dids.append(str(row["document"]))
        config = AutoConfig.from_pretrained(tokenizer_directory)
        if hasattr(config, "max_position_embeddings"):
            config.max_position_embeddings = model_max_length
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_directory,
            model_max_length=model_max_length,
            config=config,
        )
        if str(getattr(config, "model_type")).lower() == "llama":
            tokenizer.pad_token = tokenizer.eos_token
        return NerDataset(
            tokenizer=tokenizer,
            texts=texts,
            labels=labels,
            document_ids=dids,
            window_length=window_length,
            window_stride=window_stride,
        )

    def _get_datasets(self) -> None:
        log.info("Prepare dataset...")
        with scml.Timer() as tim:
            train_data_first_n: int = self.conf.getint("train_data_first_n")
            self.tra_ds = self._dataset(
                filepath=self.conf["train_data_file"],
                tokenizer_directory=self.mc["directory"],
                model_max_length=self.conf.getint("model_max_length"),
                window_length=self.conf.getint("window_length"),
                window_stride=self.conf.getint("window_stride"),
                first_n=train_data_first_n,
            )
            log.info(f"len(tra)={len(self.tra_ds):,}\ntra[0]={self.tra_ds[0]}")
            gc.collect()
            self.val_ds = self._dataset(
                filepath=self.conf["validation_data_file"],
                tokenizer_directory=self.mc["directory"],
                model_max_length=self.conf.getint("model_max_length"),
                window_length=self.conf.getint("window_length"),
                window_stride=self.conf.getint("window_stride"),
            )
            log.info(f"len(val)={len(self.val_ds):,}\nval[0]={self.val_ds[0]}")
            gc.collect()
        log.info(f"Prepare dataset...DONE. Time taken {str(tim.elapsed)}")

    def _evaluation(self, epochs: int, device: Optional[torch.device] = None) -> None:
        if self.val_ds is None:
            raise ValueError("validation dataset must not be None")
        if self.model is None:
            raise ValueError("model must not be None")
        log.info("Evaluation...")
        with scml.Timer() as tim:
            predictions = predict_ner(
                ds=self.val_ds,
                model=self.model.model_to_save,
                batch_size=self.batch_size * 8,
                device=device,
                dtype=np.float16,
            )
            y_true: List[int] = []
            y_pred: List[int] = []
            classes = NerDataset.EVALUATION_CLASSES
            for i in range(len(self.val_ds)):
                # remember to convert torch tensor to python list!
                for j, label in enumerate(self.val_ds[i]["labels"].tolist()):
                    if label not in classes:
                        continue
                    y_true.append(label)
                    # (sequences, sequence length, classes)
                    y_pred.append(np.argmax(predictions[i][j]).item())
            log.debug(f"y_true={y_true}\ny_pred={y_pred}")
            f5_scores = fbeta_score(
                y_true=y_true,
                y_pred=y_pred,
                beta=5,
                average=None,
                labels=classes,
            )
            recall_scores = recall_score(
                y_true=y_true,
                y_pred=y_pred,
                average=None,
                labels=classes,
            )
            precision_scores = precision_score(
                y_true=y_true,
                y_pred=y_pred,
                average=None,
                labels=classes,
            )
            rows = []
            for i in range(len(f5_scores)):
                rows.append(
                    (f5_scores[i], recall_scores[i], precision_scores[i], classes[i])
                )
            rows.sort()
            labels = {}
            for f5, recall, precision, label_index in rows:
                label = NerDataset.ID_TO_LABEL[label_index]
                labels[label] = {
                    "micro_f5": f5,
                    "recall": recall,
                    "precision": precision,
                }
            self.validation_result = {
                "epochs": epochs,
                "micro_f5": fbeta_score(
                    y_true=y_true,
                    y_pred=y_pred,
                    beta=5,
                    average="micro",
                    labels=classes,
                ),
                "recall": recall_score(
                    y_true=y_true,
                    y_pred=y_pred,
                    average="micro",
                    labels=classes,
                ),
                "precision": precision_score(
                    y_true=y_true,
                    y_pred=y_pred,
                    average="micro",
                    labels=classes,
                ),
                "labels": labels,
            }
        log.info(f"Evaluation...DONE. Time taken {str(tim.elapsed)}")

    def _train_final_model(
        self,
        hps: Dict[str, ParamType],
    ) -> None:
        if self.tra_ds is None:
            raise ValueError("train dataset must not be None")
        if self.val_ds is None:
            raise ValueError("validation dataset must not be None")
        log.info("Train final model on best Hps...")
        log.info(f"hps={hps}")
        gc.collect()
        torch.cuda.empty_cache()
        with scml.Timer() as tim:
            log.info(f"len(tra)={len(self.tra_ds):,}, len(val)={len(self.val_ds):,}")
            self.model = NerModel(
                pretrained_dir=Path(self.mc["directory"]),
                lr=float(hps["lr"]),
                swa_start_epoch=int(hps["swa_start_epoch"]),
                scheduler_conf=self.scheduler_conf,
                conf=mylib.transformers_conf(self.conf),
            )
            eval_every_n_steps: int = self.conf.getint("eval_every_n_steps")
            self.trainer = pl.Trainer(
                default_root_dir=self.conf["job_dir"],
                strategy=self.conf.get("train_strategy", "auto"),
                accelerator=self.accelerator,
                devices=self.devices,
                max_epochs=self.conf.getint("epochs"),
                check_val_every_n_epoch=None if eval_every_n_steps > 0 else 1,
                val_check_interval=(
                    eval_every_n_steps if eval_every_n_steps > 0 else 1.0
                ),
                callbacks=training_callbacks(
                    patience=self.conf.getint("patience"),
                    eval_every_n_steps=eval_every_n_steps,
                ),
                deterministic=False,
                logger=CSVLogger(save_dir=self.conf["job_dir"]),
            )
            num_workers: int = self.conf.getint("dataloader_num_workers")
            ckpt_path: Optional[str] = self.conf.get("resume_training_from", "")
            if ckpt_path is not None and len(ckpt_path) == 0:
                ckpt_path = None
            self.trainer.fit(
                self.model,
                train_dataloaders=DataLoader(
                    self.tra_ds,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    persistent_workers=True if num_workers > 0 else False,
                ),
                val_dataloaders=DataLoader(
                    self.val_ds,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    persistent_workers=True if num_workers > 0 else False,
                ),
                ckpt_path=ckpt_path,
            )
            # logging special params
            model = self.model.model_to_save
            white = ["weighted_layer_pooling", "log_vars"]
            for name, param in model.named_parameters():  # type: ignore
                for w in white:
                    if name.startswith(w):
                        log.info(f"{name}={param}")
        log.info(f"Train final model on best Hps...DONE. Time taken {str(tim.elapsed)}")

    def run(self) -> None:
        with scml.Timer() as tim:
            self._get_datasets()
            self._train_final_model(
                hps={
                    "lr": self.conf.getfloat("lr"),
                    "swa_start_epoch": self.conf.getfloat("swa_start_epoch"),
                },
            )
            torch.distributed.barrier()
            if self.trainer is not None:
                if self.trainer.received_sigterm:
                    log.info("Exit now because signal.SIGTERM signal was received.")
                    return
                if self.trainer.is_global_zero:
                    if self.model is not None:
                        self._save_hf_model(
                            model=self.model.model_to_save,
                            dirpath=Path(self.conf["job_dir"]),
                        )
                    device: Optional[torch.device] = None
                    if self.accelerator == "gpu":
                        device = torch.device(
                            f"cuda:{self.devices[0]}"  # type: ignore[index]
                        )
                    elif self.accelerator == "mps":
                        device = torch.device("mps")
                    self._evaluation(epochs=self.trainer.current_epoch, device=device)
                    self._save_job_config()
                    self._copy_tokenizer_files(src=Path(self.mc["directory"]))
        if self.trainer is not None and self.trainer.is_global_zero:
            log.info(
                f"Total time taken {str(tim.elapsed)}. Saved {self.conf['job_dir']}"
            )
