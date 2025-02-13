import os
from pathlib import Path
from typing import Optional
from abc import abstractmethod
from easydict import EasyDict as edict

import torch
import torch.nn as nn

from src.utils.logger import setup_logger


class AbstractTrainer:
    def __init__(
        self,
        opt: edict,
        model: nn.Module,
        experiment_path: Optional[str] = None,
        restore_training: bool = False,
        weights_path: str = ''
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.opt = opt
        self.model = model.to(self.device)
        self.current_epoch = 0
        self.experiment_dir = experiment_path
        log_file = Path(experiment_path, "training_log.log")
        self.logger = setup_logger(name="main", log_file=log_file)
        self.weigth_path = weights_path

        self.logger.info(f"Using model: {self.model.__class__.__name__}")

        if not restore_training:
            # check if the folder exists, create if it's not
            os.makedirs(experiment_path, exist_ok=True)
            self.vis_dir = os.path.join(self.experiment_dir, "visualizations")
            self.ckpt_dir = os.path.join(self.experiment_dir, "checkpoints")

            os.makedirs(self.vis_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)

        else:
            try:
                self.load_weights()
            except Exception as e:
                # raise NotImplementedError("Restoration of the model has not been implemented yet.")
                self.logger.error(f"Failed to load weights with the provided weights path: {self.weigth_path}. See the traceback below: \n{e}")


    def load_weights(self):
        self.model.load_state_dict(self.weigths_path, weights_only=True)

    @abstractmethod
    def training_epoch(self):
        raise NotImplementedError

    @abstractmethod
    def validation_epoch(self):
        raise NotImplementedError

    @abstractmethod
    def save_state(self):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError
