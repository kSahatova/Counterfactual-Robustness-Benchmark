import os
import random
import numpy as np
from typing import Any

import torch
import tensorflow as tf


DEFAULT_RANDOM_SEED = 2025


def seed_basic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


# tensorflow random seed
def seed_tf(seed=DEFAULT_RANDOM_SEED):
    tf.random.set_seed(seed)


# torch random seed
def seed_torch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# basic + tensorflow + torch
def seed_everything(seed=DEFAULT_RANDOM_SEED):
    seed_basic(seed)
    seed_tf(seed)
    seed_torch(seed)


def load_model(model: Any, framework: str = "torch", weights_path: str = "", **kwargs):
    """Load pytorch model from the given weights path"""

    try:
        if framework == "torch":
            model.load_state_dict(torch.load(weights_path))
            return model
        elif framework == "tf":
            tf.keras.models.load_model(weights_path, kwargs["custom_objects"])
    except ValueError:
        print(
            "Only pytorch or tensorflow models can loaded, check 'framework' argument which should be 'torch' or 'tf'"
        )
