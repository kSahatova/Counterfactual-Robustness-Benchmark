import os.path as osp
from typing import Dict, Optional

import matplotlib.pyplot as plt
# import seaborn as sns


def visualize_training_results(
    train_stats: Dict, val_stats: Optional[Dict] = None, save_dir: str = ""
):
    """Visualizes and  saves the results of the training process"""
    plt.style.use("seaborn-v0_8-darkgrid")
    n_cols = 2
    if not val_stats:
        n_cols = 1

    fig, axs = plt.subplots(1, n_cols, figsize=(10, 3))

    axs[0].plot(train_stats["loss"], label="train", color="tab:blue")
    axs[0].plot(val_stats["loss"], label="val", color="tab:orange")
    axs[0].legend()
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Epochs")

    axs[1].plot(train_stats["f1score"], label="train", color="tab:blue")
    axs[1].plot(val_stats["f1score"], label="val", color="tab:orange")
    axs[1].legend()
    axs[1].set_ylabel("F1 score")
    axs[1].set_xlabel("Epochs")

    file_name = "training_results.png"
    if osp.isdir(save_dir):
        fig.savefig(osp.join(save_dir, file_name))
