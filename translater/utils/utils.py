from pathlib import Path
import torch
from config.data_dictionary import ROOT, Train, BPE_Enum


def get_checkpoint_path():
    if (Train.checkpoint_dir.value).startswith("gs://"):
        checkpoint_dir = Train.checkpoint_dir.value
        return f"{checkpoint_dir}/checkpoint.pth"
    else:
        checkpoint_dir = ROOT / Path(Train.checkpoint_dir.value)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir / "checkpoint.pth"


def get_log_dir():
    if (Train.log_dir.value).startswith("gs://"):
        return Train.log_dir.value
    else:
        log_dir = ROOT / Path(Train.log_dir.value)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir


def get_bpe_path():
    path = ROOT / Path(BPE_Enum.bpe_file.value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
