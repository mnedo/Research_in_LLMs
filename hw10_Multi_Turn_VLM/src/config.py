from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    root: Path = Path(".")
    data: Path = Path("./artifacts/data")
    ckpt: Path = Path("./artifacts/checkpoints")
    logs: Path = Path("./artifacts/logs")

    def mkdirs(self) -> None:
        self.data.mkdir(parents=True, exist_ok=True)
        self.ckpt.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)


@dataclass
class EnvCfg:
    size: int = 4
    slippery: bool = False
    max_steps: int = 128


@dataclass
class DataCfg:
    n_traj: int = 2000
    val_split: float = 0.15
    seed: int = 42


@dataclass
class TrainCfg:
    batch_size: int = 64
    lr: float = 2e-4
    epochs: int = 8
    device: str = "cuda"


@dataclass
class PPOCfg:
    lr: float = 1e-4
    gamma: float = 0.99
    clip_eps: float = 0.2
    update_epochs: int = 4
    rollout_steps: int = 512
    total_updates: int = 120
