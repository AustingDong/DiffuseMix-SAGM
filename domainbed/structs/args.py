from typing import Any, List
from pydantic import BaseModel
from pathlib import Path

class Args(BaseModel):
    name: str
    configs: Any
    data_dir: Path
    dataset: str
    algorithm: str
    trial_seed: int
    seed: int
    steps: int | None
    checkpoint_freq: int | None
    test_envs: List[List[int]] | None
    holdout_fraction: float
    model_save: int | None
    deterministic: bool
    tb_freq: int
    debug: bool
    show: bool
    evalmode: str  # [fast, all].

    unique_name: str
    work_dir: Path
    out_root: Path
    out_dir: Path

    real_test_envs: List[int]

    class Config:
        protected_namespaces = ()
