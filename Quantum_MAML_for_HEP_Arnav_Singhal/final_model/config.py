# config.py
from dataclasses import dataclass, field
from typing import List
import os

@dataclass
class Config:
    # ---- Dataset Paths (done in collab) ----
    TRAIN_PATH: str = 'dataset path goes here'
    TEST_PATH: str = 'dataset path goes here'
    CHECKPOINT_DIR: str = 'dataset path goes here'
    Q_MAML_USE: bool = True
    LEARNER_HIDDEN: int = 256
    FREEZE_CNN_DURING_META: bool = True
    W0_SCALE: float = 0.01

    # ---- Data / Tasking ----
    SAMPLES: int = 90000
    META_TASK_TYPE: List[str] = field(default_factory=lambda: ['pt', 'm0'])
    META_BIN_COUNT: int = 6
    SUPPORT_SIZE: int = 8
    QUERY_SIZE: int = 8
    MAX_META_TASKS: int = 48

    # ---- Model sizes ----
    NUM_QUBITS: int = 6
    Q_DEPTH: int = 3
    ENCODING_SCHEME: str = 'angle'
    CNN_OUTPUT_DIM: int = 512
    USE_PRETRAINED_CNN: bool = True

    # ---- Task generation knobs ----
    TG_BIN_MODE: str = "quantile"
    TG_TARGET_JSD_MAX: float = 0.45
    TG_JSD_MAX_TRIES: int = 24
    TG_NUM_TASKS_PER_BIN: int = 6
    TG_TRAIN_SEED: int = 42
    TG_TEST_SEED: int = 43

    # ---- Caps & Guards ----
    TG_CAP_PT: float = 0.65
    TG_CAP_M0: float = 0.60
    GUARD_LOGREG_AUC_MIN: float = 0.58

    # ---- PCA Guard ----
    PCA_COMPONENTS: int = 128
    PCA_GUARD_CAL_TRIALS: int = 60
    PCA_GUARD_PERCENTILE: float = 0.60
    PCA_GUARD_CLAMP_MIN: float = 0.56
    PCA_GUARD_CLAMP_MAX: float = 0.90

    # ---- Warm-up / Adaptive gates ----
    WARMUP_ACCEPT_N: int = 24
    WARMUP_PCA_DELTA: float = 0.03
    ADAPTIVE_M0_LOOSE_CAP: float = 0.50
    ADAPTIVE_M0_MIN_ACCEPT: int = 3

    # ---- Training ----
    USE_ANALYTIC_GRADIENTS: bool = True
    INNER_STEPS: int = 12
    INNER_LR: float = 0.02
    OUTER_LR: float = 5e-3
    EPOCHS: int = 15
    BATCH_SIZE: int = 24
    EVAL_METRICS: bool = True
    SAVE_BEST_MODEL: bool = True

    # ---- Pretraining ----
    PRETRAIN_QMAML: bool = True
    USE_QMAML_PRETRAINED_INIT: bool = True
    QMAML_INIT_TAG: str = "qmaml_init"


config = Config()
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

__all__ = ["Config", "config"]
