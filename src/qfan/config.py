"""
Typed configuration objects for QFAN.

Jamal Slim
jamal.slim@desy.de


All hyperparameters are organized into small dataclasses per-subsystem.
Two presets are provided (d12_config, d25_config) matching the paper's
Sec. VII (d=12 main experiment) and Appendix A (d=25 extension).
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


# -------------------- sub-configs --------------------

@dataclass
class DataConfig:
    data_path: str = "cal_shower_img_12q.npy"
    test_size: float = 0.2
    seed: int = 7
    use_subset: bool = True
    subset_mode: str = "random"        # "random" | "stride" | "kmeans"
    subset_n: Optional[int] = 6000
    subset_frac: Optional[float] = None
    subset_apply: str = "before_split"  # "before_split" | "train_only"
    subset_kmeans_max_dim: int = 256
    subset_kmeans_ninit: int = 5
    subset_seed: int = 123


@dataclass
class SketchConfig:
    sketch_dim: int = 32
    max_dim_sketch: int = 16384
    use_mixer: bool = True
    nonlinearity: str = "asinh"        # "none" | "tanh" | "asinh"
    len_norm: bool = True
    clip_nonnegative: bool = True


@dataclass
class BankConfig:
    n_qubits: int = 3
    q_depth: int = 2                    # paper Fig. 3 uses L=2
    angle_dim: int = 8
    measure_weight2: bool = True        # {Z_iZ_j, X_iX_j} — critical (paper Table VIII)
    n_shots: int = 256
    feature_chunk_size: int = 256
    ridge_alpha: float = 1e-2


@dataclass
class MMDConfig:
    sigma_mults: Tuple[float, ...] = (0.5, 1.0, 2.0)


@dataclass
class TrainConfig:
    # "exact"  = parameter-shift + Adam + optional GN-preconditioner + EMA + cosine LR
    # "spsa"   = paper baseline (same MMD definition / val monitor for fair ablation)
    mode: str = "exact"

    # --- exact-gradient hyperparameters ---
    epochs: int = 60
    batch: int = 128
    val_batch: int = 512
    lr: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 5.0
    lr_warmup: int = 5
    lr_decay: str = "cosine"            # "none" | "cosine"
    lr_floor_frac: float = 0.1          # cosine final LR = lr * lr_floor_frac
    ema_beta: float = 0.9               # EMA of val_loss for smoothed monitor
    pr_tail_frac: float = 0.5           # Polyak-Ruppert averaging window
    use_precond: bool = False           # Gauss-Newton preconditioning (free but optional)
    precond_damp: float = 1e-2
    precond_every: int = 3
    all_blocks: bool = True
    block_subset: Optional[int] = None

    # --- SPSA baseline hyperparameters ---
    spsa_steps: int = 120
    spsa_batch: int = 128
    spsa_a0: float = 0.08
    spsa_c0: float = 0.12
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101


@dataclass
class ResidualConfig:
    use_safe_gate: bool = True
    resid_qubits: int = 3
    resid_max_clusters: int = 64
    resid_kmeans_ninit: int = 10
    resid_whiten: bool = True
    resid_whiten_eps: float = 1e-6
    resid_whiten_shrink: float = 1e-2
    resid_sample_temp: float = 1.0
    resid_jitter_scale: float = 0.0


@dataclass
class RolloutConfig:
    epochs: int = 6
    max_rollout_ratio: float = 0.85
    monitor_n: int = 512
    reseed: int = 2025


@dataclass
class CopulaConfig:
    enabled: bool = True
    shrink: float = 0.05
    eps: float = 1e-6


# -------------------- master config --------------------

@dataclass
class QFANConfig:
    """
    One container for the whole pipeline. Pass instances of this to the
    top-level `run_pipeline` in the scripts.
    """
    data: DataConfig = field(default_factory=DataConfig)
    sketch: SketchConfig = field(default_factory=SketchConfig)
    bank: BankConfig = field(default_factory=BankConfig)
    mmd: MMDConfig = field(default_factory=MMDConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    copula: CopulaConfig = field(default_factory=CopulaConfig)

    block_size: int = 6                 # paper: B = ceil(d/b); d=12, b=6 -> B=2
    do_prediction_eval: bool = True
    plot_loss_curve: bool = True
    run_tag: str = "d12"                # used to name output files


# -------------------- presets --------------------

def d12_config() -> QFANConfig:
    """
    Paper Sec. VII: d=12, b=6 -> B=2 blocks, nq=3, L=2 -> p_theta=12, G=2.
    Matches paper-faithful demonstration scale.
    """
    cfg = QFANConfig()
    cfg.run_tag = "d12"
    cfg.block_size = 6
    cfg.data.data_path = "cal_shower_img_12q.npy"
    cfg.bank.n_qubits = 3
    cfg.bank.q_depth = 2                # L=2, matches paper Fig. 3
    return cfg


def d25_config() -> QFANConfig:
    """
    Paper Appendix A: d=25, b=5 -> B=5 blocks, nq=3, L=3 -> p_theta=18, G=2.
    """
    cfg = QFANConfig()
    cfg.run_tag = "d25"
    cfg.block_size = 5
    cfg.data.data_path = "cal_shower_img_25q.npy"
    cfg.bank.n_qubits = 3
    cfg.bank.q_depth = 3                # L=3 (p_theta = 18), appendix A
    return cfg
