"""
qfan -- Quantum Feature Amplification Network.




Jamal Slim
jamal.slim@desy.de

Organized as:
    config       hyperparameters (dataclasses for d=12 and d=25)
    utils        numerical helpers (whiten, psd sqrt, masks)
    data         dataset loading + subset selection
    sketch       streaming count-sketch (paper Sec. III-B)
    mmd          multi-kernel MMD + analytic gradient
    quantum      Pauli grouping + parameterized shot-bank + parameter-shift
    ridge        closed-form ridge decoder
    train        EXACT parameter-shift gradient training (core novelty)
                 + SPSA baseline + Adam + Gauss-Newton preconditioning
                 + EMA tracking + cosine LR decay + PR averaging
    decoder      residual gate (classical pool sampler, paper-compatible)
    pipeline     fit_all_blocks, sample_progressive, predict_future, rollout_refine
    calibration  Gaussian-copula post-hoc correlation repair
    metrics      correlation error, W1, MSE summaries
"""

from .config import (
    QFANConfig, TrainConfig, DataConfig, SketchConfig,
    BankConfig, ResidualConfig, RolloutConfig, CopulaConfig, MMDConfig,
    d12_config, d25_config,
)
from .utils import (
    corr_nan_safe, _stable_sigmoid, _power_of_two_floor,
    _whiten, _psd_sqrt, _psd_invsqrt,
)
from .data import (
    load_dataset, select_subset_indices, _resolve_subset_n,
)
from .sketch import (
    OnlineCountSketch, build_blocks, build_prefix_sketch_cache,
)
from .mmd import (
    mmd2_rbf, multi_kernel_mmd2, multi_kernel_mmd2_and_grad_Yhat,
    median_sigma,
)
from .quantum import (
    ShotBankSpec, TrainableShotPauliBank, compute_features_chunked,
    parameter_shift_features_and_jacobian,
)
from .ridge import ridge_fit, ridge_inverse_matrix
from .train import (
    exact_loss_and_grad_theta, train_theta_exact, train_theta_spsa,
    AdamOpt,
)
from .decoder import SafeResidualGate, BlockModel
from .pipeline import (
    fit_all_blocks, sample_progressive, predict_future, rollout_refine_models,
)
from .calibration import EmpiricalGaussianCopulaCalibrator
from .metrics import correlation_error_summary

__all__ = [
    # config
    "QFANConfig", "TrainConfig", "DataConfig", "SketchConfig",
    "BankConfig", "ResidualConfig", "RolloutConfig", "CopulaConfig", "MMDConfig",
    "d12_config", "d25_config",
    # utils
    "corr_nan_safe", "_stable_sigmoid", "_power_of_two_floor",
    "_whiten", "_psd_sqrt", "_psd_invsqrt",
    # data
    "load_dataset", "select_subset_indices", "_resolve_subset_n",
    # sketch
    "OnlineCountSketch", "build_blocks", "build_prefix_sketch_cache",
    # mmd
    "mmd2_rbf", "multi_kernel_mmd2", "multi_kernel_mmd2_and_grad_Yhat",
    "median_sigma",
    # quantum
    "ShotBankSpec", "TrainableShotPauliBank", "compute_features_chunked",
    "parameter_shift_features_and_jacobian",
    # ridge
    "ridge_fit", "ridge_inverse_matrix",
    # train
    "exact_loss_and_grad_theta", "train_theta_exact", "train_theta_spsa",
    "AdamOpt",
    # decoder
    "SafeResidualGate", "BlockModel",
    # pipeline
    "fit_all_blocks", "sample_progressive", "predict_future",
    "rollout_refine_models",
    # calibration
    "EmpiricalGaussianCopulaCalibrator",
    # metrics
    "correlation_error_summary",
]
