"""
Streaming count-sketch for autoregressive prefix compression.

Jamal Slim
jamal.slim@desy.de



Paper Sec. III-B, Eq. (2):
    s_j += sum_{k in block} sigma(k) * y_k * 1[h(k) == j]

Fixed random hash h : [d] -> [m] and sign sigma : [d] -> {+-1}. An optional
near-identity mixer M decorrelates hash collisions. Prefix-length
normalization and a mild nonlinearity (asinh by default) are applied before
the sketch is projected to circuit angles.
"""

import numpy as np


class OnlineCountSketch:
    def __init__(self, sketch_dim: int = 32, max_dim: int = 16384,
                 use_mixer: bool = True, seed: int = 7,
                 nonlinearity: str = "asinh", len_norm: bool = True):
        self.m = int(sketch_dim)
        self.max_dim = int(max_dim)
        self.use_mixer = bool(use_mixer)
        self.nonlinearity = str(nonlinearity).lower()
        self.len_norm = bool(len_norm)
        self.rng = np.random.default_rng(seed)

        self.h = self.rng.integers(0, self.m, size=self.max_dim, dtype=np.int64)
        self.sgn = self.rng.choice([-1.0, 1.0], size=self.max_dim).astype(np.float64)
        if self.use_mixer:
            self.M = (np.eye(self.m) +
                      0.01 * self.rng.normal(size=(self.m, self.m))).astype(np.float64)
        else:
            self.M = None

    def init_state(self, n_samples: int):
        return np.zeros((n_samples, self.m), dtype=np.float64), 0

    def mixed(self, Sraw: np.ndarray, cur_len: int = 0) -> np.ndarray:
        Z = np.asarray(Sraw, np.float64)
        if self.len_norm:
            Z = Z / np.sqrt(max(1, int(cur_len)))
        if self.use_mixer:
            Z = Z @ self.M.T
        if self.nonlinearity == "tanh":
            return np.tanh(Z)
        if self.nonlinearity == "asinh":
            return np.arcsinh(Z)
        return Z

    def update_inplace(self, Sraw: np.ndarray, cur_len: int, X_block: np.ndarray) -> int:
        X_block = np.asarray(X_block, dtype=np.float64)
        n, bsz = X_block.shape
        new_len = cur_len + bsz
        if new_len > self.max_dim:
            raise ValueError(f"Exceeded max_dim={self.max_dim}")
        cols = self.h[cur_len:new_len]
        signs = self.sgn[cur_len:new_len][None, :]
        vals = X_block * signs
        rows = np.repeat(np.arange(n, dtype=np.int64), bsz)
        cols_big = np.tile(cols, n).astype(np.int64, copy=False)
        np.add.at(Sraw, (rows, cols_big), vals.ravel())
        return new_len


def build_blocks(d: int, block_size: int):
    blocks, s = [], 0
    while s < d:
        b = min(block_size, d - s)
        blocks.append((s, b))
        s += b
    return blocks


def build_prefix_sketch_cache(Y_train: np.ndarray, blocks, sketcher: OnlineCountSketch) -> np.ndarray:
    """
    Pre-compute the teacher-forced sketch for every block on every training
    row: cache[beta, i, :] is the sketch of y_{<beta,i}. Shape (B, N, m).
    This cache is O(N d) to build and is re-used across every SPSA/Adam step,
    matching paper Sec. IV.
    """
    Y_train = np.asarray(Y_train, np.float64)
    n, _ = Y_train.shape
    nb = len(blocks)
    Sraw, cur_len = sketcher.init_state(n)
    cache = np.zeros((nb, n, sketcher.m), dtype=np.float32)
    for bi, (start, bsz) in enumerate(blocks):
        cache[bi] = sketcher.mixed(Sraw, cur_len).astype(np.float32, copy=False)
        cur_len = sketcher.update_inplace(Sraw, cur_len, Y_train[:, start:start + bsz])
    return cache
