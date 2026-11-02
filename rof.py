"""
ROF 全變差去噪（Rudin–Osher–Fatemi, 1992），以 primal-dual 迭代求解。

    min_u  weight · TV(u) + ½‖u − f‖²

三種 TV 正則項（mode）：
    'scalar'     逐通道 TV（灰階也用這個）：  Σ_c ∫ |∇u_c| dx
    'vectorial'  向量 TV（VTV），通道耦合：   ∫ sqrt(Σ_c |∇u_c|²) dx

傳入 NumPy 陣列就在 CPU 計算，傳入 CuPy 陣列就在 GPU 計算，程式碼相同。
"""
import warnings

import numpy as np

try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import cupy as cp
        cp.zeros(1).sum()    # 確認 GPU 與 CUDA 真的能用
except Exception:            # 沒有 CuPy、沒有 NVIDIA GPU 或 CUDA 不完整時只用 NumPy
    cp = None


def get_xp(x):
    """回傳 x 所屬的陣列模組（numpy 或 cupy）。"""
    return cp.get_array_module(x) if cp is not None else np


def grad(u):
    """前向差分梯度。u: (H, W, C) → (H, W, 2, C)，第 3 維為 (∂y, ∂x)。"""
    xp = get_xp(u)
    g = xp.zeros((u.shape[0], u.shape[1], 2, u.shape[2]), dtype=u.dtype)
    g[:-1, :, 0, :] = u[1:, :, :] - u[:-1, :, :]
    g[:, :-1, 1, :] = u[:, 1:, :] - u[:, :-1, :]
    return g


def div(p):
    """散度，為 grad 的負伴隨算子（−div = grad*）。p: (H, W, 2, C) → (H, W, C)。"""
    xp = get_xp(p)
    d = xp.zeros((p.shape[0], p.shape[1], p.shape[3]), dtype=p.dtype)
    d[1:-1, :, :] += p[1:-1, :, 0, :] - p[:-2, :, 0, :]
    d[0, :, :] += p[0, :, 0, :]
    d[-1, :, :] -= p[-2, :, 0, :]
    d[:, 1:-1, :] += p[:, 1:-1, 1, :] - p[:, :-2, 1, :]
    d[:, 0, :] += p[:, 0, 1, :]
    d[:, -1, :] -= p[:, -2, 1, :]
    return d


def rof_denoise(f, weight=0.1, n_iter=100, mode='scalar', tau=0.125, sigma=0.125):
    """
    ROF 去噪。

    f       觀測影像，(H, W) 灰階或 (H, W, C) 彩色，數值範圍 [0, 1]
    weight  TV 權重 λ，越大越平滑
    mode    'scalar'（逐通道）或 'vectorial'（VTV）
    tau, sigma  primal / dual 步長；需滿足 tau·sigma·‖∇‖² ≤ 1（‖∇‖² ≤ 8）

    每次迭代：
      1. dual：  p ← proj_{|p| ≤ weight}(p + σ∇u)
                 scalar 對每個通道各自投影；vectorial 把所有通道的梯度一起投影
      2. primal：u ← (u + τ·div p + τ·f) / (1 + τ)
    """
    if mode not in ('scalar', 'vectorial'):
        raise ValueError("mode 必須是 'scalar' 或 'vectorial'")
    xp = get_xp(f)
    gray = f.ndim == 2
    f = f[:, :, None] if gray else f

    u = f.copy()
    p = xp.zeros((f.shape[0], f.shape[1], 2, f.shape[2]), dtype=f.dtype)
    norm_axes = 2 if mode == 'scalar' else (2, 3)

    for _ in range(n_iter):
        p_new = p + sigma * grad(u)
        norm = xp.sqrt(xp.sum(p_new ** 2, axis=norm_axes, keepdims=True))
        p = p_new / xp.maximum(1.0, norm / weight)
        u = (u + tau * div(p) + tau * f) / (1.0 + tau)

    return u[:, :, 0] if gray else u
