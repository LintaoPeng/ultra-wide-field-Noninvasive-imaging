import numpy as np
import cupy as cp  # 如果使用GPU则使用cupy，否则用numpy

def BasicPhaseRetrieval(sautocorr_temp, beta_start, beta_step, beta_stop, N_iter, init_guess, useGPU):
    # 选择适当的数组库
    xp = cp if useGPU else np
    
    # 初始化变量
    sautocorr = xp.array(sautocorr_temp, dtype=xp.float32)
    g1 = xp.array(init_guess, dtype=xp.float32)
    
    # 初始化误差记录
    recons_err = []
    BETAS = xp.arange(beta_start, beta_stop + beta_step, beta_step)
    
    # 主要迭代部分
    ii = 0
    for beta in BETAS:
        for _ in range(N_iter):
            ii += 1
            G_uv = xp.fft.fft2(g1)
            g1_tag = xp.real(xp.fft.ifft2(sautocorr * xp.exp(1j * xp.angle(G_uv))))
            g1 = g1_tag * (g1_tag >= 0) + (g1_tag < 0) * (g1 - beta * g1_tag)
            recons_err.append(xp.mean((xp.abs(xp.fft.fft2(g1)) - sautocorr) ** 2).get() if useGPU else xp.mean((xp.abs(xp.fft.fft2(g1)) - sautocorr) ** 2))

    # 错误减少部分
    for _ in range(N_iter):
        ii += 1
        G_uv = xp.fft.fft2(g1)
        g1_tag = xp.real(xp.fft.ifft2(sautocorr * xp.exp(1j * xp.angle(G_uv))))
        g1 = g1_tag * (g1_tag >= 0)
        recons_err.append(xp.mean((xp.abs(xp.fft.fft2(g1)) - sautocorr) ** 2).get() if useGPU else xp.mean((xp.abs(xp.fft.fft2(g1)) - sautocorr) ** 2))

    return (g1.get() if useGPU else g1), recons_err


