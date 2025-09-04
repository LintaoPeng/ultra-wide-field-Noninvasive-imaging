import numpy as np

def autoCorr(X):
    # 计算2D傅里叶变换
    F = np.fft.fft2(X)
    # 计算自相关函数
    Au = np.fft.fftshift(np.fft.ifft2(np.abs(F) ** 2))
    return Au.real  # 取实部
