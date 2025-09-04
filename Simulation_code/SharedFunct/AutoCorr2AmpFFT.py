import numpy as np

def AutoCorr2AmpFFT(AutoCorr):
    # 计算2D逆傅里叶变换并取绝对值的平方根
    AmpFFT = np.sqrt(np.abs(np.fft.fftshift(np.fft.ifft2(AutoCorr))))
    return AmpFFT

