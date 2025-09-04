import numpy as np

def conv2fft(X, Y):
    # 计算卷积
    Convol = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(X) * np.fft.fft2(Y)))
    return Convol.real  # 取实部
