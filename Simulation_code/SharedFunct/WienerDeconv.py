import numpy as np
from scipy.signal import tukey
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def WienerDeconv(I, PSF, k=0.01, HF_cut=0, Th=0.01):

    # 计算傅里叶变换
    I_fft = fft2(I)
    PSF_fft = fft2(PSF)

    C = (np.mean(PSF) * k) ** 2
    O_fft = fftshift(I_fft * np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + C))

    N1, N2 = O_fft.shape

    # 生成Tukey窗口
    r1 = HF_cut / N1
    W1 = tukey(N1, r1)
    r2 = HF_cut / N2
    W2 = tukey(N2, r2)
    
    #LowPass_Filter = np.kron(W1, W2.T)
    LowPass_Filter = np.kron(W1[:, np.newaxis], W2[np.newaxis, :])  # Kronecker 乘积
    print(LowPass_Filter.shape)

    # 反傅里叶变换并进行低通滤波
    O = abs(fftshift(ifft2(ifftshift(O_fft * LowPass_Filter))))

    # 归一化
    O /= np.max(O)
    
    # 应用阈值
    val = np.max(O)
    O = O / val - Th
    O[O < 0] = 0
    O = val * O / (1 - Th)

    return O
