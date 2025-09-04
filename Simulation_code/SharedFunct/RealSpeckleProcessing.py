import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def RealSpeckleProcessing(I_in, Filter_Length=15):
    I_in = I_in.astype(np.float64)
    N_Speckle_1, N_Speckle_2 = I_in.shape

    # 开始滤波
    Large_Speckle_intensity_FFT = fftshift(fft2(I_in))
    LowPass_Filter = np.zeros((N_Speckle_1, N_Speckle_2))

    # 创建低通滤波器
    LowPass_Filter[(N_Speckle_1 // 2 - Filter_Length):(N_Speckle_1 // 2 + Filter_Length),
                   (N_Speckle_2 // 2 - Filter_Length):(N_Speckle_2 // 2 + Filter_Length)] = 1

    # 应用低通滤波器
    Large_Speckle_intensity_FFT *= LowPass_Filter
    Large_Speckle_intensity_LowPassVer = abs(ifft2(ifftshift(Large_Speckle_intensity_FFT)))

    # 归一化
    Large_Speckle_intensity_LowPassVer /= np.max(Large_Speckle_intensity_LowPassVer)

    # 计算新的强度图像
    Large_Speckle_intensity_New = I_in / Large_Speckle_intensity_LowPassVer

    # 应用高斯滤波
    I_out = gaussian_filter(Large_Speckle_intensity_New, sigma=0.5)

    return I_out
