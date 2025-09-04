import numpy as np

def Measurement(I, SNR=40, Bits=16):
    # 添加噪声
    Noise = np.random.randn(*I.shape)
    Ampl = np.sqrt(np.sum(I ** 2) / np.sum(Noise ** 2)) / 10 ** (SNR / 20)
    Noise *= Ampl
    I_new = I + Noise

    # 量化
    Qmax = 2 ** Bits - 1
    I_new -= I_new.min()
    I_out = np.round(Qmax * I_new / I_new.max()).astype(np.uint16)
    
    return I_out
