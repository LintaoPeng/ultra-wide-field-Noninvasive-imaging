import numpy as np

def SpeckleGen(Nr, Nc, SpeckleSize, coherent):
    # 创建网格
    r, c = np.meshgrid(np.arange(-Nc, Nc + 1), np.arange(-Nr, Nr + 1))

    R2 = r ** 2 + c ** 2
    Laser = np.exp(-(R2 / ((Nc + Nr) / 2 / SpeckleSize) ** 2))  # 激光输入：高斯光束

    # 生成散射相位
    ScatPhase = 2 * np.pi * np.random.rand(*Laser.shape)
    
    # 计算傅里叶变换
    S = np.fft.fft2(np.fft.fftshift(Laser * np.exp(1j * ScatPhase)))
    S = S / np.linalg.norm(S.ravel())  # 归一化

    # 检查相干性
    if coherent.upper() == "INCOHERENT":
        S = np.abs(S) ** 2  # 非相干PSF
        S = S / np.sum(S)  # 归一化
    elif coherent.upper() != "COHERENT":
        raise ValueError('TYPO: Coherent or Incoherent light?')
        
    return S
