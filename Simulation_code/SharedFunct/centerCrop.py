import numpy as np

def centerCrop(M, RC):
    # RC = [r, c] 表示要裁剪的行数和列数
    center = np.ceil(np.array(M.shape) / 2).astype(int)
    # 裁剪矩阵 M，从中心裁剪指定的行和列
    C = M[center[0] - RC[0]: center[0] + RC[0], center[1] - RC[1]: center[1] + RC[1]]
    return C

