import numpy as np

def centerImg(input):
    # 找到图像中最亮点的位置
    BrightRow = np.argmax(np.max(input, axis=1))
    BrightCol = np.argmax(input[BrightRow])

    # 获取图像的行数和列数
    Row, Col = input.shape

    # 计算将最亮点移到中心的位置偏移量
    shift_row = int(np.floor(Row / 2 - BrightRow))
    shift_col = int(np.floor(Col / 2 - BrightCol))

    # 使用 np.roll 实现循环移位
    output = np.roll(input, shift=(shift_row, shift_col), axis=(0, 1))
    
    return output

