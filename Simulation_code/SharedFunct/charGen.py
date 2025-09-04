import numpy as np
import cv2

def charGen(ch, row, col, offSet, Textsize, font=cv2.FONT_HERSHEY_SIMPLEX):
    # 初始化空白图像
    img = np.zeros((row, col), dtype=np.uint8)
    
    # 计算字符位置
    pos = (int(col / 2) + offSet[0], int(row / 2) + offSet[1])

    # 添加文字到图像
    cv2.putText(img, ch, pos, font, Textsize / 50, (255,), thickness=1, lineType=cv2.LINE_AA)

    # 转换为灰度图像（已经是单通道，无需额外处理）
    return img
