import numpy as np

def loadMNISTImages(filename, n):
    # 读取MNIST图像文件
    with open(filename, 'rb') as fp:
        # 检查文件的 "magic number" 以确保文件格式正确
        magic = int(np.frombuffer(fp.read(4), dtype=np.dtype('>i4')))
        assert magic == 2051, f'Bad magic number in {filename}'

        # 读取图像的数量、行数和列数
        numImages = int(np.frombuffer(fp.read(4), dtype=np.dtype('>i4')))
        numRows = int(np.frombuffer(fp.read(4), dtype=np.dtype('>i4')))
        numCols = int(np.frombuffer(fp.read(4), dtype=np.dtype('>i4')))

        # 读取图像数据并将其转换为numpy数组
        images = np.frombuffer(fp.read(), dtype=np.uint8)
        images = images.reshape(numImages, numRows, numCols)

    # 转置图像以与MATLAB的格式一致，并提取第n张图像
    #image = images[n - 1].T  # Python索引从0开始，因此n-1
    image = images[n - 1]
    return image
