import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

# 创建一个示例二维数组 Y
Y = np.array([
    [0, 1, 2, 3, 2, 1, 0],
    [1, 2, 3, 4, 3, 2, 1],
    [2, 3, 4, 5, 4, 3, 2],
    [3, 4, 5, 6, 5, 4, 3],
    [2, 3, 4, 5, 4, 3, 2],
    [1, 2, 3, 4, 3, 2, 1],
    [0, 1, 2, 3, 2, 1, 0]
])

# 使用最大滤波器找到局部最大值
size = 3  # 选择一个较小的滤波器大小，便于观察
filtered = scipy.ndimage.maximum_filter(Y, size=size)

# 二值化峰值位置
peaks = np.where(Y == filtered, 1, 0)

print("原始数组 Y:")
print(Y)
print("\n应用最大滤波器后的数组 filtered:")
print(filtered)
print("\n二值化峰值位置后的数组 peaks:")
print(peaks)

# 可视化原始数组和峰值位置
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(Y, cmap='gray')
ax[0].set_title("原始数组 Y")
ax[1].imshow(peaks, cmap='gray')
ax[1].set_title("二值化峰值位置 peaks")
plt.show()
