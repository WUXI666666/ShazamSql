import librosa as lr
import numpy as np
import matplotlib.pyplot as plt
import scipy
import librosa
from scipy import ndimage
def createfingerprint(x, plot=False):
    # 生成频谱图并转换为对数刻度
    X = lr.stft(x, n_fft=1024, hop_length=100, window="blackman")
    X = np.abs(X)
    L, W = np.shape(X)

    # 提取峰值并使用最大滤波器选择邻域中的最高峰
    output = peakextract(X)
    output = scipy.ndimage.maximum_filter(output, size=25)
    max_peak = np.max(output)
    output = np.where(output == 0 , -1 , output)
    output = np.where(X == output, 1, 0)

    # 如果启用，显示包含原始频谱图的指纹图像
    if plot:
        plt.imshow(X)
        y_ind, x_ind = np.where(output != 0)
        plt.scatter(x=x_ind, y=y_ind, c='r', s=8.0)
        plt.gca().invert_yaxis()
        plt.xlabel('Frames')
        plt.ylabel('Bins')
        plt.draw()

    return output
# 创建指纹的函数
# def compute_constellation_map(Y, dist_freq=7, dist_time=7, thresh=0.01):
#     result = ndimage.maximum_filter(Y, size=[2*dist_freq+1, 2*dist_time+1], mode='constant')
#     Cmap = np.logical_and(Y == result, result > thresh)
#     return Cmap
# def createfingerprint(x, plot=False):
#     # 生成频谱图并转换为对数刻度
#     X = librosa.stft(x, n_fft=1024, hop_length=32, window="blackman")
#     X = np.abs(X)
#     L, W = np.shape(X)

#     # 提取峰值
#     output = compute_constellation_map(X, dist_freq=7, dist_time=7, thresh=0.01)

#     # 如果启用，显示包含原始频谱图的指纹图像
#     if plot:
#         plt.imshow(np.log1p(X), origin='lower', aspect='auto', cmap='gray_r', interpolation='nearest')
#         y_ind, x_ind = np.where(output != 0)
#         plt.scatter(x=x_ind, y=y_ind, c='r', s=8.0)
#         plt.gca().invert_yaxis()
#         plt.xlabel('Time (frames)')
#         plt.ylabel('Frequency (bins)')
#         plt.title('Constellation Map')
#         plt.show()

#     return output

def peakextract(S):
    # 初始化峰值矩阵
    row, col = np.shape(S)
    peaks = np.zeros((row, col))

    # 将频率谱分为对数频段
    bands = np.array([[1, 11], [12, 21], [22, 41], [42, 81], [82, 161], [162, 513]])

    # 在每个频段中的每个时间帧中找到最大值并更新峰值矩阵
    for i in range(col):
        for j in range(6):
            q1 = bands[j, 0]
            q2 = bands[j, 1]
            frame_band = S[q1:q2, i]
            localpeak = np.max(frame_band)
            index = np.where(frame_band == localpeak)
            peaks[q1 + index[0], i] = localpeak

    return peaks

def createhashes(peaks: np.ndarray, song_id: int, sample_rate: int = 8192, hop_length: int = 410) -> np.ndarray:
    peaks = np.transpose(peaks)
    points = np.where(peaks != 0)
    num_points = np.shape(points[0])[0]
    hash_matrix = []

    max_delta_time = 16000  # 最大时间差，14位二进制数的最大值为16383

    # 计算时间刻度
    
    for i in range(num_points):
        for j in range(1, min(16, num_points - i)):  # 限制邻域大小为7
            freq_anchor = points[1][i]
            freq_other = points[1][i+j]
            delta_time_frames = abs(points[0][i] - points[0][i+j])
            time_anchor = points[0][i]


            # 只在时间差小于等于 max_delta_time 时添加到哈希表
            if delta_time_frames <= max_delta_time:
                # 将频率锚点、目标频率和时间差编码为一个32位整数
                hash_value = (freq_anchor << 23) | (freq_other << 14) | delta_time_frames
                hash_matrix.append([hash_value, time_anchor, song_id])

    hash_matrix = np.array(hash_matrix)
    return hash_matrix







