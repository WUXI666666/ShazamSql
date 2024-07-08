from genericpath import isfile
from typing import List, Tuple
import numpy as np
import os
import librosa as lr
import librosa.display
import matplotlib.pyplot as plt
from scipy import ndimage
from fingerprint import createfingerprint, createhashes
import mysql.connector
from record import recordaudio

# 指纹提取参数
dist_freq = 11
dist_time = 7
tol_freq = 1
tol_time = 1

def compute_constellation_map(Y, dist_freq=7, dist_time=7, thresh=0.01):
    result = ndimage.maximum_filter(Y, size=[2*dist_freq+1, 2*dist_time+1], mode='constant')
    Cmap = np.logical_and(Y == result, result > thresh)
    return Cmap

def plot_constellation_map(Cmap, Y=None, xlim=None, ylim=None, title='',
                           xlabel='Time (sample)', ylabel='Frequency (bins)',
                           s=5, color='r', marker='o', figsize=(7, 3), dpi=72):
    if Cmap.ndim > 1:
        (K, N) = Cmap.shape
    else:
        K = Cmap.shape[0]
        N = 1
    if Y is None:
        Y = np.zeros((K, N))
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    im = ax.imshow(Y, origin='lower', aspect='auto', cmap='gray_r', interpolation='nearest')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    Fs = 1
    if xlim is None:
        xlim = [-0.5/Fs, (N-0.5)/Fs]
    if ylim is None:
        ylim = [-0.5/Fs, (K-0.5)/Fs]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    n, k = np.argwhere(Cmap == 1).T
    ax.scatter(k, n, color=color, s=s, marker=marker)
    plt.tight_layout()
    return fig, ax, im

def compute_spectrogram(audio_path, Fs=22050, N=2048, H=1024, bin_max=128, frame_max=None):
    x, Fs = librosa.load(audio_path, sr=Fs)
    x_duration = len(x) / Fs
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann')
    if bin_max is None:
        bin_max = X.shape[0]
    if frame_max is None:
        frame_max = X.shape[0]
    Y = np.abs(X[:bin_max, :frame_max])
    return Y

def match_binary_matrices_tol(C_ref, C_est, tol_freq=0, tol_time=0):
    assert C_ref.shape == C_est.shape, "Dimensions need to agree"
    N = np.sum(C_ref)
    M = np.sum(C_est)
    C_est_max = ndimage.maximum_filter(C_est, size=(2*tol_freq+1, 2*tol_time+1), mode='constant')
    C_AND = np.logical_and(C_est_max, C_ref)
    TP = np.sum(C_AND)
    FN = N - TP
    FP = M - TP
    return TP, FN, FP, C_AND

def compute_matching_function(C_D, C_Q, tol_freq=1, tol_time=1):
    L = C_D.shape[1]
    N = C_Q.shape[1]
    M = L - N
    assert M >= 0, "Query must be shorter than document"
    Delta = np.zeros(L)
    for m in range(M + 1):
        C_D_crop = C_D[:, m:m+N]
        TP, FN, FP, C_AND = match_binary_matrices_tol(C_D_crop, C_Q, tol_freq=tol_freq, tol_time=tol_time)
        Delta[m] = TP
    shift_max = np.argmax(Delta)
    return Delta, shift_max

def fetchID_from_mysql(hash_matrix: np.ndarray) -> List[Tuple[str, float, float]]:
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='123456',
        database='audio_fingerprint'
    )
    cursor = conn.cursor()

    # 收集所有要查询的哈希值
    hash_values = [int(song_hash[0]) for song_hash in hash_matrix]
    hash_values_str = ','.join(map(str, hash_values))

    # 批量查询哈希值
    query = f"SELECT hash_value, song_id, time_anchor FROM fingerprints WHERE hash_value IN ({hash_values_str})"
    cursor.execute(query)
    matched_pairs_list = cursor.fetchall()

    if not matched_pairs_list:
        cursor.close()
        conn.close()
        return [("Not found!", 0, 0)]

    num_pairs = len(hash_matrix)

    matched_pairs_by_song = {}
    for hash_value, song_id, time_anchor in matched_pairs_list:
        if song_id not in matched_pairs_by_song:
            matched_pairs_by_song[song_id] = set()
        matched_pairs_by_song[song_id].add(hash_value)

    ratios = []
    for song_id, unique_hashes in matched_pairs_by_song.items():
        ratio = len(unique_hashes) / num_pairs  # 使用唯一哈希数除以查询哈希数计算匹配率
        anchors = [time_anchor for hash_value, s_id, time_anchor in matched_pairs_list if s_id == song_id]
        if(anchors):
            max_anchor=max(anchors)+1
            bin_size=min(max_anchor,2050)
            hist_max = np.max(np.histogram(anchors, bins=range(0, max(anchors) + 1, bin_size-1))[0]) 
        else:
            hist_max=0
        ratios.append((song_id, ratio, hist_max))

    top_matches = sorted(ratios, key=lambda x: (-x[1], -x[2]))[:3]

    song_names = {}
    for song_id, _, _ in top_matches:
        cursor.execute("SELECT name FROM songs WHERE id = %s", (song_id,))
        song_name = cursor.fetchone()[0]
        song_names[song_id] = song_name

    cursor.close()
    conn.close()

    top_match_info = [(song_names[song_id], ratio, hist_max) for song_id, ratio, hist_max in top_matches]

    return top_match_info






def recognize_song_from_path(query_path: str) -> None:
    try:
        x, fs = lr.load(query_path, sr=22050, mono=True)
        F_print = createfingerprint(x,False)
        song_id = 0
        hash_matrix = createhashes(F_print, song_id=song_id)
        top_matches = fetchID_from_mysql(hash_matrix)

        print('Top 3 Matches:')
        print('Song Name\t\tRatio\t\tHist max')
        for match in top_matches:
            song_name, ratio, hist = match
            print(f"{song_name}\t\t{ratio:.4f}\t\t{hist:.4f}")

        best_match = top_matches[0][0]
        print(f"\nMost matched song: {best_match}")
    except Exception as e:
        print(f"Error: {e}")

def recognize_song(x: np.ndarray) -> None:
    try:
        F_print = createfingerprint(x)
        song_id = 0
        hash_matrix = createhashes(F_print, song_id=song_id)
        top_matches = fetchID_from_mysql(hash_matrix)

        print('Top 3 Matches:')
        print('Song Name\t\tRatio\t\tHist max')
        for match in top_matches:
            song_name, ratio, hist = match
            print(f"{song_name}\t\t{ratio:.4f}\t\t{hist:.4f}")

        best_match = top_matches[0][0]
        print(f"\nMost matched song: {best_match}")
    except Exception as e:
        print(f"Error: {e}")

def compare_2songs(path1, path2):
    Y1 = compute_spectrogram(path1)
    Y2 = compute_spectrogram(path2)
    CM1 = compute_constellation_map(Y1, dist_freq, dist_time)
    CM2 = compute_constellation_map(Y2, dist_freq, dist_time)
    Delta, shift_max = compute_matching_function(CM1, CM2, tol_freq=tol_freq, tol_time=tol_time)
    print(Delta[shift_max])
    plot_constellation_map(CM1, np.log(1 + 1 * Y1), color='r', s=30, title=path1)
    plot_constellation_map(CM2, np.log(1 + 1 * Y2), color='r', s=30, title=path2)

def compare_dir(path, fn_query):
    Y_q = compute_spectrogram(fn_query)
    CMP_q = compute_constellation_map(Y_q, dist_freq, dist_time)
    for fn in os.listdir(path):
        if os.path.isfile(os.path.join(path, fn)):
            if fn.endswith(".wav"):
                fn = os.path.join(path, fn)
                print(fn)
                Y_d = compute_spectrogram(fn)
                CMP_d = compute_constellation_map(Y_d, dist_freq, dist_time)
                Delta, shift_max = compute_matching_function(CMP_d, CMP_q, tol_freq=0, tol_time=0)
                print(Delta[shift_max])
                plot_constellation_map(CMP_d, np.log(1 + 1 * Y_d), color='r', s=30, title=fn)

recognize_song_from_path("./tests/test_1_1.wav")

# recognize_song(recordaudio())
plt.show()  
