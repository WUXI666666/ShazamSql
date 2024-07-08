from genericpath import isfile
import pickle
import numpy as np
import os
import librosa
import librosa.display
from fingerprint import createfingerprint, createhashes
import glob
from typing import List
from save_to_mysql import save_to_mysql
# 创建反向索引文件
def create_database(directory: str = './songs', extensions: List[str] = ['*.wav']) -> None:
    server_tables = []
    song_names = []
    files = []

    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))

    for i, file in enumerate(files):
        song_name = os.path.basename(file)
        song_names.append(song_name)

        x, fs = librosa.load(file, sr=22050, mono=True)
        F_print = createfingerprint(x)
        hash_matrix = createhashes(F_print, song_id=i)

        server_tables.append(hash_matrix)
        print(f"Processed {i+1}/{len(files)}: {song_name}")

    reverse_index = create_reverse_index(server_tables)
    save_reverse_index_to_file(reverse_index, 'reverse_index.pkl')
    save_song_names_to_file(song_names, 'song_names.pkl')

def create_reverse_index(server_tables: List[np.ndarray]) -> dict:
    reverse_index = {}
    for song_id, table in enumerate(server_tables):
        for hash_value, time_anchor, _ in table:
            if hash_value not in reverse_index:
                reverse_index[hash_value] = []
            reverse_index[hash_value].append((hash_value,song_id, time_anchor))
    return reverse_index

def save_reverse_index_to_file(reverse_index: dict, file_path: str) -> None:
    with open(file_path, 'wb') as f:
        pickle.dump(reverse_index, f)

def save_song_names_to_file(song_names: List[str], file_path: str) -> None:
    with open(file_path, 'wb') as f:
        pickle.dump(song_names, f)

def load_reverse_index_from_file(file_path: str) -> dict:
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_song_names_from_file(file_path: str) -> List[str]:
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# 创建mysql数据库
# def create_database(directory: str = './songs', extensions: List[str] = ['*.wav']) -> None:
#     server_tables = []
#     song_names = []
#     files = []

#     for ext in extensions:
#         files.extend(glob.glob(os.path.join(directory, ext)))

#     for i, file in enumerate(files):
#         song_name = os.path.basename(file)
#         song_names.append(song_name)

#         x, fs = librosa.load(file, sr=22050, mono=True)
#         F_print = createfingerprint(x)
#         hash_matrix = createhashes(F_print, song_id=i)

#         server_tables.append(hash_matrix)
#         print(f"Processed {i+1}/{len(files)}: {song_name}")

#     save_to_mysql(server_tables, song_names)
if __name__=="__main__":
    create_database()
