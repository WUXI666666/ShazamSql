from genericpath import isfile
import numpy as np
import os
import librosa
import librosa.display
from fingerprint import createfingerprint, createhashes
import glob
from typing import List
from save_to_mysql import save_to_mysql

# 创建数据库
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

    save_to_mysql(server_tables, song_names)

# Example usage
create_database()
