import mysql.connector
from typing import List
import numpy as np
def save_to_mysql(server_tables: List[np.ndarray], song_names: List[str]) -> None:
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='a15182757117!',
        database='audio_fingerprint'
    )
    cursor = conn.cursor()

    cursor.execute("DELETE FROM fingerprints")
    cursor.execute("DELETE FROM songs")

    for i, song_name in enumerate(song_names):
        cursor.execute("INSERT INTO songs (name) VALUES (%s)", (song_name,))
        song_id = cursor.lastrowid

        for hash_value, time_anchor, _ in server_tables[i]:
            cursor.execute(
                "INSERT INTO fingerprints (hash_value, time_anchor, song_id) VALUES (%s, %s, %s)",
                (int(hash_value), int(time_anchor), int(song_id))
            )

    conn.commit()
    cursor.close()
    conn.close()