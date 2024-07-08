import mysql.connector
from typing import List
import numpy as np

def save_to_mysql(server_tables: List[np.ndarray], song_names: List[str]) -> None:
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='123456',
        database='audio_fingerprint'
    )
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS songs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fingerprints (
        id INT AUTO_INCREMENT PRIMARY KEY,
        hash_value BIGINT NOT NULL,
        time_anchor INT NOT NULL,
        song_id INT,
        FOREIGN KEY (song_id) REFERENCES songs(id)
    )
    """)

    cursor.execute("CREATE INDEX  idx_hash_value ON fingerprints(hash_value)")

    for i, song_name in enumerate(song_names):
        cursor.execute("INSERT INTO songs (name) VALUES (%s)", (song_name,))
        song_id = cursor.lastrowid

        fingerprints_data = [
            (int(hash_value), int(time_anchor), int(song_id))
            for hash_value, time_anchor, _ in server_tables[i]
        ]

        cursor.executemany(
            "INSERT INTO fingerprints (hash_value, time_anchor, song_id) VALUES (%s, %s, %s)",
            fingerprints_data
        )

    conn.commit()
    cursor.close()
    conn.close()
