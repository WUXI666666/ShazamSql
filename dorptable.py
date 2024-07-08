import mysql.connector
def drop_table():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='123456',
        database='audio_fingerprint'
    )
    cursor = conn.cursor()

    cursor.execute("drop table audio_fingerprint.fingerprints")
    cursor.execute("drop table audio_fingerprint.songs")

    conn.commit()
    cursor.close()
    conn.close()
drop_table()
