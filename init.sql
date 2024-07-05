

USE audio_fingerprint;

CREATE TABLE songs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);

CREATE TABLE fingerprints (
    id INT AUTO_INCREMENT PRIMARY KEY,
    hash_value BIGINT NOT NULL,
    time_anchor INT NOT NULL,
    song_id INT,
    FOREIGN KEY (song_id) REFERENCES songs(id)
);
CREATE INDEX idx_hash_value ON fingerprints(hash_value);
