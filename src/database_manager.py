import sqlite3
from datetime import datetime


DB_PATH = "../database/security.db"


def create_database():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            tracking_id INTEGER,
            timestamp TEXT,
            image_path TEXT
        )
    """)

    conn.commit()
    conn.close()

    print("Database and alerts table created successfully!")


def save_alert(event_type, tracking_id, image_path):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO alerts (
            event_type,
            tracking_id,
            timestamp,
            image_path
        )
        VALUES (?, ?, ?, ?)
    """, (event_type, tracking_id, timestamp, image_path))

    conn.commit()
    conn.close()

    print("Alert saved to database!")


create_database()