import sqlite3

def init_db():

    conn = sqlite3.connect("vehicle.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vehicle(
        user_id TEXT,
        vehicle_id TEXT,
        battery_type TEXT,
        buying_price REAL,
        buying_date TEXT,
        manufacture_date TEXT,
        PRIMARY KEY(vehicle_id)
    )
    """)

    conn.commit()
    conn.close()