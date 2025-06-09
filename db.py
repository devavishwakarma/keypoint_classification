import sqlite3

def create_connection():
    conn = sqlite3.connect('users.db', check_same_thread=False)
    return conn

def create_user_table():
    conn = create_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL,
                    email TEXT,
                    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

def add_user(username, password, email):
    conn = create_connection()
    c = conn.cursor()
    c.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)', (username, password, email))
    conn.commit()
    conn.close()

def authenticate_user(username, password):
    conn = create_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
    data = c.fetchone()
    conn.close()
    return data
