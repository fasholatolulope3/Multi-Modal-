import sqlite3
import time

DB_FILE = "liveness_records.db"

def get_connection():
    """Returns an isolated sqlite3 connection for the thread."""
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS student_sessions (
            session_id TEXT PRIMARY KEY,
            last_updated REAL,
            movement_status TEXT,
            multiple_faces INTEGER,
            no_face INTEGER,
            warning TEXT
        )
    ''')
    try:
        c.execute("ALTER TABLE student_sessions ADD COLUMN student_name TEXT DEFAULT 'Unknown'")
    except sqlite3.OperationalError:
        pass
        
    try:
        c.execute("ALTER TABLE student_sessions ADD COLUMN matric_number TEXT DEFAULT 'Unknown'")
    except sqlite3.OperationalError:
        pass
        
    c.execute('''
        CREATE TABLE IF NOT EXISTS exam_submissions (
            session_id TEXT PRIMARY KEY,
            response TEXT,
            submitted_at REAL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS exam_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            question TEXT,
            updated_at REAL
        )
    ''')
    
    # Initialize with default question if empty
    c.execute("INSERT OR IGNORE INTO exam_config (id, question, updated_at) VALUES (1, 'Question 1: Explain the geopolitical implications of Active Gravity Control.', ?)", (time.time(),))
    
    conn.commit()
    conn.close()
