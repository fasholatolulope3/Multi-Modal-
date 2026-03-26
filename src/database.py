import sqlite3

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
    conn.commit()
    conn.close()
