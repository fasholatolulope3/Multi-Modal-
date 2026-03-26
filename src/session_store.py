import time
import logging
import sqlite3
from src.database import get_connection

logger = logging.getLogger(__name__)

def update_student_telemetry(student_id, student_name, matric_number, data):
    try:
        conn = get_connection()
        c = conn.cursor()
        
        # INSERT OR REPLACE handles both new connections and updates based on the PRIMARY KEY
        c.execute('''
            INSERT OR REPLACE INTO student_sessions 
            (session_id, last_updated, movement_status, multiple_faces, no_face, warning, student_name, matric_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            student_id,
            time.time(),
            data.get('movement_status', 'Unknown'),
            1 if data.get('multiple_faces') else 0,
            1 if data.get('no_face') else 0,
            data.get('warning', ''),
            student_name,
            matric_number
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error writing to database: {e}")

def get_all_students_telemetry():
    result = {}
    try:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM student_sessions")
        rows = c.fetchall()
        
        for row in rows:
            result[row['session_id']] = {
                "student_name": row['student_name'] if 'student_name' in row.keys() else 'Unknown',
                "matric_number": row['matric_number'] if 'matric_number' in row.keys() else 'Unknown',
                "last_updated": row['last_updated'],
                "telemetry": {
                    "movement_status": row['movement_status'],
                    "multiple_faces": bool(row['multiple_faces']),
                    "no_face": bool(row['no_face']),
                    "warning": row['warning']
                }
            }
        conn.close()
    except Exception as e:
        logger.error(f"Error reading from database: {e}")
        
    return result

def submit_exam_response(student_id, student_name, matric_number, response):
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO exam_submissions
            (session_id, response, submitted_at)
            VALUES (?, ?, ?)
        ''', (student_id, response, time.time()))
        
        # Ensure student is visible in Admin telemetry loop even if webcam failed
        c.execute('''
            INSERT OR IGNORE INTO student_sessions
            (session_id, last_updated, movement_status, multiple_faces, no_face, warning, student_name, matric_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (student_id, time.time(), "Exam Submitted", 0, 0, "No Video Telemetry Logged", student_name, matric_number))
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error submitting exam to database: {e}")

def get_exam_submission(student_id):
    try:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT response, submitted_at FROM exam_submissions WHERE session_id = ?", (student_id,))
        row = c.fetchone()
        conn.close()
        if row:
            return {"response": row["response"], "submitted_at": row["submitted_at"]}
    except Exception as e:
        logger.error(f"Error reading exam submission from database: {e}")
    return None

def delete_student_record(student_id):
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("DELETE FROM student_sessions WHERE session_id = ?", (student_id,))
        c.execute("DELETE FROM exam_submissions WHERE session_id = ?", (student_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error deleting record from database: {e}")

def get_exam_question():
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT question FROM exam_config WHERE id = 1")
        row = c.fetchone()
        conn.close()
        return row[0] if row else "No question set."
    except Exception as e:
        logger.error(f"Error reading exam config from database: {e}")
    return "Error loading question from database."

def set_exam_question(question):
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("UPDATE exam_config SET question = ?, updated_at = ? WHERE id = 1", (question, time.time()))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error writing exam config to database: {e}")

