import sqlite3
from datetime import datetime
import os

DB_NAME = "pancrescan.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create Patients Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mrn TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create Scans Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            model_used TEXT,
            scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def add_patient(mrn, name, age, gender):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "INSERT INTO patients (mrn, name, age, gender) VALUES (?, ?, ?, ?)",
            (mrn, name, age, gender)
        )
        conn.commit()
        patient_id = c.lastrowid
        conn.close()
        return patient_id, "Success"
    except sqlite3.IntegrityError:
        return None, "Patient with this MRN already exists."
    except Exception as e:
        return None, str(e)

def get_patient(mrn):
    conn = get_db_connection()
    c = conn.cursor()
    patient = c.execute("SELECT * FROM patients WHERE mrn = ?", (mrn,)).fetchone()
    conn.close()
    return patient

def get_all_patients():
    conn = get_db_connection()
    c = conn.cursor()
    patients = c.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
    conn.close()
    return patients

def add_scan(patient_id, filename, prediction, confidence, model_used):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO scans (patient_id, filename, prediction, confidence, model_used) VALUES (?, ?, ?, ?, ?)",
        (patient_id, filename, prediction, confidence, model_used)
    )
    conn.commit()
    scan_id = c.lastrowid
    conn.close()
    return scan_id

def get_patient_history(patient_id):
    conn = get_db_connection()
    c = conn.cursor()
    scans = c.execute("SELECT * FROM scans WHERE patient_id = ? ORDER BY scan_date DESC", (patient_id,)).fetchall()
    conn.close()
    return scans

# Initialize DB on module import
init_db()
