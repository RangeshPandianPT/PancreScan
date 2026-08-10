import pytest
import sqlite3
import os
from src.ui.utils import database

# Fixture to use a test database
@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    test_db = "test_pancrescan.db"
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.init_db()
    yield
    if os.path.exists(test_db):
        os.remove(test_db)

def test_add_and_get_patient():
    pid, msg = database.add_patient("MRN123", "John Doe", 45, "Male")
    assert pid is not None
    
    patient = database.get_patient("MRN123")
    assert patient is not None
    assert patient["name"] == "John Doe"
    assert patient["age"] == 45

def test_duplicate_patient_mrn():
    database.add_patient("MRN123", "John Doe", 45, "Male")
    pid, msg = database.add_patient("MRN123", "Jane Doe", 30, "Female")
    assert pid is None
    assert "already exists" in msg

def test_add_scan():
    pid, _ = database.add_patient("MRN123", "John Doe", 45, "Male")
    scan_id = database.add_scan(pid, "scan1.jpg", "Tumor", 0.95, "UNet")
    assert scan_id is not None
    
    history = database.get_patient_history(pid)
    assert len(history) == 1
    assert history[0]["prediction"] == "Tumor"
    assert history[0]["confidence"] == 0.95
