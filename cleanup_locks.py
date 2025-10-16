# cleanup_locks.py
import os
import shutil
from pathlib import Path

def cleanup_rayonix_locks(data_dir="./rayonix_data"):
    """Clean up all lock files from previous runs"""
    data_path = Path(data_dir)
    
    lock_files = [
        data_path / "rayonix_db.lock",
        data_path / "blockchain_db" / "LOCK",
        data_path / "consensus_data" / "LOCK",
        data_path / "utxo_db" / "LOCK"
    ]
    
    for lock_file in lock_files:
        if lock_file.exists():
            try:
                lock_file.unlink()
                print(f"Removed lock file: {lock_file}")
            except Exception as e:
                print(f"Failed to remove {lock_file}: {e}")
    
    print("Lock cleanup completed")

if __name__ == "__main__":
    cleanup_rayonix_locks()