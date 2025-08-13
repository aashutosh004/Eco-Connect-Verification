# modules/dbio.py
import os, time, threading, tempfile
import numpy as np

_db_lock = threading.Lock()

def save_dict_npy(path, obj):
    """Atomic write: write to temp then replace."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _db_lock:
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path))
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                np.save(f, obj)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)  # atomic on same filesystem
        finally:
            if os.path.exists(tmp_name):
                try: os.remove(tmp_name)
                except OSError: pass

def load_dict_npy(path, retries=2, delay=0.15):
    """Robust read with small retries if a writer just replaced the file."""
    if not os.path.exists(path) or (os.path.isfile(path) and os.path.getsize(path) < 8):
        return {}
    last_exc = None
    for _ in range(retries + 1):
        try:
            with _db_lock:
                return np.load(path, allow_pickle=True).item()
        except (EOFError, ValueError) as e:
            last_exc = e
            time.sleep(delay)
    # As a final safeguard, treat as empty DB instead of crashing the API
    return {}
