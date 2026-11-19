"""Small helpers: logging, atomic JSON, deterministic seeds, the session deadline."""
import datetime
import hashlib
import json
import math
import os
import time

EXIT_PAUSED = 75          # a worker stopped because the session deadline is near

_LOG_FILE = None


def set_log_file(path):
    global _LOG_FILE
    _LOG_FILE = path
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    if _LOG_FILE:
        with open(_LOG_FILE, "a") as f:
            f.write(line + "\n")


def save_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    raise TypeError(f"not JSON serialisable: {type(o)}")


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def stable_seed(*parts):
    """Deterministic 31-bit seed from any printable parts (same on every machine)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def time_left():
    """Seconds until the session deadline (FMIA_DEADLINE, epoch seconds); inf if unset."""
    d = os.environ.get("FMIA_DEADLINE")
    return float(d) - time.time() if d else math.inf


def fmt_s(s):
    if s == math.inf:
        return "inf"
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"
