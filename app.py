from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os, time, logging, uuid

from config import (
    DEBUG, DEBUG_ROUTES, LOG_LEVEL, MAX_CONTENT_LENGTH, ALLOWED_IMG_EXT,
    ENROLL_DIR, PROOFS_DIR, DB_DIR, UPLOAD_DIR, CORS_ORIGINS
)
from modules.errors import register_error_handlers
from modules.face_enroll import enroll_user
from modules.face_verify import verify_face
from modules.liveness_check import check_liveness
from modules.task_detection import detect_task

app = Flask(__name__, template_folder="templates", static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
app.logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

# dirs
for d in (DB_DIR, ENROLL_DIR, PROOFS_DIR, UPLOAD_DIR, "debug"):
    os.makedirs(d, exist_ok=True)

# CORS (restrict in prod)
try:
    from flask_cors import CORS
    if CORS_ORIGINS:
        CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})
except Exception:
    pass

register_error_handlers(app)

def _rid() -> str:
    return uuid.uuid4().hex[:8]

def _is_allowed_image(filename: str) -> bool:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext in ALLOWED_IMG_EXT

def _secure_save(upload, folder, fname_hint):
    os.makedirs(folder, exist_ok=True)
    fname = secure_filename(upload.filename or fname_hint)
    if not _is_allowed_image(fname):
        raise ValueError("unsupported_file_type")
    path = os.path.join(folder, fname)
    upload.save(path)
    return path

# ---------- pages ----------
@app.get("/")
def index():
    return render_template("index.html")

# ---------- API ----------
@app.get("/api/health")
def health():
    return jsonify({"ok": True})

@app.post("/api/signup")
def api_signup():
    """Accepts multi-shot via images[] (preferred). Falls back to single 'image' if needed.
       De-duplicates identical images by MD5 before enrolling each shot.
    """
    rid = _rid()
    username = request.form.get("username")
    files_multi = request.files.getlist("images[]") or []
    single = request.files.get("image")
    files = list(files_multi)
    if single:
        files.append(single)

    if not username or not files:
        return jsonify({"status": False, "message": "username and image(s) required"}), 400

    # de-duplicate frames by hash
    import hashlib
    seen = set()
    kept = []
    for f in files:
        bio = f.read()
        h = hashlib.md5(bio).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        f.stream.seek(0)  # rewind to allow saving
        kept.append(f)

    if not kept:
        return jsonify({"status": False, "message": "no unique images provided"}), 400

    shots_saved = 0
    try:
        # enroll every kept shot; enroll_user appends & recomputes centroid
        for i, f in enumerate(kept, 1):
            path = _secure_save(f, ENROLL_DIR, f"{username}_{i}.jpg")
            ok, msg = enroll_user(username, path)
            if not ok:
                app.logger.info(f"[{rid}] enroll STOP at shot={i} -> {ok} {msg}")
                return jsonify({"status": ok, "message": msg}), 400
            shots_saved += 1
    except ValueError:
        return jsonify({"status": "error", "message": "Unsupported file type"}), 400

    app.logger.info(f"[{rid}] enroll user={username} shots_in={len(files)} unique={len(kept)} saved={shots_saved}")
    return jsonify({"status": True, "message": f"User enrolled. shots={shots_saved}"}), 200

@app.post("/api/verify")
def api_verify():
    rid = _rid()
    username = request.form.get("username")
    task_type = request.form.get("task")
    file = request.files.get("file")
    if not username or not task_type or not file:
        return jsonify({"status": "rejected", "reason": "username, task, file required"}), 400

    try:
        path = _secure_save(file, PROOFS_DIR, f"{username}_proof.jpg")
    except ValueError:
        return jsonify({"status": "rejected", "reason": "Unsupported file type"}), 400

    t0 = time.perf_counter()
    face_ok, sim = verify_face(username, path)
    t1 = time.perf_counter()
    if not face_ok:
        app.logger.info(f"[{rid}] face_mismatch user={username} sim={sim:.3f} t={t1-t0:.3f}s")
        return jsonify({"status": "rejected", "reason": "Face mismatch", "similarity": sim})

    # run liveness + task in parallel
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_live = ex.submit(check_liveness, path)
        fut_task = ex.submit(detect_task, path, task_type)
        live_ok, live_score = fut_live.result()
        task_ok = fut_task.result()
    t2 = time.perf_counter()

    app.logger.info(f"[{rid}] verify user={username} face={t1-t0:.3f}s l+y={t2-t1:.3f}s total={t2-t0:.3f}s "
                    f"sim={sim:.3f} live={live_score:.3f} task_ok={task_ok}")

    if not live_ok:
        return jsonify({"status": "rejected", "reason": "Liveness failed", "liveness_score": live_score})
    if not task_ok:
        return jsonify({"status": "rejected", "reason": "Task not verified"})

    from datetime import datetime, timezone
    return jsonify({
        "status": "approved",
        "user": username,
        "task": task_type,
        "similarity": float(sim),
        "liveness_score": float(live_score),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

# Optional: enable debug routes only when you want
# if DEBUG_ROUTES:
#     @app.post("/api/debug/liveness")
#     def api_debug_liveness():
#         f = request.files.get("file")
#         if not f: return jsonify({"ok": False, "msg": "file required"}), 400
#         p = _secure_save(f, PROOFS_DIR, "live.jpg")
#         ok, score, probs_path, crop_path = check_liveness_debug(p)
#         return jsonify({"ok": ok, "score": float(score), "probs": probs_path, "crop_saved": crop_path})

if __name__ == "__main__":
    # Dev server only
    try:
        import torch, cv2, os as _os
        torch.set_num_threads(max(1, os.cpu_count() // 2))
        cv2.setNumThreads(0)
        _os.environ.setdefault("OMP_NUM_THREADS", str(max(1, os.cpu_count() // 2)))
        _os.environ.setdefault("MKL_NUM_THREADS", str(max(1, os.cpu_count() // 2)))
    except Exception:
        pass
    app.run(host="0.0.0.0", port=5000, debug=DEBUG)
