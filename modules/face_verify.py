import cv2, numpy as np
from pathlib import Path
from modules.facekit import get_face_app
from modules.dbio import load_dict_npy
from config import DB_PATH, EMBEDDING_THRESHOLD

DB_PATH = Path(DB_PATH)

def _norm(v): return v / (np.linalg.norm(v) + 1e-10)

def verify_face(username: str, proof_path: str):
    db = load_dict_npy(DB_PATH)
    centroid = db.get(f"{username}__centroid")
    shots = db.get(username)
    if centroid is None and shots is None:
        return False, 0.0
    if isinstance(shots, np.ndarray):
        shots = [shots]

    img = cv2.imread(proof_path)
    if img is None:
        return False, 0.0

    faces = get_face_app().get(img)
    if not faces:
        return False, 0.0

    # best probe from proof
    best_query, best_probe = None, -1.0
    for f in faces:
        emb = _norm(f.embedding)
        s = float(np.dot(emb, centroid)) if centroid is not None else -1.0
        if s > best_probe:
            best_probe, best_query = s, emb
    if best_query is None:
        best_query = _norm(faces[0].embedding)

    sims = []
    if centroid is not None:
        sims.append(float(np.dot(best_query, centroid)))
    if shots:
        sims.extend(float(np.dot(best_query, _norm(e))) for e in shots)

    best = max(sims) if sims else -1.0
    return (best >= EMBEDDING_THRESHOLD), best
