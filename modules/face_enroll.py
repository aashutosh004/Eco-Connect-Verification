import cv2, numpy as np
from pathlib import Path
from modules.facekit import get_face_app
from modules.dbio import load_dict_npy, save_dict_npy
from config import DB_PATH

DB_PATH = Path(DB_PATH)

def _largest(faces):
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

def enroll_user(username: str, image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return False, "Could not read image"

    faces = get_face_app().get(img)
    if not faces:
        return False, "No face detected"

    f = _largest(faces)
    emb = f.embedding
    emb = emb / (np.linalg.norm(emb) + 1e-10)

    db = load_dict_npy(DB_PATH)
    shots = db.get(username)
    if shots is None:
        shots = []
    elif isinstance(shots, np.ndarray):
        shots = [shots]

    shots.append(emb)
    if len(shots) > 5:
        shots = shots[-5:]
    db[username] = shots

    centroid = np.mean(np.stack(shots, axis=0), axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
    db[f"{username}__centroid"] = centroid

    save_dict_npy(DB_PATH, db)
    return True, f"User enrolled. shots={len(shots)}"
