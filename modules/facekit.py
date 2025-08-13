import threading, torch
from insightface.app import FaceAnalysis
from config import INSIGHT_PACK, DET_W, DET_H

PROVIDERS = ['CPUExecutionProvider']
if torch.cuda.is_available():
    PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

_face_app = None
_lock = threading.Lock()

def get_face_app() -> FaceAnalysis:
    global _face_app
    if _face_app is None:
        with _lock:
            if _face_app is None:
                fa = FaceAnalysis(name=INSIGHT_PACK, providers=PROVIDERS)
                ctx_id = 0 if ('CUDAExecutionProvider' in PROVIDERS and torch.cuda.is_available()) else -1
                fa.prepare(ctx_id=ctx_id, det_size=(DET_W, DET_H))
                _face_app = fa
    return _face_app
