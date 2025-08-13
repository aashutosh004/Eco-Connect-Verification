import os, json, cv2, torch, numpy as np, torch.nn.functional as F
from dataclasses import dataclass
from modules.facekit import get_face_app
from config import (
    LIVENESS_ENABLED, LIVENESS_MODEL_PTH, LIVENESS_IMG_SIZE,
    LIVENESS_THRESHOLD, LIVENESS_NUM_CLASSES, REAL_CLASS_INDEX
)

from modules.third_party.MiniFASNet import MiniFASNetV2

@dataclass
class _State:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: torch.nn.Module | None = None

STATE = _State()

def _init_once():
    if STATE.model is None:
        if not os.path.exists(LIVENESS_MODEL_PTH):
            raise FileNotFoundError(f"Liveness model not found: {LIVENESS_MODEL_PTH}")
        model = MiniFASNetV2(embedding_size=128, conv6_kernel=(5,5),
                             num_classes=LIVENESS_NUM_CLASSES, img_channel=3)
        ckpt = torch.load(LIVENESS_MODEL_PTH, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        model.eval().to(STATE.device)
        STATE.model = model

def _crop_face(img_bgr: np.ndarray) -> np.ndarray:
    faces = get_face_app().get(img_bgr)
    if faces:
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        x1,y1,x2,y2 = map(int, f.bbox)
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size > 0: return crop
    h,w = img_bgr.shape[:2]; s = min(h,w); y0=(h-s)//2; x0=(w-s)//2
    return img_bgr[y0:y0+s, x0:x0+s]

def _preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.resize(img_bgr, (LIVENESS_IMG_SIZE, LIVENESS_IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    img = np.transpose(img, (2,0,1))[None,...]
    return torch.from_numpy(img).to(STATE.device)

@torch.inference_mode()
def check_liveness(image_path: str):
    if not LIVENESS_ENABLED:
        return True, 1.0
    _init_once()
    img = cv2.imread(image_path)
    if img is None:
        return False, 0.0
    crop = _crop_face(img)
    x = _preprocess(crop)
    logits = STATE.model(x)
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    idx = REAL_CLASS_INDEX if REAL_CLASS_INDEX < probs.shape[0] else int(np.argmax(probs))
    real_prob = float(probs[idx])
    return (real_prob >= LIVENESS_THRESHOLD), real_prob

# Optional debug helper (keep off in prod unless DEBUG_ROUTES=true)
@torch.inference_mode()
def check_liveness_debug(image_path: str):
    _init_once()
    img = cv2.imread(image_path)
    if img is None: return False, 0.0, None, None
    crop = _crop_face(img)
    os.makedirs("debug", exist_ok=True)
    cp = os.path.join("debug", "liveness_crop.jpg"); cv2.imwrite(cp, crop)
    x = _preprocess(crop)
    probs = F.softmax(STATE.model(x), dim=1).detach().cpu().numpy()[0].tolist()
    with open(os.path.join("debug","liveness_probs.json"),"w") as f:
        json.dump({"probs": probs}, f)
    idx = REAL_CLASS_INDEX if REAL_CLASS_INDEX < len(probs) else int(np.argmax(probs))
    return (probs[idx] >= LIVENESS_THRESHOLD), float(probs[idx]), "debug/liveness_probs.json", cp
