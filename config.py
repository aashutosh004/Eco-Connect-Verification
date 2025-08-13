import os

# ---------- General ----------
ENV = os.getenv("ENV", "production")  # production|staging|dev
DEBUG = ENV != "production"
DEBUG_ROUTES = os.getenv("DEBUG_ROUTES", "false").lower() == "true"

# ---------- HTTP / Security ----------
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "8")) * 1024 * 1024
ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png"}
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ENROLL_DIR = os.path.join(UPLOAD_DIR, "enroll")
PROOFS_DIR = os.path.join(UPLOAD_DIR, "proofs")
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "embeddings_db.npy")
MODELS_DIR = os.path.join(BASE_DIR, "models")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", os.path.join(MODELS_DIR, "best.pt"))
LIVENESS_MODEL_PTH = os.getenv("LIVENESS_MODEL_PTH", os.path.join(MODELS_DIR, "2.7_80x80_MiniFASNetV2.pth"))

# ---------- InsightFace ----------
INSIGHT_PACK = os.getenv("INSIGHT_PACK", "buffalo_l")  # buffalo_l or antelopev2
DET_W = int(os.getenv("DET_W", "416"))
DET_H = int(os.getenv("DET_H", "416"))

# ---------- Thresholds ----------
EMBEDDING_THRESHOLD = float(os.getenv("EMBEDDING_THRESHOLD", "0.55"))
LIVENESS_THRESHOLD  = float(os.getenv("LIVENESS_THRESHOLD", "0.80"))

# ---------- Liveness ----------
LIVENESS_ENABLED = os.getenv("LIVENESS_ENABLED", "true").lower() == "true"
LIVENESS_IMG_SIZE = int(os.getenv("LIVENESS_IMG_SIZE", "80"))
LIVENESS_NUM_CLASSES = int(os.getenv("LIVENESS_NUM_CLASSES", "3"))
REAL_CLASS_INDEX = int(os.getenv("REAL_CLASS_INDEX", "2"))  # MiniFASNetV2 often 2==real

# ---------- Logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
