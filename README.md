# Eco-Connect Verification (Face + Liveness + Task)
Server-side Python service for verifying a user’s identity (face embedding), checking liveness (anti-spoof), and validating a task/action in a submitted image using YOLO. Ships with a minimal HTML test page.

> **Stack**: Flask API · InsightFace (buffalo_l) · MiniFASNetV2 (PyTorch) · Ultralytics YOLO (pt/ONNX) · NumPy · onnxruntime (CPU/GPU)

---

## ✨ Features
- **Enrollment with multi‑shot support**: send multiple `images[]` and the service builds a **centroid embedding** per user (stores up to 5 latest shots).
- **Verification pipeline (parallel)**: face **match** + **liveness** + **YOLO task** run concurrently for faster responses.
- **Production‑leaning defaults**: request size caps, allowed extensions, atomic DB writes, unified error JSON, CORS allow‑list.
- **Simple test UI** at `/` for webcam capture or file upload (no external frontend required for quick tests).
- **CPU‑first**: runs on CPU; **auto‑uses GPU** if `onnxruntime-gpu` + CUDA are available.
- **Model‑agnostic**: swap InsightFace packs (`buffalo_l`/`antelopev2`), point YOLO to your custom `.pt`/`.onnx`, adjust liveness hyper‑params.

---

## 📁 Directory Layout
```
.
├─ app.py                      # Flask app (API + test page)
├─ config.py                   # All tunables (env-driven)
├─ requirements.txt
├─ templates/
│  └─ index.html               # Test UI (webcam + upload)
├─ models/
│  ├─ best.pt                  # YOLO model (your custom labels)
│  └─ 2.7_80x80_MiniFASNetV2.pth  # Liveness model
├─ modules/
│  ├─ dbio.py                  # atomic .npy read/write
│  ├─ errors.py                # unified JSON errors
│  ├─ face_enroll.py           # enrollment pipeline
│  ├─ face_verify.py           # verification (centroid + shots)
│  ├─ facekit.py               # InsightFace loader
│  ├─ liveness_check.py        # MiniFASNetV2 inference
│  ├─ task_detection.py        # YOLO task detection
│  └─ third_party/MiniFASNet.py
├─ uploads/
│  ├─ enroll/                  # saved enrollment shots
│  └─ proofs/                  # saved verification images
├─ database/
│  └─ embeddings_db.npy        # user embeddings DB
└─ .env.example                # example environment overrides
```

---

## ✅ Prerequisites
- **Python** 3.10–3.11 (recommended)
- **OS**: Windows / Linux / macOS
- **Optional GPU**: CUDA + `onnxruntime-gpu` (matching CUDA/cuDNN)

**Ubuntu system deps (example):**
```bash
sudo apt-get update
sudo apt-get install -y python3-dev build-essential libgl1
```

---

## 🚀 Setup

### 1) Create & activate a virtual environment
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```

### 2) Install Python packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) (Recommended) Track large model files with Git LFS
```bash
git lfs install
git lfs track "models/*.pt" "models/*.pth"
git add .gitattributes
git commit -m "chore: track model binaries with LFS"
```

### 4) Configure environment (optional)
Copy `.env.example` → `.env` and adjust as needed.

```dotenv
# .env
ENV=dev
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000
EMBEDDING_THRESHOLD=0.55
LIVENESS_THRESHOLD=0.80
INSIGHT_PACK=buffalo_l
YOLO_MODEL_PATH=models/best.pt
LIVENESS_MODEL_PTH=models/2.7_80x80_MiniFASNetV2.pth
DET_W=416
DET_H=416
MAX_CONTENT_LENGTH_MB=8
```

> **Tip:** If you switch `INSIGHT_PACK` (e.g., `buffalo_l` ↔ `antelopev2`), **re‑enroll all users** to avoid drift in embeddings.

---

## ▶️ Run (Development)
From inside the virtual environment:
```bash
flask --app app.py run
# or
python app.py
```
Open: **http://127.0.0.1:5000/** for the built-in test page.

---

## 🔧 Configuration Reference (`config.py`)
| Variable | Default | Meaning |
|---|---|---|
| `ENV` | `production` | non‑prod enables debug runner in `app.py` |
| `CORS_ORIGINS` | *(empty)* | comma‑separated list of allowed front‑end origins |
| `MAX_CONTENT_LENGTH_MB` | `8` | request size cap in MB |
| `INSIGHT_PACK` | `buffalo_l` | InsightFace model pack (`buffalo_l` or `antelopev2`) |
| `DET_W`, `DET_H` | `416` | detector input size (smaller → faster CPU) |
| `EMBEDDING_THRESHOLD` | `0.55` | face similarity threshold (cosine 0..1) |
| `LIVENESS_ENABLED` | `true` | toggle liveness check |
| `LIVENESS_THRESHOLD` | `0.80` | probability threshold for “real” |
| `LIVENESS_IMG_SIZE` | `80` | MiniFASNetV2 input size |
| `LIVENESS_NUM_CLASSES` | `3` | num classes used by the model |
| `REAL_CLASS_INDEX` | `2` | index of “real” class |
| `YOLO_MODEL_PATH` | `models/best.pt` | YOLO model path (pt/onnx) |
| `LIVENESS_MODEL_PTH` | `models/2.7_80x80_MiniFASNetV2.pth` | liveness model path |

---

## 🧠 How It Works (High Level)
1. **Enrollment** (`/api/signup`): Accepts 1–5 images for a username. Deduplicates frames by MD5, extracts face embeddings with InsightFace, stores **per‑user centroid** and raw shots.
2. **Verification** (`/api/verify`): Given a username, task label, and proof image:
   - **Face match**: cosine similarity vs. stored centroid.
   - **Liveness**: MiniFASNetV2 classifies crop as real/spoof; thresholds decide pass/fail.
   - **Task detection**: YOLO detects required activity class via **alias mapping** (see `TASK_ALIASES`).
   - All three run **in parallel**; final status is approved only if **all pass**.

---

## 🌐 API Reference
**Base path:** `/api`

### Health
```http
GET /api/health
```
**200**
```json
{ "ok": true }
```

### Signup / Enrollment
> **Recommended**: send multiple shots via `images[]` (3–5 diverse angles). A single `image` is accepted as fallback.

```http
POST /api/signup
Content-Type: multipart/form-data
Form-Data:
  username: string
  images[]: file (repeat 3–5x)  # preferred
  image:    file                 # optional single-shot
```
**200**
```json
{ "status": true, "message": "User enrolled. shots=3" }
```
**4xx** on validation/face errors.

**Notes**
- Duplicate frames (exact bytes) are de-duplicated via **MD5**.
- Service keeps up to **5 most recent shots** per user and stores a **centroid embedding**.

### Verify Task
Runs **face match + liveness + YOLO** concurrently.

```http
POST /api/verify
Content-Type: multipart/form-data
Form-Data:
  username: string
  task: "plantation" | "waste collection" | "feeding animal"
  file: image
```
**200 (approved)**
```json
{
  "status":"approved",
  "user":"ritesh",
  "task":"plantation",
  "similarity": 0.71,
  "liveness_score": 0.99,
  "timestamp": "2025-08-10T17:16:47.546803Z"
}
```
**200 (rejected examples)**
```json
{ "status":"rejected","reason":"Face mismatch","similarity": 0.59 }
{ "status":"rejected","reason":"Liveness failed","liveness_score": 0.01 }
{ "status":"rejected","reason":"Task not verified" }
```

**Task label aliases** (edit in `modules/task_detection.py::TASK_ALIASES`):
- `"plantation"` → `person_planting`, `people_planting`, `planting`
- `"waste collection"` → `person-collecting-waste`, …
- `"feeding animal"` → `person_feeding_animal`, …

---

## 🧪 Quick cURL
```bash
# Enroll (single image)
curl -F "username=ritesh" -F "image=@/path/to/shot1.jpg" http://127.0.0.1:5000/api/signup

# Enroll (multi-shot)
curl -F "username=ritesh" \
     -F "images[]=@/path/shot1.jpg" \
     -F "images[]=@/path/shot2.jpg" \
     -F "images[]=@/path/shot3.jpg" \
     http://127.0.0.1:5000/api/signup

# Verify
curl -F "username=ritesh" -F "task=plantation" -F "file=@/path/proof.jpg" \
     http://127.0.0.1:5000/api/verify
```

---

## 📈 Performance Tips
- **CPU only**: keep `DET_W/H` at **320–416** for faster face detection; use YOLO `imgsz=512` (already set) or lower if needed.
- **GPU**: install `onnxruntime-gpu` with matching **CUDA/cuDNN**; the app will auto‑select CUDA provider.
- **Throughput**: run behind **gunicorn/uvicorn + gevent/uvloop** in prod; put **Nginx** in front for TLS & static.
- **Queues**: for high QPS or bursty loads, consider **Celery/RQ** and persistent model worker processes.

---

## 🔐 Security & Production Notes
- This repo includes a simple **test page**; **disable it** in production if your front‑end handles UX.
- Set **CORS** to the exact front‑end origin, e.g. `CORS_ORIGINS=https://your-frontend-domain`.
- Put behind a reverse proxy with **HTTPS** (Nginx/Traefik).
- Persist `database/embeddings_db.npy` to a **durable volume** (bind mount or cloud disk).
- Treat model files as assets: **Git LFS** or fetch from **object storage** during deploy.
- **Biometrics compliance**: storing embeddings and liveness results may be regulated; consult local laws and add consent/retention policies.

---

## 🧰 Troubleshooting
**Face mismatch is low (~0.58)**
- Enroll **3–5** diverse shots; ensure **good lighting**; include **frontal and slight angles**.
- **Re‑enroll** users if switching `INSIGHT_PACK`.

**Liveness always fails**
- Confirm `REAL_CLASS_INDEX` with a temporary debug helper to inspect per‑class probs.
- Check the **face crop** quality (poor detections → poor liveness).

**YOLO not detecting**
- Verify `YOLO_MODEL_PATH` and label names in `TASK_ALIASES`.
- Try `imgsz=640` for tiny images (note: **slower** on CPU).

**ONNX “CUDAExecutionProvider not available”**
- You’re on CPU. Install `onnxruntime-gpu` and matching **CUDA** to use the GPU.

---

## 🧪 Test Page
- Navigate to `/` to use the built‑in **webcam**/**file upload** UI.
- For production, lock it behind auth or remove the route entirely.

---

## 🧱 Implementation Notes
- **Atomic writes**: `modules/dbio.py` writes `.npy` via temp‑file + `os.replace` to avoid partial writes under concurrent access.
- **Unified errors**: `modules/errors.py` centralizes error responses (consistent JSON shape, HTTP codes).
- **Enrollment logic**: `modules/face_enroll.py` dedupes frames, extracts embeddings, updates centroid & shot set.
- **Verification logic**: `modules/face_verify.py` computes cosine similarity against centroid, supports multi‑shot matching.
- **Liveness**: `modules/liveness_check.py` runs MiniFASNetV2; thresholds set in `config.py`.
- **Task detection**: `modules/task_detection.py` loads YOLO and checks **alias → class** match for a pass.

---

## 🧪 Postman Collection (optional)
You can quickly test by importing two requests:
- **Signup**: `POST /api/signup` (multipart form)
- **Verify**: `POST /api/verify` (multipart form)

---

## 📦 Deployment (Example)
**Gunicorn (WSGI) + Nginx**
```bash
pip install gunicorn gevent
gunicorn -k gevent -w 2 -b 0.0.0.0:5000 app:app
```
- Terminate TLS and serve static via **Nginx**; forward `/api/*` to Gunicorn.
- Persist `database/` and `uploads/` via volumes.

**Docker (sketch)**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=5000
EXPOSE 5000
CMD ["python", "app.py"]
```

---

## 📜 License
**AGPL-3.0** — If you modify the service and offer it to users over a network, you **must** provide the source code of your modified version under the same license. See `LICENSE`.

---

## 🙏 Credits
- **InsightFace** (face detection/recognition)
- **SilentFace / MiniFASNet** (liveness)
- **Ultralytics YOLO** (task detection)

---

## 🧭 Maintainers
-> Team-Z(Ashutosh)
