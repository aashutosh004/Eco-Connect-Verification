import os, cv2
from typing import List, Tuple
from ultralytics import YOLO
from config import YOLO_MODEL_PATH

yolo_model = YOLO(YOLO_MODEL_PATH)
print("[YOLO] Loaded names:", yolo_model.names)

TASK_ALIASES = {
    "waste": ["person-collecting-waste","person_collecting_waste","waste_collection"],
    "waste collection": ["person-collecting-waste","person_collecting_waste","waste_collection"],
    "plantation": ["person_planting","people_planting","planting"],
    "planting": ["person_planting","people_planting","planting"],
    "feeding": ["person_feeding_animal","feeding_animal","animal_feeding"],
    "feeding animal": ["person_feeding_animal","feeding_animal","animal_feeding"],
    "person-collecting-waste": ["person-collecting-waste"],
    "person_collecting_waste": ["person_collecting_waste"],
    "person_planting": ["person_planting"],
    "person_feeding_animal": ["person_feeding_animal"],
}

TASK_MIN_CONF = {k: 0.25 for k in TASK_ALIASES.keys()}

def _annotate_and_save(result, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, result.plot())

def _labels_and_confs(result) -> List[Tuple[str, float]]:
    pairs, names = [], yolo_model.names
    if hasattr(result,"boxes") and result.boxes is not None and len(result.boxes) > 0:
        for cls_id, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            pairs.append((names[int(cls_id)], float(conf)))
    return pairs

def _match_task(task_type: str, dets: List[Tuple[str, float]]) -> bool:
    t = (task_type or "").strip().lower()
    aliases = TASK_ALIASES.get(t, [t])
    min_conf = TASK_MIN_CONF.get(t, 0.25)
    for alias in aliases:
        an = alias.strip().lower()
        for lbl, conf in dets:
            if lbl.strip().lower() == an and conf >= min_conf:
                return True
    return False

def detect_task(path: str, task_type: str) -> bool:
    results = yolo_model.predict(source=path, imgsz=512, conf=0.25, iou=0.5, verbose=False)
    if not results: return False
    try:
        _annotate_and_save(results[0], os.path.join("debug","yolo_annotated.jpg"))
    except Exception:
        pass
    dets = _labels_and_confs(results[0])
    return _match_task(task_type, dets)

