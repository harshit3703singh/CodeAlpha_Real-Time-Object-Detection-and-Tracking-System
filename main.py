import cv2
import os
import time
import logging
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

logging.getLogger("ultralytics").setLevel(logging.WARNING)

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/yolov8n.pt"
VIDEO_PATH = "videos/sample.mp4"
OUTPUT_PATH = "output/output_video.mp4"

USE_WEBCAM = True
CAMERA_INDEXES = [0, 1, 2]

CONF_THRES = 0.45
IOU_THRES = 0.5

MAX_AGE = 40
N_INIT = 3
COSINE_DIST = 0.25

SAVE_OUTPUT = True
MIN_BOX_SIZE = 20

# =========================
# HELPERS
# =========================
def ensure_dir(path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

def open_camera():
    for idx in CAMERA_INDEXES:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Camera opened successfully at index {idx}")
            return cap
        cap.release()
    return None

def open_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if cap.isOpened():
        print(f"Video opened successfully: {VIDEO_PATH}")
        return cap
    return None

def get_color(track_id):
    np.random.seed(track_id + 1000)
    c = np.random.randint(60, 255, 3)
    return int(c[0]), int(c[1]), int(c[2])

def draw_label(frame, x1, y1, text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y1 - th - baseline - 10)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 12, top + th + baseline + 8), color, -1)
    cv2.putText(frame, text, (x1 + 6, top + th + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

# =========================
# INIT
# =========================
ensure_dir(OUTPUT_PATH)

model = YOLO(MODEL_PATH)
tracker = DeepSort(
    max_age=MAX_AGE,
    n_init=N_INIT,
    max_cosine_distance=COSINE_DIST
)

cap = open_camera() if USE_WEBCAM else open_video()

if cap is None or not cap.isOpened():
    print("Error: Camera/Video open nahi hua.")
    print("Try:")
    print("1. CAMERA_INDEXES = [1, 0, 2]")
    print("2. USE_WEBCAM = False")
    print("3. Camera permission check karo")
    raise SystemExit

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0:
    fps = 30

writer = None
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("\nControls: q = quit, s = save screenshot")
print("Starting tracking...")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received. Exiting...")
        break

    start_time = time.time()

    results = model.predict(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]
    detections = []

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            w = int(x2 - x1)
            h = int(y2 - y1)

            if conf < CONF_THRES:
                continue
            if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                continue

            detections.append(([int(x1), int(y1), w, h], conf, cls_name))

    tracks = tracker.update_tracks(detections, frame=frame)

    active_tracks = 0

    for track in tracks:
        if not track.is_confirmed():
            continue
        if track.time_since_update > 1:
            continue

        active_tracks += 1
        track_id = int(track.track_id)

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        color = get_color(track_id)

        class_name = "Object"
        if hasattr(track, "det_class") and track.det_class is not None:
            class_name = str(track.det_class)
        elif hasattr(track, "class_name") and track.class_name is not None:
            class_name = str(track.class_name)

        label = f"{class_name} | ID:{track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, x1, y1, label, color)

    fps_now = 1.0 / max(1e-6, time.time() - start_time)

    cv2.putText(frame, "YOLOv8 + DeepSORT Tracking", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "FPS: {:.1f}".format(fps_now), (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Active Tracks: {}".format(active_tracks), (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Professional Object Detection & Tracking", frame)

    if writer is not None:
        writer.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        shot_path = f"output/screenshot_{int(time.time())}.jpg"
        cv2.imwrite(shot_path, frame)
        print(f"Screenshot saved: {shot_path}")

# =========================
# CLEANUP
# =========================
cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()

print("Finished successfully.")