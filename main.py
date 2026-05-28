import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ============================================
# LOAD YOLO MODEL
# ============================================
model = YOLO("models/yolov8n.pt")

# ============================================
# INITIALIZE DEEP SORT TRACKER
# ============================================
tracker = DeepSort(
    max_age=30,
    n_init=2,
    max_cosine_distance=0.3
)

# ============================================
# SELECT INPUT SOURCE
# ============================================

print("\n========== OBJECT DETECTION & TRACKING ==========")
print("1. Webcam")
print("2. Video File")

choice = input("Enter choice (1 or 2): ")

if choice == "1":

    print("\nOpening Webcam...\n")

    # TRY DIFFERENT CAMERA INDEXES
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

elif choice == "2":

    print("\nOpening Video File...\n")

    video_path = "videos/sample.mp4"

    cap = cv2.VideoCapture(video_path)

else:
    print("Invalid Choice")
    exit()

# ============================================
# CHECK VIDEO SOURCE
# ============================================
if not cap.isOpened():
    print("Error: Unable to open camera/video")
    exit()

# ============================================
# VIDEO PROPERTIES
# ============================================
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if fps == 0:
    fps = 30

# ============================================
# OUTPUT VIDEO
# ============================================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    "output/output_video.mp4",
    fourcc,
    fps,
    (width, height)
)

# ============================================
# RANDOM COLORS
# ============================================
np.random.seed(42)

colors = np.random.randint(
    0,
    255,
    size=(1000, 3),
    dtype="uint8"
)

# ============================================
# MAIN LOOP
# ============================================
while True:

    ret, frame = cap.read()

    if not ret:
        print("\nVideo Finished")
        break

    # ========================================
    # YOLO OBJECT DETECTION
    # ========================================
    results = model(
        frame,
        conf=0.6,
        iou=0.5,
        stream=True
    )

    detections = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            # GET BOX COORDINATES
            x1, y1, x2, y2 = box.xyxy[0]

            x1, y1, x2, y2 = map(
                int,
                [x1, y1, x2, y2]
            )

            # CONFIDENCE
            confidence = float(box.conf[0])

            # CLASS ID
            class_id = int(box.cls[0])

            # CLASS NAME
            class_name = model.names[class_id]

            # FILTER LOW CONFIDENCE
            if confidence > 0.6:

                w = x2 - x1
                h = y2 - y1

                detections.append(
                    (
                        [x1, y1, w, h],
                        confidence,
                        class_name
                    )
                )

    # ========================================
    # UPDATE TRACKER
    # ========================================
    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    # ========================================
    # DRAW TRACKING
    # ========================================
    for track in tracks:

        if not track.is_confirmed():
            continue

        # TRACK ID
        track_id = int(track.track_id)

        # GET TRACK BOX
        x1, y1, x2, y2 = map(
            int,
            track.to_ltrb()
        )

        # REMOVE EXTRA PADDING
        padding = 5

        x1 += padding
        y1 += padding
        x2 -= padding
        y2 -= padding

        # SAFETY LIMITS
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(0, x2)
        y2 = max(0, y2)

        # BOX WIDTH & HEIGHT
        w = x2 - x1
        h = y2 - y1

        # IGNORE HUGE FALSE BOXES
        if w > frame.shape[1] * 0.8:
            continue

        if h > frame.shape[0] * 0.8:
            continue

        # COLOR
        color = colors[
            track_id % len(colors)
        ].tolist()

        # ====================================
        # DRAW RECTANGLE
        # ====================================
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # ====================================
        # LABEL TEXT
        # ====================================
        label = f"ID: {track_id}"

        # LABEL BACKGROUND
        cv2.rectangle(
            frame,
            (x1, y1 - 30),
            (x1 + 120, y1),
            color,
            -1
        )

        # LABEL
        cv2.putText(
            frame,
            label,
            (x1 + 10, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # ========================================
    # TITLE
    # ========================================
    cv2.putText(
        frame,
        "Real-Time Object Detection & Tracking",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # ========================================
    # SHOW OUTPUT
    # ========================================
    cv2.imshow(
        "YOLOv8 Object Detection & Tracking",
        frame
    )

    # ========================================
    # SAVE OUTPUT VIDEO
    # ========================================
    out.write(frame)

    # ========================================
    # EXIT BUTTON
    # ========================================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================
# RELEASE EVERYTHING
# ============================================
cap.release()

out.release()

cv2.destroyAllWindows()

print("\nProcessing Finished Successfully!")