
import os
import time
from datetime import datetime

import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from database_manager import save_alert
from email_alert import send_alert_email

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALERTS_DIR = os.path.join(BASE_DIR, "alerts")
os.makedirs(ALERTS_DIR, exist_ok=True)

PERSON_MODEL_PATH = os.path.join(BASE_DIR, "src", "yolo11n.pt")


# --------------------------------------------------
# 2. LOAD MODELS
# --------------------------------------------------

print("Loading person detection model...")
person_model = YOLO(PERSON_MODEL_PATH)

print("Loading fire/smoke detection model...")
fire_model_path = hf_hub_download(
    repo_id="rabahdev/fire-smoke-yolov8n",
    filename="best.pt"
)
fire_model = YOLO(fire_model_path)

print("Loading weapon detection model...")
weapon_model_path = hf_hub_download(
    repo_id="cosgun99/gun-knife-yolo11n",
    filename="best.pt"
)
weapon_model = YOLO(weapon_model_path)

print("All models loaded successfully!")
print("Fire/smoke classes:", fire_model.names)
print("Weapon classes:", weapon_model.names)


# --------------------------------------------------
# 3. CAMERA
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access camera")
    raise SystemExit


# --------------------------------------------------
# 4. INTRUSION SETTINGS
# --------------------------------------------------

zone_x1, zone_y1 = 200, 150
zone_x2, zone_y2 = 500, 400

intruders_inside = set()
intruder_missing_frames = {}
INTRUDER_GRACE_FRAMES = 15


# --------------------------------------------------
# 5. EVENT SETTINGS
# --------------------------------------------------

CONFIDENCE_THRESHOLD = 0.35
IMAGE_SIZE = 640
IOU_THRESHOLD = 0.45

REQUIRED_FRAMES = 3
CLEAR_FRAMES = 10
EVENT_COOLDOWN_SECONDS = 30

event_types = ("FIRE", "SMOKE", "WEAPON_GUN", "WEAPON_KNIFE")

event_counts = {event: 0 for event in event_types}
missing_counts = {event: 0 for event in event_types}
active_events = set()
last_event_alert_time = {event: 0.0 for event in event_types}


# --------------------------------------------------
# 6. SAVE SCREENSHOT AND DATABASE ALERT
# --------------------------------------------------


def save_event(event_type, tracking_id, frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(
        ALERTS_DIR,
        f"{event_type.lower()}_{timestamp}.jpg"
    )

    if not cv2.imwrite(filename, frame):
        print("Failed to save screenshot:", filename)
        return False

    try:
        save_alert(event_type, tracking_id, filename)
        print(f"{event_type} ALERT SAVED")
        print("Screenshot:", filename)

    except Exception as error:
        print("Database alert failed:", error)
        return False

    # Send email only after the screenshot and database alert are saved.
    send_alert_email(event_type, tracking_id, filename)

    return True


def confirm_and_save(event_type, frame, tracking_id=None):
    """Save an event only after consecutive detections and cooldown."""
    now = time.monotonic()

    if event_type in active_events:
        return

    if event_counts[event_type] < REQUIRED_FRAMES:
        return

    active_events.add(event_type)

    if now - last_event_alert_time[event_type] < EVENT_COOLDOWN_SECONDS:
        print(f"{event_type} detected; alert suppressed by cooldown.")
        return

    if save_event(event_type, tracking_id, frame):
        last_event_alert_time[event_type] = now


# --------------------------------------------------
# 7. MAIN SURVEILLANCE LOOP
# --------------------------------------------------

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read camera frame")
            break

        annotated_frame = frame.copy()

        # ==========================================
        # A. PERSON DETECTION AND TRACKING
        # ==========================================

        person_results = person_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            verbose=False
        )

        annotated_frame = person_results[0].plot()

        cv2.rectangle(
            annotated_frame,
            (zone_x1, zone_y1),
            (zone_x2, zone_y2),
            (0, 0, 255),
            3
        )

        cv2.putText(
            annotated_frame,
            "RESTRICTED AREA",
            (zone_x1, max(zone_y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        current_intruders = set()
        person_boxes = person_results[0].boxes

        if person_boxes is not None and person_boxes.id is not None:
            for i, box in enumerate(person_boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                person_id = int(person_boxes.id[i].item())

                inside_zone = (
                    zone_x1 < center_x < zone_x2
                    and zone_y1 < center_y < zone_y2
                )

                if inside_zone:
                    current_intruders.add(person_id)

                    if person_id not in intruders_inside:
                        cv2.putText(
                            annotated_frame,
                            "INTRUSION DETECTED",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            3
                        )

                        save_event("INTRUSION", person_id, annotated_frame)

        # Remember intruders across short tracking interruptions.
        for person_id in current_intruders:
            intruder_missing_frames[person_id] = 0

        for person_id in list(intruders_inside):
            if person_id not in current_intruders:
                intruder_missing_frames[person_id] = (
                    intruder_missing_frames.get(person_id, 0) + 1
                )

                if intruder_missing_frames[person_id] > INTRUDER_GRACE_FRAMES:
                    intruders_inside.discard(person_id)
                    intruder_missing_frames.pop(person_id, None)

        intruders_inside.update(current_intruders)

        for person_id in list(intruder_missing_frames):
            if person_id not in intruders_inside:
                intruder_missing_frames.pop(person_id, None)

        # ==========================================
        # B. FIRE AND SMOKE DETECTION
        # ==========================================

        fire_result = fire_model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )[0]

        detected_events = set()

        if fire_result.boxes is not None:
            for box in fire_result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                label = str(fire_model.names[class_id]).upper()

                if label not in ("FIRE", "SMOKE"):
                    continue

                detected_events.add(label)

                fx1, fy1, fx2, fy2 = (
                    box.xyxy[0].cpu().numpy().astype(int)
                )

                cv2.rectangle(
                    annotated_frame,
                    (fx1, fy1),
                    (fx2, fy2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated_frame,
                    f"{label}: {confidence:.2f}",
                    (fx1, max(fy1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        # ==========================================
        # C. WEAPON DETECTION
        # ==========================================

        weapon_result = weapon_model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=0.40,
            iou=IOU_THRESHOLD,
            verbose=False
        )[0]

        if weapon_result.boxes is not None:
            for box in weapon_result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                label = str(weapon_model.names[class_id]).lower()

                if label not in ("gun", "knife"):
                    continue

                event_type = f"WEAPON_{label.upper()}"
                detected_events.add(event_type)

                wx1, wy1, wx2, wy2 = (
                    box.xyxy[0].cpu().numpy().astype(int)
                )

                cv2.rectangle(
                    annotated_frame,
                    (wx1, wy1),
                    (wx2, wy2),
                    (0, 165, 255),
                    2
                )

                cv2.putText(
                    annotated_frame,
                    f"{label.upper()}: {confidence:.2f}",
                    (wx1, max(wy1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2
                )

        # ==========================================
        # D. CONFIRM EVENTS AND APPLY COOLDOWN
        # ==========================================

        for event_type in event_types:
            if event_type in detected_events:
                event_counts[event_type] += 1
                missing_counts[event_type] = 0
            else:
                event_counts[event_type] = 0
                missing_counts[event_type] += 1

                if missing_counts[event_type] >= CLEAR_FRAMES:
                    active_events.discard(event_type)

            confirm_and_save(event_type, annotated_frame)

        # ==========================================
        # E. DISPLAY STATUS
        # ==========================================

        if current_intruders:
            cv2.putText(
                annotated_frame,
                "INTRUSION DETECTED",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        status_y = 210

        for event_type in event_types:
            if event_type in active_events:
                cv2.putText(
                    annotated_frame,
                    f"{event_type.replace('_', ' ')} ALERT ACTIVE",
                    (20, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )
                status_y += 30

        cv2.imshow(
            "AI Home Security - Integrated Detection",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Surveillance system stopped.")
