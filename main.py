import cv2
import json
import pandas as pd
from ultralytics import YOLO

# =========================
# LOAD MODELS
# =========================
ball_model = YOLO("models/ball.pt")
pose_model = YOLO("models/yolov8m-pose.pt")

# =========================
# VIDEO INPUT
# =========================
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "output/annotated.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# =========================
# OUTPUT STORAGE
# =========================
results_data = []
frame_idx = 0

# =========================
# SHOT STABILITY LOGIC
# =========================
stable_shot = "unknown"
shot_lock_frames = 10
shot_lock_counter = 0

# =========================
# SHOT CLASSIFIER
# =========================
def classify_shot(kp):

    if kp is None or len(kp) < 11:
        return "unknown"

    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_WRIST = 9
    RIGHT_WRIST = 10

    try:
        rw = kp[RIGHT_WRIST]
        rs = kp[RIGHT_SHOULDER]

        # safety check
        if rw[0] == 0 and rw[1] == 0:
            return "unknown"

        # SMASH
        if rw[1] < rs[1] - 40:
            return "smash"

        # FOREHAND
        if rw[0] > rs[0] + 50:
            return "forehand"

        # BACKHAND
        if rw[0] < rs[0] - 50:
            return "backhand"

    except:
        return "unknown"

    return "unknown"

# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    # =========================
    # BALL DETECTION
    # =========================
    ball_results = ball_model.predict(frame, conf=0.25, verbose=False)[0]

    for box in ball_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "Ball", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # =========================
    # POSE DETECTION
    # =========================
    pose_results = pose_model.predict(frame, conf=0.5, verbose=False)[0]

    current_shot = "unknown"

    if pose_results.keypoints is not None:

        for person in pose_results.keypoints.xy:

            keypoints = person.tolist()
            current_shot = classify_shot(keypoints)

            # =========================
            # SHOT LOCK LOGIC (IMPORTANT)
            # =========================
            if shot_lock_counter > 0:
                shot_lock_counter -= 1

            else:
                if current_shot != "unknown" and current_shot != stable_shot:
                    stable_shot = current_shot
                    shot_lock_counter = shot_lock_frames

            # draw keypoints
            for kp in keypoints:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    # =========================
    # OVERLAYS
    # =========================
    timestamp = frame_idx / fps

    cv2.putText(frame, f"Shot: {stable_shot}",
                (40, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 3)

    cv2.putText(frame, f"Frame: {frame_idx}",
                (40, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    cv2.putText(frame, f"Time: {timestamp:.2f}s",
                (40, 140), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    # =========================
    # STORE RESULTS
    # =========================
    results_data.append({
        "frame": frame_idx,
        "timestamp": round(timestamp, 2),
        "shot_type": stable_shot
    })

    # =========================
    # WRITE VIDEO
    # =========================
    out.write(frame)

    # =========================
    # PROGRESS LOG
    # =========================
    if frame_idx % 30 == 0:
        print(f"Processing frame {frame_idx}")

    frame_idx += 1

# =========================
# CLEANUP
# =========================
cap.release()
out.release()

# =========================
# SAVE OUTPUT FILES
# =========================
df = pd.DataFrame(results_data)
df.to_csv("output/shots.csv", index=False)

with open("output/shots.json", "w") as f:
    json.dump(results_data, f, indent=4)

print("DONE ✔ Output saved in /output")