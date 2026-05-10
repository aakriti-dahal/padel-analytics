from ultralytics import YOLO

ball_model = YOLO("models/ball.pt")
pose_model = YOLO("models/yolov8m-pose.pt")


def detect_ball(frame):
    results = ball_model(frame)[0]

    balls = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])

        balls.append({
            "bbox": [x1, y1, x2, y2],
            "conf": conf
        })

    return balls


def detect_players(frame):
    results = pose_model(frame)[0]

    players = []

    if results.keypoints is not None:

        for kps in results.keypoints.xy:

            players.append({
                "keypoints": kps.tolist()
            })

    return players