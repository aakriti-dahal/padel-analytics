🎾 Padel Game Analytics — Shot Classification System

A computer vision pipeline that analyzes padel gameplay videos to detect players, track the ball, and classify shots (forehand, backhand, smash) using pose estimation and rule-based logic.

Built as part of the Layman AI assignment to demonstrate practical understanding of computer vision, object detection, and video analytics.

------------------------------------------------------------

🎯 Objective

Given a padel match video, the system:

- Detects ball and players
- Extracts pose keypoints using YOLOv8 pose model
- Classifies shots into:
  - Forehand
  - Backhand
  - Smash
  - Unknown
- Generates structured outputs for analysis

------------------------------------------------------------

🧠 Approach

1. Object Detection
- Ball detection using a pretrained Hugging Face model (ball.pt)
- Player detection using YOLOv8m-pose

2. Pose-Based Feature Extraction
- Extracts key landmarks:
  - wrists
  - shoulders
- Uses spatial relationships between joints for classification

3. Shot Classification (Rule-Based)
- Forehand → wrist right of shoulder
- Backhand → wrist left of shoulder
- Smash → wrist above shoulder

4. Temporal Stabilization
- Applies a 10-frame lock system
- Prevents flickering between predictions

5. Video Analytics Pipeline
- Frame-by-frame processing
- Output visualization + structured logging

------------------------------------------------------------

⚙️ Requirements

ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0

Install dependencies:
pip install -r requirements.txt

------------------------------------------------------------

▶️ How to Run

python main.py

Make sure:
- input.mp4 is placed in the root directory
- Pretrained models are inside /models

------------------------------------------------------------

📤 Outputs

1. Annotated Video
output/annotated.mp4

Includes:
- Ball bounding boxes
- Pose keypoints
- Shot classification overlay
- Frame number + timestamp

------------------------------------------------------------

2. CSV Output
output/shots.csv

Example:

frame | timestamp | shot_type
12    | 0.40      | forehand
25    | 0.83      | smash

------------------------------------------------------------

3. JSON Output
output/shots.json

Example:

[
  {
    "frame": 12,
    "timestamp": 0.4,
    "shot_type": "forehand"
  }
]

------------------------------------------------------------

📊 Features

- Player and ball detection
- Pose estimation using pretrained models
- Shot classification (forehand, backhand, smash)
- Temporal smoothing for stable predictions
- Video annotation output
- Structured data export (CSV + JSON)

------------------------------------------------------------

⚠️ Limitations

- Rule-based classification (not a deep learning model)
- Accuracy depends on pose detection quality
- Ball detection may fail in fast motion scenes
- No player identity tracking

------------------------------------------------------------

🔮 Future Improvements

- Add player tracking (DeepSORT / ByteTrack)
- Improve shot classification using ML models
- Add rally detection (game phase analysis)
- Build dashboard for visualization

------------------------------------------------------------

👨‍💻 Tech Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- Pretrained Hugging Face model
- NumPy
- Pandas
