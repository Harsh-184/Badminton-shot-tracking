Badminton Shot Tracking

Detects shuttle shot contact points from webcam or video, maps them onto a canonical badminton court via homography, and exports a heatmap, CSV, and optional annotated video. Built with OpenCV + MediaPipe Pose.

Features

4-click court calibration → homography to a canonical court

Motion-based shuttle candidate + pose-derived “impact line” heuristic

Outputs: contacts.csv, heatmap.png, annotated.mp4

Works with webcam (--source 0) or video files

Setup

Use a Python 3.11 virtual environment and install requirements:

python -m venv .venv
# Windows
.\.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt

Calibrate the court (once per camera/view)

Opens the first frame; click corners in order Bottom-Left → Bottom-Right → Top-Right → Top-Left, then press s to save.

# Webcam (index 0)
python -m scripts.calibrate_court --source 0 --out calib.yaml

# OR a video file
python -m scripts.calibrate_court --source "C:\path\to\match.mp4" --out calib.yaml

Run the tracker
# Live webcam
python -m src.main --source 0 --calib calib.yaml --outdir output --save-video

# OR a video file
python -m src.main --source "C:\path\to\match.mp4" --calib calib.yaml --outdir output --save-video


Outputs (in output/):

contacts.csv — time_s, frame_idx, player, court_x, court_y

heatmap.png — shot contact heatmap

annotated.mp4 — overlay video (when --save-video)

Tuning (in src/main.py)

Proximity threshold (distance from shuttle to impact line): d < 25 → increase (e.g., 35) for more hits, decrease for stricter matches.

Impulse threshold (speed change around contact): speed > 2.0 → lower (e.g., 1.5) for sensitivity, raise to reduce false positives.
