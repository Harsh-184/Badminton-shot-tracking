import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

class PoseTracker:
    """
    Wraps MediaPipe Pose to return key landmarks needed to approximate racket reach.
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self):
        self.pose.close()

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(frame_rgb)
        if not res.pose_landmarks:
            return None

        lm = res.pose_landmarks.landmark

        def pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

        key = {
            "left_wrist": pt(mp_pose.PoseLandmark.LEFT_WRIST.value),
            "right_wrist": pt(mp_pose.PoseLandmark.RIGHT_WRIST.value),
            "left_elbow": pt(mp_pose.PoseLandmark.LEFT_ELBOW.value),
            "right_elbow": pt(mp_pose.PoseLandmark.RIGHT_ELBOW.value),
            "left_shoulder": pt(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
            "right_shoulder": pt(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
        }

        def impact_line(elbow, wrist, extend=1.5):
            # Extend elbow→wrist outward to approximate racket reach.
            v = wrist - elbow
            if np.linalg.norm(v) < 1e-2:
                return None
            start = wrist
            end = wrist + extend * v
            return (start, end)

        lines = {
            "left_arm_line": impact_line(key["left_elbow"], key["left_wrist"]),
            "right_arm_line": impact_line(key["right_elbow"], key["right_wrist"]),
        }

        return {"keypoints": key, "lines": lines}
