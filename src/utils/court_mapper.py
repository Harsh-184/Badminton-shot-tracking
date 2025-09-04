import cv2
import numpy as np
import yaml

CANONICAL_W = 800
CANONICAL_H = 400

def compute_homography(pts_src):
    """
    pts_src: list[(x,y)] in image/frame order: BL, BR, TR, TL
    Maps to a canonical court rectangle (800x400).
    """
    dst = np.array([[0, CANONICAL_H - 1],
                    [CANONICAL_W - 1, CANONICAL_H - 1],
                    [CANONICAL_W - 1, 0],
                    [0, 0]], dtype=np.float32)
    src = np.array(pts_src, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, method=0)
    return H

def save_calibration(H, path):
    data = {"H": H.tolist(), "w": CANONICAL_W, "h": CANONICAL_H}
    with open(path, "w") as f:
        yaml.dump(data, f)

def load_calibration(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    H = np.array(data["H"], dtype=float)
    w = int(data.get("w", CANONICAL_W))
    h = int(data.get("h", CANONICAL_H))
    return H, w, h

def warp_point(pt, H):
    p = np.array([[pt[0], pt[1], 1.0]], dtype=np.float64).T
    wp = H @ p
    wp = wp / wp[2]
    return np.array([float(wp[0]), float(wp[1])])

def within_court(pt, w=CANONICAL_W, h=CANONICAL_H):
    x, y = pt
    return 0 <= x < w and 0 <= y < h
