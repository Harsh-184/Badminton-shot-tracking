import argparse
import cv2
from src.utils.court_mapper import compute_homography, save_calibration

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="0 for webcam or path to video")
    ap.add_argument("--out", default="calib.yaml", help="Output calibration file")
    args = ap.parse_args()

    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("Cannot open source")

    ret, frame = cap.read()
    if not ret:
        raise SystemExit("Cannot read a frame for calibration")

    base = frame.copy()
    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

    win = "Click 4 court corners (BL, BR, TR, TL)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)

    instructions = "Click BL, BR, TR, TL. Press 'r' reset, 's' save, 'q' quit."
    while True:
        vis = frame.copy()
        cv2.putText(vis, instructions, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for i, p in enumerate(points):
            cv2.putText(vis, f"{i+1}", (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            points = []
            frame = base.copy()
        if key == ord('s') and len(points) == 4:
            H = compute_homography(points)
            save_calibration(H, args.out)
            print(f"Saved calibration to {args.out}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
