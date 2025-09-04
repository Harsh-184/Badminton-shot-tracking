import argparse
import os
import csv
import cv2
import numpy as np

from src.utils.pose_tracker import PoseTracker
from src.utils.shuttle_tracker import ShuttleTracker
from src.utils.court_mapper import load_calibration, warp_point, within_court
from src.utils.heatmap import CourtHeatmap

def line_point_distance(p1, p2, p0):
    """Distance from point p0 to line segment p1-p2 (and projection)."""
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p0 = np.array(p0, dtype=float)
    d = p2 - p1
    if np.allclose(d, 0):
        return np.linalg.norm(p0 - p1), p1
    t = np.clip(np.dot(p0 - p1, d) / np.dot(d, d), 0.0, 1.0)
    proj = p1 + t * d
    return float(np.linalg.norm(p0 - proj)), proj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="0 for webcam or path to video")
    ap.add_argument("--calib", required=True, help="calibration YAML path")
    ap.add_argument("--outdir", default="output", help="output directory")
    ap.add_argument("--save-video", action="store_true", help="save annotated mp4")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    H, court_w, court_h = load_calibration(args.calib)
    heatmap = CourtHeatmap(width=court_w, height=court_h, bins_x=16, bins_y=8)

    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("Cannot open source")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(os.path.join(args.outdir, "annotated.mp4"),
                                 fourcc, fps, (vw, vh))

    pose = PoseTracker(0.5, 0.5)
    shuttle = ShuttleTracker()

    contacts_csv = os.path.join(args.outdir, "contacts.csv")
    with open(contacts_csv, "w", newline="") as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow(["time_s", "frame_idx", "player", "court_x", "court_y"])

        frame_idx = 0
        last_contact_time = 0.0
        cooldown = 0.25  # seconds

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            t_s = frame_idx / fps

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            pose_data = pose.process(frame)
            lines = pose_data["lines"] if pose_data else {}

            candidate, trace = shuttle.update(gray)

            player_hit = None
            contact_point = None
            if candidate is not None and lines:
                dists = []
                for name, line in lines.items():
                    if line is None:
                        continue
                    d, proj = line_point_distance(line[0], line[1], candidate)
                    dists.append((d, name, proj))

                if dists:
                    d, name, proj = min(dists, key=lambda x: x[0])
                    speed = ShuttleTracker.speed(trace, window=5)
                    if d < 25 and speed > 2.0 and (t_s - last_contact_time) > cooldown:
                        player_hit = "left" if "left" in name else "right"
                        contact_point = proj
                        last_contact_time = t_s

            if contact_point is not None:
                court_pt = warp_point(contact_point, H)
                if within_court(court_pt, court_w, court_h):
                    heatmap.add(court_pt[0], court_pt[1])
                    wcsv.writerow([f"{t_s:.3f}", frame_idx, player_hit,
                                   f"{court_pt[0]:.2f}", f"{court_pt[1]:.2f}"])

            # Draw overlays
            vis = frame.copy()
            for name, line in lines.items() if lines else []:
                if line is None:
                    continue
                cv2.line(vis, tuple(map(int, line[0])), tuple(map(int, line[1])),
                         (0, 255, 255), 2)
            for p in trace:
                cv2.circle(vis, tuple(map(int, p)), 2, (255, 255, 255), -1)
            if candidate is not None:
                cv2.circle(vis, tuple(map(int, candidate)), 4, (0, 0, 255), -1)

            cv2.putText(vis, f"frame {frame_idx}  t={t_s:.2f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)

            if writer is not None:
                writer.write(vis)

            cv2.imshow("Badminton Shot Tracking (q=quit)", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    heatmap_path = os.path.join(args.outdir, "heatmap.png")
    heatmap.render(heatmap_path)

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    if writer is not None:
        writer.release()

    print(f"[Saved] {contacts_csv}")
    print(f"[Saved] {heatmap_path}")
    if writer is not None:
        print(f"[Saved] {os.path.join(args.outdir, 'annotated.mp4')}")

if __name__ == "__main__":
    main()
