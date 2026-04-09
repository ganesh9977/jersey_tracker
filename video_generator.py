"""
Jersey Tracker - Simulation Video Generator
=============================================
Creates a synthetic 90-second MP4 that simulates a room camera feed.
Three employees walk around wearing jerseys numbered  7,  23,  and  42.

  Person #7   – enters immediately, stays the entire video
                → will trigger the alert during the demo
  Person #23  – enters at  t=5 s, leaves at t=40 s, returns at t=60 s
  Person #42  – enters at t=18 s, leaves at t=70 s

Run directly:
    python video_generator.py
"""

import cv2
import numpy as np
import math
import sys


# ---------------------------------------------------------------------------
# Helper – draw a single "person" on the canvas
# ---------------------------------------------------------------------------
def _draw_person(frame, x: float, y: float,
                 jersey_color, number: str, width=90, height=170):
    x, y = int(x), int(y)

    # --- shadow ---
    shadow = np.array([[x - 3, y + height + 4],
                        [x + width + 3, y + height + 4],
                        [x + width, y + height],
                        [x, y + height]], np.int32)
    cv2.fillPoly(frame, [shadow], (160, 160, 160))

    # --- legs ---
    leg_w = width // 2 - 8
    cv2.rectangle(frame,
                  (x + 6, y + height),
                  (x + leg_w + 6, y + height + 55),
                  (40, 40, 110), -1)
    cv2.rectangle(frame,
                  (x + width - leg_w - 6, y + height),
                  (x + width - 6, y + height + 55),
                  (40, 40, 110), -1)

    # --- arms (same jersey colour) ---
    arm_y = y + 35
    cv2.rectangle(frame, (x - 16, arm_y), (x, arm_y + 65), jersey_color, -1)
    cv2.rectangle(frame, (x - 16, arm_y), (x, arm_y + 65), (20, 20, 20), 1)
    cv2.rectangle(frame, (x + width, arm_y), (x + width + 16, arm_y + 65),
                  jersey_color, -1)
    cv2.rectangle(frame, (x + width, arm_y), (x + width + 16, arm_y + 65),
                  (20, 20, 20), 1)

    # --- jersey body ---
    cv2.rectangle(frame, (x, y + 30), (x + width, y + height), jersey_color, -1)
    cv2.rectangle(frame, (x, y + 30), (x + width, y + height), (20, 20, 20), 2)

    # --- head ---
    head_cx = x + width // 2
    head_cy = y + 18
    cv2.circle(frame, (head_cx, head_cy), 24, (210, 180, 145), -1)  # skin
    cv2.circle(frame, (head_cx, head_cy), 24, (20, 20, 20), 2)
    # simple hair
    cv2.ellipse(frame, (head_cx, head_cy - 10), (24, 14), 0, 180, 360,
                (60, 40, 30), -1)

    # --- jersey number ---
    # Use FONT_HERSHEY_SIMPLEX (single-stroke, unambiguous glyphs) and render
    # as DARK INK ON WHITE BACKGROUND so EasyOCR reads "7" as "7", not "4".
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3.2 if len(number) == 1 else 2.1
    thickness  = 5

    (tw, th), baseline = cv2.getTextSize(number, font, font_scale, thickness)
    body_top    = y + 30
    body_bottom = y + height
    body_h      = body_bottom - body_top

    tx = x + (width - tw) // 2
    ty = body_top + (body_h + th) // 2

    # White backing rectangle – gives maximum OCR contrast regardless of jersey colour
    pad_x, pad_y = 14, 10
    cv2.rectangle(frame,
                  (tx - pad_x,      ty - th - pad_y),
                  (tx + tw + pad_x, ty + baseline + pad_y),
                  (255, 255, 255), -1)
    # Thin dark border around the white box (improves digit boundary detection)
    cv2.rectangle(frame,
                  (tx - pad_x,      ty - th - pad_y),
                  (tx + tw + pad_x, ty + baseline + pad_y),
                  (60, 60, 60), 2)

    # Dark ink number on white  (EasyOCR reads dark-on-light far more reliably)
    cv2.putText(frame, number, (tx, ty), font, font_scale, (15, 15, 15), thickness)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def create_simulation_video(output_path: str = "simulation.mp4",
                             width: int = 960,
                             height: int = 640,
                             fps: int = 30,
                             duration: int = 120):  # 120s → jersey #7 fires at 90s
    """
    Render the simulation and write it to *output_path*.

    Parameters
    ----------
    output_path : destination file (mp4 / avi)
    width, height : frame dimensions in pixels
    fps  : frames per second
    duration : total length in seconds
    """

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"[ERROR] Cannot open VideoWriter for '{output_path}'")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Person definitions
    # Each person has:
    #   number        – jersey number string
    #   color         – BGR jersey colour
    #   x, y          – starting position (top-left of bounding rect)
    #   vx, vy        – velocity (pixels / frame)
    #   visible_from  – seconds when person enters the scene
    #   visible_until – seconds when person leaves (-1 = stays forever)
    # ------------------------------------------------------------------
    W, H = width, height  # shorthand
    # Velocities are in pixels-per-frame at 30 fps.
    # Keep them low (≤ 0.4) so persons drift slowly around the room.
    persons = [
        dict(number="7",   color=(30, 30, 210),    # red jersey
             x=130.0,  y=200.0, vx=0.32,  vy=0.20,
             visible_from=0,    visible_until=9999),  # stays whole video → 90s alert
        dict(number="23",  color=(30, 190, 40),    # green jersey
             x=480.0,  y=160.0, vx=-0.26, vy=0.30,
             visible_from=5,    visible_until=60),   # leaves mid-video
        dict(number="42",  color=(200, 90, 30),    # orange jersey
             x=680.0,  y=280.0, vx=0.22,  vy=-0.26,
             visible_from=20,   visible_until=95),   # enters late, leaves near end
    ]

    person_w, person_h = 90, 170   # body dimensions (same for all)
    total_frames = fps * duration

    print(f"[•] Generating simulation: {output_path}  "
          f"({width}x{height} @ {fps}fps, {duration}s) …")

    for frame_idx in range(total_frames):
        t = frame_idx / fps

        # ---- background: a tiled floor + walls ----
        bg = np.full((height, width, 3), (210, 210, 210), dtype=np.uint8)

        # tile grid
        tile = 60
        for gx in range(0, width, tile):
            cv2.line(bg, (gx, 0), (gx, height), (190, 190, 190), 1)
        for gy in range(0, height, tile):
            cv2.line(bg, (0, gy), (width, gy), (190, 190, 190), 1)

        # wall / baseboard at top
        cv2.rectangle(bg, (0, 0), (width, 50), (170, 170, 170), -1)
        cv2.line(bg, (0, 50), (width, 50), (130, 130, 130), 2)

        # room label
        cv2.putText(bg, "CCTV FEED  |  JERSEY TRACKER SIMULATION",
                    (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (80, 80, 80), 2)

        # ---- update & draw each person ----
        for p in persons:
            if not (p["visible_from"] <= t < p["visible_until"]):
                continue

            # move
            p["x"] += p["vx"]
            p["y"] += p["vy"]

            # bounce off safe zone (keep person fully visible)
            margin = 20
            if p["x"] < margin:
                p["x"] = margin;          p["vx"] *= -1
            if p["x"] + person_w > W - margin:
                p["x"] = W - margin - person_w; p["vx"] *= -1
            if p["y"] < 55:
                p["y"] = 55;               p["vy"] *= -1
            if p["y"] + person_h + 60 > H - margin:
                p["y"] = H - margin - person_h - 60; p["vy"] *= -1

            _draw_person(bg, p["x"], p["y"],
                         p["color"], p["number"],
                         person_w, person_h)

        # ---- footer: plain gray text only (no saturated colours → no false OCR hits) ----
        # Thin gray progress bar (unsaturated → won't trigger blob detector)
        bar_len = int((t / duration) * (W - 20))
        cv2.rectangle(bg, (10, H - 5), (10 + bar_len, H - 2), (130, 130, 130), -1)

        out.write(bg)

        if frame_idx % (fps * 10) == 0:
            pct = int(100 * frame_idx / total_frames)
            print(f"    {pct:3d}%  frame {frame_idx}/{total_frames}", end="\r")

    out.release()
    print(f"\n[✓] Video saved → {output_path}  "
          f"({total_frames} frames, {total_frames / fps:.0f}s)")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate jersey-tracker simulation video")
    ap.add_argument("--output",   default="simulation.mp4",  help="Output file path")
    ap.add_argument("--width",    type=int, default=960,     help="Frame width")
    ap.add_argument("--height",   type=int, default=640,     help="Frame height")
    ap.add_argument("--fps",      type=int, default=30,      help="Frames per second")
    ap.add_argument("--duration", type=int, default=90,      help="Duration in seconds")
    args = ap.parse_args()

    create_simulation_video(
        output_path=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
    )
