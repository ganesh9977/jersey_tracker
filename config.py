"""
Jersey Tracker - Configuration
================================
All settings are readable from environment variables so the app
can be deployed to any environment without code changes.
"""

import os

# ---------------------------------------------------------------------------
# Video source
# ---------------------------------------------------------------------------
# Accepts:
#   "simulation.mp4"   - generated simulation file (default)
#   0                  - local webcam
#   "rtsp://..."       - IP camera RTSP stream
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "simulation.mp4")

# ---------------------------------------------------------------------------
# Room identifier
# ---------------------------------------------------------------------------
ROOM_ID = os.getenv("ROOM_ID", "Room A - CCTV 1")

# ---------------------------------------------------------------------------
# Detection settings
# ---------------------------------------------------------------------------
# "synthetic" - colour-blob + EasyOCR (works with the simulation video)
# "yolo"      - YOLOv8 person detection + EasyOCR (use for real cameras)
DETECTION_MODE = os.getenv("DETECTION_MODE", "synthetic")

YOLO_MODEL       = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_CONFIDENCE  = float(os.getenv("YOLO_CONFIDENCE", "0.5"))
OCR_CONFIDENCE   = float(os.getenv("OCR_CONFIDENCE", "0.3"))

# Minimum blob area (pixels²) for the synthetic detector
SYNTHETIC_MIN_AREA = int(os.getenv("SYNTHETIC_MIN_AREA", "2500"))

# Run full detection every N frames (higher = faster, less frequent updates)
DETECT_EVERY_N_FRAMES = int(os.getenv("DETECT_EVERY_N_FRAMES", "8"))

# ---------------------------------------------------------------------------
# Presence-tracking settings
# ---------------------------------------------------------------------------
# How many seconds a person must be present continuously to trigger an alert.
#   30  seconds  → demo / test mode   (default)
#   600 seconds  → production (10 min)
ALERT_THRESHOLD_SECONDS = int(os.getenv("ALERT_THRESHOLD", "90"))

# How many seconds of absence before we consider the person "gone" and
# reset their continuous timer.
ABSENCE_TIMEOUT_SECONDS = int(os.getenv("ABSENCE_TIMEOUT", "4"))

# ---------------------------------------------------------------------------
# Alert / email settings  (mock – no real SMTP needed)
# ---------------------------------------------------------------------------
ALERT_EMAIL_TO   = os.getenv("ALERT_EMAIL_TO",   "admin@company.com")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM",  "alerts@jerseytracker.com")

# ---------------------------------------------------------------------------
# Flask / server settings
# ---------------------------------------------------------------------------
FLASK_HOST  = os.getenv("FLASK_HOST",  "0.0.0.0")
FLASK_PORT  = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# MJPEG stream JPEG quality (1-100)
STREAM_JPEG_QUALITY = int(os.getenv("STREAM_JPEG_QUALITY", "80"))
