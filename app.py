"""
Jersey Tracker - Flask Application
=====================================
Orchestrates the video-processing pipeline and exposes:

  GET  /               → HTML dashboard
  GET  /video_feed     → MJPEG live stream
  GET  /api/status     → JSON: current jersey tracking state
  GET  /api/alerts     → JSON: alert history
  GET  /api/config     → JSON: current runtime config

Start with:
    python app.py

Or with overrides:
    ALERT_THRESHOLD=600 DETECTION_MODE=yolo python app.py
"""

import cv2
import threading
import time
import logging
import sys
import os

from flask import Flask, render_template, Response, jsonify

# ---- local modules ----
import config
from detector      import create_detector
from tracker       import PresenceTracker
from alert_manager import AlertManager

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Shared state  (written by background thread, read by Flask routes)
# ---------------------------------------------------------------------------
_frame_lock    = threading.Lock()
_current_frame = None   # type: ignore[assignment]  (cv2 Mat or None)
_processing_ok = False          # True once first frame is processed

tracker       = PresenceTracker(
    alert_threshold=config.ALERT_THRESHOLD_SECONDS,
    absence_timeout=config.ABSENCE_TIMEOUT_SECONDS,
)
alert_manager = AlertManager(
    email_to=config.ALERT_EMAIL_TO,
    email_from=config.ALERT_EMAIL_FROM,
)


# ---------------------------------------------------------------------------
# Annotation helper
# ---------------------------------------------------------------------------
def _draw_annotations(frame, detections: list, status: dict):
    """Overlay bounding boxes, jersey timers, and alert banners."""
    h, w = frame.shape[:2]

    # ---- per-person overlays ----
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        number          = det.get("jersey_number")
        rec             = status.get(number, {}) if number else {}

        alert_sent  = rec.get("alert_sent",       False)
        cont_time   = rec.get("continuous_time",  0)
        in_frame    = rec.get("in_frame",          True)

        # colour coding
        if alert_sent:
            box_color = (0, 0, 220)       # red  → alert fired
        elif number:
            box_color = (0, 210, 0)       # green → identified
        else:
            box_color = (0, 140, 255)     # orange → unidentified

        # bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        if number:
            mins = int(cont_time // 60)
            secs = int(cont_time % 60)
            label = f"#{number}  {mins:02d}:{secs:02d}"

            # label background
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
            cv2.rectangle(frame,
                          (x1, max(0, y1 - lh - 12)),
                          (x1 + lw + 10, y1),
                          box_color, -1)
            cv2.putText(frame, label,
                        (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65,
                        (255, 255, 255), 2)

            # alert banner below box
            if alert_sent:
                banner = " ALERT SENT "
                (bw, bh), _ = cv2.getTextSize(
                    banner, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
                cv2.rectangle(frame,
                              (x1, y2),
                              (x1 + bw + 6, y2 + bh + 8),
                              (0, 0, 180), -1)
                cv2.putText(frame, banner,
                            (x1 + 3, y2 + bh + 2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6,
                            (255, 255, 255), 2)

    # ---- HUD: top-left ----
    threshold_label = (
        f"Threshold: {config.ALERT_THRESHOLD_SECONDS}s  |  "
        f"Mode: {config.DETECTION_MODE.upper()}"
    )
    cv2.putText(frame, threshold_label,
                (8, h - 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (230, 230, 230), 1)

    # ---- HUD: bottom bar ----
    active  = sum(1 for s in status.values() if s.get("in_frame"))
    alerts  = alert_manager.get_alert_count()
    info    = f"Active: {active}  |  Alerts: {alerts}  |  Room: {config.ROOM_ID}"
    cv2.putText(frame, info,
                (8, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)

    # ---- HUD: timestamp (bottom-right) ----
    ts = time.strftime("%Y-%m-%d  %H:%M:%S")
    (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, ts,
                (w - tw - 10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)

    return frame


# ---------------------------------------------------------------------------
# Background video-processing thread
# ---------------------------------------------------------------------------
def video_processing_loop():
    global _current_frame, _processing_ok

    # ---- verify video source ----
    source = config.VIDEO_SOURCE
    logger.info(f"Opening video source: {source!r}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"Cannot open video source '{source}'.")
        logger.error("Tip: run  python video_generator.py  to create simulation.mp4")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay  = 1.0 / fps
    detect_every = config.DETECT_EVERY_N_FRAMES
    frame_idx    = 0

    logger.info(f"Initialising detector (mode={config.DETECTION_MODE!r}) …")
    detector = create_detector(
        mode=config.DETECTION_MODE,
        model_path=config.YOLO_MODEL,
        yolo_confidence=config.YOLO_CONFIDENCE,
        ocr_confidence=config.OCR_CONFIDENCE,
    )
    logger.info("Detector ready. Starting processing loop.")

    last_detections: list[dict] = []
    last_cleanup = time.time()   # periodically prune stale "away" records

    while True:
        loop_start = time.time()

        # ---- prune absent jerseys every 15 s (keeps UI panel clean) ----
        if loop_start - last_cleanup > 15:
            tracker.cleanup_old_absent(max_absence_seconds=20)
            last_cleanup = loop_start

        ret, frame = cap.read()
        if not ret:
            # Loop the video file
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            time.sleep(0.1)
            continue

        frame_idx += 1

        # ---- run detection every N frames (throttle OCR cost) ----
        if frame_idx % detect_every == 0:
            try:
                last_detections = detector.detect(frame)
            except Exception as exc:
                logger.error(f"Detection error: {exc}", exc_info=True)
                last_detections = []

        # ---- collect jersey numbers that are currently visible ----
        visible_numbers = [
            d["jersey_number"]
            for d in last_detections
            if d.get("jersey_number")
        ]

        # ---- update presence tracker ----
        status = tracker.update(visible_numbers)

        # ---- fire any pending alerts ----
        for number in visible_numbers:
            if tracker.should_alert(number):
                rec = tracker.get_record(number)
                alert_manager.send_alert(
                    jersey_number=number,
                    duration_seconds=rec.get("continuous_time", 0),
                    room_id=config.ROOM_ID,
                )
                tracker.mark_alert_sent(number)

        # ---- annotate frame ----
        annotated = _draw_annotations(frame.copy(), last_detections, status)

        # ---- store frame for streaming ----
        with _frame_lock:
            _current_frame = annotated
        _processing_ok = True

        # ---- pace to source fps ----
        elapsed = time.time() - loop_start
        sleep   = max(0.0, frame_delay - elapsed)
        if sleep:
            time.sleep(sleep)


# ---------------------------------------------------------------------------
# MJPEG generator
# ---------------------------------------------------------------------------
def _mjpeg_generator():
    """Yield MJPEG-encoded frames for the /video_feed route."""
    quality = [int(cv2.IMWRITE_JPEG_QUALITY), config.STREAM_JPEG_QUALITY]

    while True:
        with _frame_lock:
            frame = _current_frame

        if frame is None:
            # Not ready yet – send a "please wait" placeholder
            time.sleep(0.1)
            continue

        ret, buf = cv2.imencode('.jpg', frame, quality)
        if not ret:
            time.sleep(0.05)
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buf.tobytes()
            + b'\r\n'
        )
        time.sleep(1.0 / 25)   # cap MJPEG at 25 fps to the browser


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template(
        'index.html',
        room_id=config.ROOM_ID,
        alert_threshold=config.ALERT_THRESHOLD_SECONDS,
        detection_mode=config.DETECTION_MODE,
    )


@app.route('/video_feed')
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/api/status')
def api_status():
    return jsonify({
        "jerseys":          tracker.get_status(),
        "alert_count":      alert_manager.get_alert_count(),
        "alert_threshold":  config.ALERT_THRESHOLD_SECONDS,
        "processing_ok":    _processing_ok,
        "server_time":      time.strftime("%H:%M:%S"),
        "room_id":          config.ROOM_ID,
    })


@app.route('/api/alerts')
def api_alerts():
    return jsonify(alert_manager.get_alerts())


@app.route('/api/config')
def api_config():
    return jsonify({
        "video_source":             str(config.VIDEO_SOURCE),
        "detection_mode":           config.DETECTION_MODE,
        "alert_threshold_seconds":  config.ALERT_THRESHOLD_SECONDS,
        "absence_timeout_seconds":  config.ABSENCE_TIMEOUT_SECONDS,
        "room_id":                  config.ROOM_ID,
        "yolo_model":               config.YOLO_MODEL,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Validate video source exists (for file-based sources)
    source = config.VIDEO_SOURCE
    if isinstance(source, str) and not source.startswith(("rtsp://", "http")):
        if not os.path.isfile(source):
            logger.warning(
                f"Video file '{source}' not found. "
                "Run  python video_generator.py  to create it."
            )

    # Start background processing thread
    proc_thread = threading.Thread(
        target=video_processing_loop,
        daemon=True,
        name="VideoProcessor",
    )
    proc_thread.start()
    logger.info(f"Background processing thread started.")

    # Start Flask
    logger.info(
        f"Dashboard → http://localhost:{config.FLASK_PORT}  "
        f"(alert in {config.ALERT_THRESHOLD_SECONDS}s, "
        f"mode={config.DETECTION_MODE})"
    )
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
        threaded=True,
        use_reloader=False,   # prevents double-starting the background thread
    )
