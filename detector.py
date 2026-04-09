"""
Jersey Tracker - Detection Module
===================================
Two detector implementations:

  SyntheticPersonDetector
      Uses HSV-saturation contour detection to locate coloured person shapes
      in the simulation video, then runs EasyOCR to read the jersey number.
      No YOLO needed – works out of the box with simulation.mp4.

  YOLOJerseyDetector
      Uses YOLOv8 (ultralytics) to detect *real* people, crops the torso
      region, then runs EasyOCR on it.  Use this for live IP cameras.

  create_detector(mode)
      Factory that returns the right implementation based on config.
"""

import cv2
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _preprocess_for_ocr(region: np.ndarray) -> np.ndarray:
    """
    Prepare a region for EasyOCR.

    Optimised for dark-ink-on-white-background jersey numbers
    (as rendered by the simulation video generator):
      1. Grayscale
      2. Upscale if too small (EasyOCR drops accuracy below ~60 px height)
      3. Sharpen to make digit strokes crisp
      4. Binary threshold – converts to pure black / white which EasyOCR
         handles most reliably and removes any jersey-colour bleed-through
    """
    if region.size == 0:
        return region

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # --- upscale if needed ---
    if gray.shape[0] < 80 or gray.shape[1] < 50:
        scale = max(2, 100 // max(gray.shape[0], 1))
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    # --- sharpening kernel ---
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float32)
    sharp = cv2.filter2D(gray, -1, kernel)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)

    # --- Otsu binarisation: dark text → black, white bg → white ---
    _, binary = cv2.threshold(sharp, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _extract_number(results, min_confidence: float) -> str | None:
    """
    Pick the best jersey-number token from EasyOCR results.
    Accepts 1–2 digit integers in the range 1–99 (standard jersey numbers).
    Rejects 0, 3-digit numbers (timestamps/frame-counts), and all non-digit strings.
    """
    candidates = []
    for (_, text, conf) in results:
        text = text.strip().replace(" ", "")
        if not text.isdigit():
            continue
        if conf < min_confidence:
            continue
        val = int(text)
        # Valid jersey range: 1–99  (excludes 0, 100+ which are timestamp artefacts)
        if 1 <= val <= 99:
            candidates.append((conf, text))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None


# ---------------------------------------------------------------------------
# Synthetic detector  (simulation video)
# ---------------------------------------------------------------------------
class SyntheticPersonDetector:
    """
    Detects coloured person-shaped blobs in the simulation video and reads
    jersey numbers with EasyOCR.

    Strategy
    --------
    1. Build a saturation mask (HSV S-channel > threshold).
    2. Find large contours (area > min_area) whose bounding rect is taller
       than it is wide (person-shaped).
    3. Merge bounding rects that are close to each other (arms + body split).
    4. For each merged region, apply OCR to the centre 60 % of its height
       (the jersey body, away from head and legs).
    """

    # A person-sized blob must be at least this tall (pixels).
    # Progress bars, thin lines, etc. are << 80 px and are filtered out.
    MIN_BLOB_HEIGHT = 80

    # Ignore blobs whose TOP is within this many pixels of the frame edges.
    # Keeps the header label zone and the bottom timestamp strip clean.
    EDGE_MARGIN = 65

    # Temporal voting: a blob position (grid cell) keeps the last N OCR reads.
    # The most-voted number wins.  Prevents single-frame misreads from creating
    # spurious jersey records (e.g. "4" instead of "7" for one frame).
    VOTE_HISTORY   = 5   # frames to keep per blob slot
    VOTE_THRESHOLD = 2   # minimum votes needed to trust a number

    def __init__(self, min_area: int = 4000, ocr_confidence: float = 0.35):
        self.min_area       = min_area
        self.ocr_confidence = ocr_confidence
        self._init_ocr()
        # _vote_buf: grid_key → deque of (number | None) readings
        self._vote_buf: dict = {}

    def _init_ocr(self):
        try:
            import easyocr
            logger.info("Loading EasyOCR model …")
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR ready.")
            self.ocr_available = True
        except ImportError:
            logger.warning("EasyOCR not installed – jersey numbers won't be read.")
            self.reader = None
            self.ocr_available = False

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[dict]:
        """Return list of detections: {bbox, jersey_number, confidence}."""
        h, w = frame.shape[:2]

        # ---- saturation mask ----
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat  = hsv[:, :, 1]
        _, mask = cv2.threshold(sat, 75, 255, cv2.THRESH_BINARY)

        # Remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # ---- find contours ----
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)

            # ── Guard 1: must be tall enough to be a person ──────────
            if bh < self.MIN_BLOB_HEIGHT:
                continue

            # ── Guard 2: reject blobs in the header / footer strips ──
            # (prevents reading numbers from timestamp text near edges)
            if y < self.EDGE_MARGIN or (y + bh) > (h - self.EDGE_MARGIN):
                continue

            # ── Guard 3: taller than wide (person body shape) ────────
            if bh >= bw * 0.8:
                rects.append([x, y, x + bw, y + bh])

        if not rects:
            return []

        # ---- merge nearby rects (handles split blobs) ----
        merged = self._merge_rects(rects, gap=30)

        # ---- OCR on each merged region + temporal voting ----
        # Snap each blob to a coarse grid cell (80×80 px) to get a stable key
        # even when the person moves a few pixels between frames.
        from collections import deque, Counter

        detections       = []
        active_keys: set = set()

        for (x1, y1, x2, y2) in merged:
            # Expand vertically to include head
            pad_top    = 45
            pad_bottom = 60
            rx1 = max(0, x1 - 20)
            ry1 = max(0, y1 - pad_top)
            rx2 = min(w, x2 + 20)
            ry2 = min(h, y2 + pad_bottom)

            # Jersey body = middle 35 %–85 % of person height
            person_h   = ry2 - ry1
            jersey_y1  = ry1 + int(person_h * 0.35)
            jersey_y2  = ry1 + int(person_h * 0.85)
            jersey_roi = frame[jersey_y1:jersey_y2, rx1:rx2]

            raw_number = self._ocr_region(jersey_roi)

            # ── voting: accumulate OCR reads per blob grid-slot ───────
            grid_key = (x1 // 80, y1 // 80)
            active_keys.add(grid_key)

            buf = self._vote_buf.setdefault(
                grid_key, deque(maxlen=self.VOTE_HISTORY))
            buf.append(raw_number)

            # Count non-None votes and pick winner if it meets the threshold
            counts = Counter(v for v in buf if v is not None)
            if counts:
                winner, votes = counts.most_common(1)[0]
                number = winner if votes >= self.VOTE_THRESHOLD else None
            else:
                number = None

            detections.append({
                "bbox":          (rx1, ry1, rx2, ry2),
                "jersey_number": number,
                "confidence":    1.0,
            })

        # Prune stale vote-buffer slots (blobs that left the frame)
        for key in list(self._vote_buf):
            if key not in active_keys:
                del self._vote_buf[key]

        return detections

    # ------------------------------------------------------------------
    def _merge_rects(self, rects: list, gap: int = 30) -> list:
        """Union-find merge of rectangles whose expanded bounds overlap."""
        if not rects:
            return []
        rects = sorted(rects, key=lambda r: r[0])
        merged = [rects[0]]
        for cur in rects[1:]:
            prev = merged[-1]
            # Expand prev by gap to check overlap
            if cur[0] <= prev[2] + gap and cur[1] <= prev[3] + gap \
                    and cur[2] >= prev[0] - gap:
                merged[-1] = [
                    min(prev[0], cur[0]),
                    min(prev[1], cur[1]),
                    max(prev[2], cur[2]),
                    max(prev[3], cur[3]),
                ]
            else:
                merged.append(cur)
        return merged

    # ------------------------------------------------------------------
    def _ocr_region(self, region: np.ndarray) -> str | None:
        """Run EasyOCR on *region* and return best numeric token."""
        if not self.ocr_available or region.size == 0:
            return None
        try:
            processed = _preprocess_for_ocr(region)
            results   = self.reader.readtext(processed, detail=1,
                                             paragraph=False,
                                             allowlist='0123456789')
            return _extract_number(results, self.ocr_confidence)
        except Exception as exc:
            logger.debug(f"OCR error: {exc}")
            return None


# ---------------------------------------------------------------------------
# YOLO + EasyOCR detector  (real cameras / IP cameras)
# ---------------------------------------------------------------------------
class YOLOJerseyDetector:
    """
    Production detector for live camera feeds.

    1. YOLOv8 (class=person) localises each employee.
    2. EasyOCR reads the jersey number from the torso crop.
    """

    def __init__(self,
                 model_path:      str   = "yolov8n.pt",
                 yolo_confidence: float = 0.5,
                 ocr_confidence:  float = 0.3):
        self.yolo_confidence = yolo_confidence
        self.ocr_confidence  = ocr_confidence
        self._init_yolo(model_path)
        self._init_ocr()

    def _init_yolo(self, model_path: str):
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLOv8 model: {model_path} …")
            self.yolo = YOLO(model_path)
            logger.info("YOLOv8 ready.")
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise

    def _init_ocr(self):
        try:
            import easyocr
            logger.info("Loading EasyOCR model …")
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR ready.")
        except ImportError:
            logger.error("EasyOCR not installed. Run: pip install easyocr")
            raise

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[dict]:
        results     = self.yolo(frame, classes=[0], verbose=False)
        detections  = []

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.yolo_confidence:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ph = y2 - y1

                # Torso crop: 35 % – 85 % of person height
                t_y1 = y1 + int(ph * 0.35)
                t_y2 = y1 + int(ph * 0.85)
                torso = frame[t_y1:t_y2, x1:x2]

                number = self._ocr_torso(torso)

                detections.append({
                    "bbox":          (x1, y1, x2, y2),
                    "jersey_number": number,
                    "confidence":    conf,
                })

        return detections

    # ------------------------------------------------------------------
    def _ocr_torso(self, region: np.ndarray) -> str | None:
        if region.size == 0:
            return None
        try:
            processed = _preprocess_for_ocr(region)
            results   = self.reader.readtext(processed, detail=1,
                                             paragraph=False,
                                             allowlist='0123456789')
            return _extract_number(results, self.ocr_confidence)
        except Exception as exc:
            logger.debug(f"OCR error: {exc}")
            return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_detector(mode: str = "synthetic",
                    model_path:      str   = "yolov8n.pt",
                    yolo_confidence: float = 0.5,
                    ocr_confidence:  float = 0.3):
    """
    Return the appropriate detector.

    Parameters
    ----------
    mode : "synthetic" | "yolo"
    """
    mode = mode.lower()
    if mode == "yolo":
        logger.info("Creating YOLOJerseyDetector (production mode)")
        return YOLOJerseyDetector(model_path, yolo_confidence, ocr_confidence)
    else:
        logger.info("Creating SyntheticPersonDetector (simulation mode)")
        return SyntheticPersonDetector(ocr_confidence=ocr_confidence)
