"""
Jersey Tracker - Presence Tracking Module
==========================================
Thread-safe tracker that maintains per-jersey-number presence records
and exposes timing metrics to the rest of the application.

Key concepts
------------
continuous_time
    Seconds the person has been *continuously* in the frame
    (reset when they are absent for longer than absence_timeout).

total_time
    Cumulative seconds across all appearances (never reset).

alert_threshold
    When continuous_time >= alert_threshold the caller should fire an alert.

absence_timeout
    How many seconds of absence before we consider the person "gone" and
    reset their continuous timer.  A small value (3-5 s) is fine for the
    simulation because frames are delivered rapidly.
"""

import time
import threading
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PresenceRecord:
    """Mutable state for one jersey number."""

    __slots__ = (
        "jersey_number",
        "first_seen",
        "last_seen",
        "continuous_start",
        "continuous_time",
        "pre_absence_total",
        "total_time",
        "in_frame",
        "alert_sent",
        "alert_time",
    )

    def __init__(self, jersey_number: str, now: float):
        self.jersey_number      = jersey_number
        self.first_seen         = now
        self.last_seen          = now
        self.continuous_start   = now
        self.continuous_time    = 0.0
        self.pre_absence_total  = 0.0     # total before current run
        self.total_time         = 0.0
        self.in_frame           = True
        self.alert_sent         = False
        self.alert_time         = None    # ISO string when alert was sent

    def to_dict(self) -> dict:
        return {
            "jersey_number":    self.jersey_number,
            "continuous_time":  round(self.continuous_time, 1),
            "total_time":       round(self.total_time, 1),
            "in_frame":         self.in_frame,
            "alert_sent":       self.alert_sent,
            "alert_time":       self.alert_time,
            "first_seen":       datetime.fromtimestamp(self.first_seen).strftime("%H:%M:%S"),
            "last_seen":        datetime.fromtimestamp(self.last_seen).strftime("%H:%M:%S"),
        }


# ---------------------------------------------------------------------------
class PresenceTracker:
    """
    Call update(detected_numbers) on every processed frame.
    Query get_status() to obtain a JSON-serialisable snapshot.
    """

    def __init__(self,
                 alert_threshold:    int = 600,
                 absence_timeout:    int = 4):
        """
        Parameters
        ----------
        alert_threshold  : seconds until an alert is due  (default = 10 min)
        absence_timeout  : seconds of absence before the timer resets
        """
        self.alert_threshold  = alert_threshold
        self.absence_timeout  = absence_timeout

        self._records: dict[str, PresenceRecord] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    def update(self, detected_numbers: list[str]) -> dict:
        """
        Ingest the set of jersey numbers visible in the current frame.

        Returns
        -------
        dict  Serialisable status snapshot (same as get_status()).
        """
        with self._lock:
            now = time.time()

            # -- handle newly detected or returning jerseys --------------
            for number in detected_numbers:
                if number not in self._records:
                    # First time we see this jersey
                    rec = PresenceRecord(number, now)
                    self._records[number] = rec
                    logger.info(f"Jersey #{number} entered the room.")
                else:
                    rec = self._records[number]
                    if not rec.in_frame:
                        # Person just came back
                        absence = now - rec.last_seen
                        if absence > self.absence_timeout:
                            # Long enough absence → reset continuous timer
                            rec.continuous_start  = now
                            rec.continuous_time   = 0.0
                            rec.pre_absence_total = rec.total_time
                            logger.info(f"Jersey #{number} re-entered after "
                                        f"{absence:.0f}s absence. Timer reset.")
                        else:
                            # Brief blip – do NOT reset timer
                            logger.debug(f"Jersey #{number} blip ({absence:.1f}s), "
                                         "timer kept.")
                        rec.in_frame = True

                    rec.last_seen        = now
                    rec.continuous_time  = now - rec.continuous_start
                    rec.total_time       = rec.pre_absence_total + rec.continuous_time

            # -- mark jerseys no longer in frame ------------------------
            for number, rec in self._records.items():
                if rec.in_frame and number not in detected_numbers:
                    if now - rec.last_seen > self.absence_timeout:
                        rec.in_frame = False
                        logger.info(f"Jersey #{number} left the room "
                                    f"(was there for {rec.continuous_time:.0f}s).")

            return self.get_status()

    # ------------------------------------------------------------------
    def should_alert(self, jersey_number: str) -> bool:
        """True if *jersey_number* has been present long enough AND no alert sent yet."""
        with self._lock:
            rec = self._records.get(jersey_number)
            if rec is None:
                return False
            return (
                rec.in_frame
                and rec.continuous_time >= self.alert_threshold
                and not rec.alert_sent
            )

    def mark_alert_sent(self, jersey_number: str):
        """Call this after the alert has been dispatched."""
        with self._lock:
            rec = self._records.get(jersey_number)
            if rec:
                rec.alert_sent = True
                rec.alert_time = datetime.now().isoformat()
                logger.warning(f"Alert marked sent for Jersey #{jersey_number}.")

    # ------------------------------------------------------------------
    def get_record(self, jersey_number: str) -> dict:
        """Return a copy of the raw record for *jersey_number* (or {})."""
        with self._lock:
            rec = self._records.get(jersey_number)
            if rec is None:
                return {}
            return {
                "jersey_number":   rec.jersey_number,
                "continuous_time": rec.continuous_time,
                "total_time":      rec.total_time,
                "in_frame":        rec.in_frame,
                "alert_sent":      rec.alert_sent,
            }

    def get_status(self) -> dict:
        """Return a JSON-serialisable snapshot of all tracked jerseys."""
        with self._lock:
            return {num: rec.to_dict() for num, rec in self._records.items()}

    def cleanup_old_absent(self, max_absence_seconds: int = 30):
        """
        Remove records for jerseys that have been continuously absent for
        longer than *max_absence_seconds*.

        This keeps the UI panel clean – only jerseys that have appeared
        recently (or are currently in the room) are shown.
        """
        with self._lock:
            now = time.time()
            to_remove = [
                num for num, rec in self._records.items()
                if not rec.in_frame
                and (now - rec.last_seen) > max_absence_seconds
            ]
            for num in to_remove:
                logger.info(f"Pruning stale record for Jersey #{num} "
                            f"(absent {now - self._records[num].last_seen:.0f}s).")
                del self._records[num]

    def reset(self):
        """Clear all records (for testing)."""
        with self._lock:
            self._records.clear()
            logger.info("Tracker reset.")
