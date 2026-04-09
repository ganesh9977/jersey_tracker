"""
Jersey Tracker - Alert Manager
================================
Handles alert generation and history storage.

Currently implements **mock** email alerts: the alert is formatted like a
real email and printed to the console / log, but no SMTP connection is made.

To wire up real email later, replace _send_mock_email() with an SMTP or
SendGrid call – the rest of the application does not need to change.
"""

import logging
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Thread-safe alert dispatcher and history store.

    Parameters
    ----------
    email_to   : recipient address (shown in mock alerts)
    email_from : sender address (shown in mock alerts)
    """

    def __init__(self,
                 email_to:   str = "admin@company.com",
                 email_from: str = "alerts@jerseytracker.com"):
        self.email_to   = email_to
        self.email_from = email_from
        self._alerts: list[dict] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def send_alert(self,
                   jersey_number:   str,
                   duration_seconds: float,
                   room_id:         str = "Unknown Room") -> dict:
        """
        Create and store an alert for *jersey_number*.

        Parameters
        ----------
        jersey_number    : the jersey number string (e.g. "7")
        duration_seconds : how long the person has been present (continuous)
        room_id          : human-readable room identifier

        Returns
        -------
        dict  The alert record that was stored.
        """
        minutes   = duration_seconds / 60.0
        timestamp = datetime.now()

        subject = (
            f"[JERSEY ALERT] #{jersey_number} present for "
            f"{minutes:.1f} min in {room_id}"
        )
        body = (
            f"Automated alert from Jersey Tracker\n"
            f"{'─' * 52}\n"
            f"Room         : {room_id}\n"
            f"Jersey #     : {jersey_number}\n"
            f"Duration     : {int(duration_seconds // 60):02d}m "
            f"{int(duration_seconds % 60):02d}s\n"
            f"Threshold    : {int(duration_seconds // 60 + 1):02d} min\n"
            f"Triggered at : {timestamp.strftime('%Y-%m-%d  %H:%M:%S')}\n"
            f"{'─' * 52}\n"
            f"Please review the camera feed immediately.\n"
        )

        alert = {
            "id":               None,          # filled in below
            "jersey_number":    jersey_number,
            "room_id":          room_id,
            "duration_seconds": round(duration_seconds, 1),
            "duration_display": (
                f"{int(duration_seconds // 60):02d}m "
                f"{int(duration_seconds % 60):02d}s"
            ),
            "timestamp":        timestamp.isoformat(),
            "timestamp_display": timestamp.strftime("%Y-%m-%d  %H:%M:%S"),
            "status":           "sent",
            "recipient":        self.email_to,
            "sender":           self.email_from,
            "subject":          subject,
            "body":             body,
        }

        with self._lock:
            alert["id"] = len(self._alerts) + 1
            self._alerts.append(alert)

        self._send_mock_email(alert)
        return alert

    # ------------------------------------------------------------------
    def _send_mock_email(self, alert: dict):
        """
        Mock SMTP send – logs to console and logger.
        Replace with real SMTP / SendGrid logic here.
        """
        border = "=" * 62
        logger.warning(
            f"ALERT | Jersey #{alert['jersey_number']} | "
            f"Duration: {alert['duration_display']} | "
            f"Room: {alert['room_id']}"
        )
        print(f"\n{border}")
        print(f"  [EMAIL ALERT SENT]")
        print(f"  To      : {alert['recipient']}")
        print(f"  From    : {alert['sender']}")
        print(f"  Subject : {alert['subject']}")
        print(f"  ---")
        for line in alert["body"].splitlines():
            print(f"  {line}")
        print(f"{border}\n")

    # ------------------------------------------------------------------
    def get_alerts(self) -> list[dict]:
        """Return all alerts, most recent first."""
        with self._lock:
            return list(reversed(self._alerts))

    def get_alert_count(self) -> int:
        with self._lock:
            return len(self._alerts)

    def get_alerts_for_jersey(self, jersey_number: str) -> list[dict]:
        with self._lock:
            return [a for a in self._alerts
                    if a["jersey_number"] == jersey_number]
