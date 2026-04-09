/**
 * Jersey Tracker – Frontend Logic
 * ================================
 * Polls /api/status every 2 seconds and /api/alerts every 3 seconds.
 * Locally interpolates the timers every second so the UI feels live
 * even between API polls.
 */

"use strict";

// ── State ──────────────────────────────────────────────────────────
const state = {
  jerseys:        {},    // jersey_number → { ...apiData, _localTick }
  alerts:         [],
  alertThreshold: 600,   // overwritten from API
  hiddenAlerts:   new Set(),
};

let pollTimer    = null;
let alertTimer   = null;
let tickInterval = null;

// ── Cached DOM refs ────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Utility ───────────────────────────────────────────────────────
function fmtTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function lerp(a, b, t) { return a + (b - a) * t; }

// ── API calls ─────────────────────────────────────────────────────
async function fetchStatus() {
  try {
    const res  = await fetch("/api/status");
    const data = await res.json();

    state.alertThreshold = data.alert_threshold || 600;

    // Merge API data into local state, preserving _localStart for interpolation
    const incoming = data.jerseys || {};
    for (const [num, rec] of Object.entries(incoming)) {
      const prev = state.jerseys[num];
      state.jerseys[num] = {
        ...rec,
        _localStart: prev?._localStart ?? Date.now() / 1000,
        _apiTime:    rec.continuous_time,
        _apiAt:      Date.now() / 1000,
      };
    }

    // Mark jerseys no longer in the API response as away
    for (const num of Object.keys(state.jerseys)) {
      if (!incoming[num]) {
        state.jerseys[num].in_frame = false;
      }
    }

    // Update live badge + server time
    const badge = $("live-badge");
    if (badge) {
      badge.className = "live-badge";
      badge.innerHTML = '<span class="live-dot"></span> LIVE';
    }
    if ($("server-time") && data.server_time) {
      $("server-time").textContent = data.server_time;
    }

  } catch (err) {
    console.warn("Status fetch failed:", err);
    const badge = $("live-badge");
    if (badge) {
      badge.className = "live-badge offline";
      badge.innerHTML = '<span class="live-dot"></span> OFFLINE';
    }
  }
}

async function fetchAlerts() {
  try {
    const res  = await fetch("/api/alerts");
    const data = await res.json();
    state.alerts = data;
    renderAlerts();
  } catch (err) {
    console.warn("Alerts fetch failed:", err);
  }
}

// ── Render jersey cards ───────────────────────────────────────────
function renderCards() {
  const container = $("jersey-cards");
  if (!container) return;

  const entries = Object.entries(state.jerseys);

  if (entries.length === 0) {
    container.innerHTML =
      '<div class="empty-state" id="empty-jerseys">' +
      '<div class="empty-icon">👀</div><div>Scanning for jerseys…</div></div>';
    updateSummaryBadges(0);
    return;
  }

  // Sort: in-frame first, then by jersey number
  entries.sort(([, a], [, b]) => {
    if (a.in_frame !== b.in_frame) return a.in_frame ? -1 : 1;
    return parseInt(a.jersey_number) - parseInt(b.jersey_number);
  });

  const activeCount = entries.filter(([, r]) => r.in_frame).length;
  updateSummaryBadges(activeCount);

  let html = "";
  for (const [num, rec] of entries) {
    const contTime = interpolatedTime(rec);
    const pct      = Math.min(100, (contTime / state.alertThreshold) * 100);
    const barClass = pct >= 100 ? "danger" : pct >= 75 ? "warn" : "";

    let cardClass  = rec.in_frame ? (rec.alert_sent ? "alert" : "in-room") : "away";
    let statusHtml = "";

    if (rec.alert_sent) {
      statusHtml = '<span class="card-status status-alert">⚠ Alert Sent</span>';
    } else if (rec.in_frame) {
      statusHtml = '<span class="card-status status-inroom">In Room</span>';
    } else {
      statusHtml = '<span class="card-status status-away">Away</span>';
    }

    const alertBadge = rec.alert_sent
      ? `<div class="alert-sent-badge">📧 Alert sent at ${rec.alert_time ? new Date(rec.alert_time).toLocaleTimeString() : "–"}</div>`
      : "";

    html += `
      <div class="jersey-card ${cardClass}" id="card-${num}">
        <div class="card-top">
          <span class="jersey-number">#${num}</span>
          ${statusHtml}
        </div>
        <div class="timer-row">
          <span class="timer-display" id="timer-${num}">${fmtTime(contTime)}</span>
          <div class="progress-bar-track">
            <div class="progress-bar-fill ${barClass}" id="bar-${num}"
                 style="width:${pct}%"></div>
          </div>
        </div>
        <div class="card-meta">
          <span>First seen: ${rec.first_seen || "–"}</span>
          <span>Last seen: ${rec.last_seen || "–"}</span>
        </div>
        ${alertBadge}
      </div>`;
  }
  container.innerHTML = html;
}

// Interpolate the timer so it ticks every second without waiting for API
function interpolatedTime(rec) {
  if (!rec.in_frame) return rec.continuous_time || 0;
  const elapsed = Date.now() / 1000 - (rec._apiAt || 0);
  return (rec._apiTime || 0) + elapsed;
}

function updateSummaryBadges(activeCount) {
  const ab = $("active-badge");
  if (ab) {
    ab.textContent = `${activeCount} In Room`;
    ab.className   = activeCount > 0 ? "badge active" : "badge";
  }
  const alertBadge = $("alert-count-badge");
  if (alertBadge) {
    const n = state.alerts.length;
    alertBadge.textContent = `${n} Alert${n !== 1 ? "s" : ""}`;
    alertBadge.className   = n > 0 ? "badge danger-badge" : "badge danger-badge zero";
  }
}

// ── Render alerts ─────────────────────────────────────────────────
function renderAlerts() {
  const container = $("alerts-container");
  if (!container) return;

  const visible = state.alerts.filter(a => !state.hiddenAlerts.has(a.id));

  if (visible.length === 0) {
    container.innerHTML =
      '<div class="empty-state" id="empty-alerts">No alerts triggered yet.</div>';
    return;
  }

  let html = "";
  for (const a of visible) {
    const ts = a.timestamp_display || new Date(a.timestamp).toLocaleString();
    html += `
      <div class="alert-item" id="alert-${a.id}">
        <div class="alert-number">#${a.jersey_number}</div>
        <div class="alert-body">
          <div class="alert-subject">Jersey #${a.jersey_number} – ${a.room_id}</div>
          <div class="alert-detail">Present for <strong>${a.duration_display}</strong> · ${a.recipient}</div>
          <span class="alert-status">✓ Sent</span>
        </div>
        <div class="alert-time">${ts}</div>
      </div>`;
  }
  container.innerHTML = html;
}

// ── Tick: update timer displays without re-rendering everything ────
function tickTimers() {
  for (const [num, rec] of Object.entries(state.jerseys)) {
    const timerEl = $(`timer-${num}`);
    const barEl   = $(`bar-${num}`);
    if (!timerEl) continue;

    const t   = interpolatedTime(rec);
    const pct = Math.min(100, (t / state.alertThreshold) * 100);

    timerEl.textContent = fmtTime(t);

    // ── timer text colour ──────────────────────────────────────────
    if (pct >= 100) {
      timerEl.className = "timer-display danger";
    } else if (pct >= 75) {
      timerEl.className = "timer-display warn";
    } else {
      timerEl.className = "timer-display";
    }

    // ── progress bar fill ──────────────────────────────────────────
    if (barEl) {
      barEl.style.width = `${pct}%`;
      if (pct >= 100) {
        barEl.className = "progress-bar-fill danger";
      } else if (pct >= 75) {
        barEl.className = "progress-bar-fill warn";
      } else {
        barEl.className = "progress-bar-fill";
      }
    }
  }
}

// ── Clear alert display (visual only) ────────────────────────────
window.clearAlertDisplay = function () {
  for (const a of state.alerts) { state.hiddenAlerts.add(a.id); }
  renderAlerts();
};

// ── Boot ──────────────────────────────────────────────────────────
async function init() {
  // Initial fetches
  await fetchStatus();
  await fetchAlerts();
  renderCards();

  // Poll API
  pollTimer  = setInterval(async () => { await fetchStatus(); renderCards(); }, 2000);
  alertTimer = setInterval(fetchAlerts, 3000);

  // Local timer tick every second (smooth updates)
  tickInterval = setInterval(tickTimers, 1000);
}

document.addEventListener("DOMContentLoaded", init);
