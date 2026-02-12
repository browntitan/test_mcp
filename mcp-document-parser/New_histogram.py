#!/usr/bin/env python3
"""
OpenWebUI message-level usage metrics (sessionized from chat JSON message timestamps)

Key definitions:
- Event = a USER message inside chat JSON (role == 'user') timestamped by its own `timestamp` field when present,
  otherwise inferred from the earliest child (assistant) message timestamp.
- Session = consecutive USER messages where each new message arrives within <= 10 minutes of the previous one.
- Active user-week = user has at least one USER message on each weekday Mon–Thu within that week.
- We include ONLY Monday–Thursday events. Friday/weekend are excluded.

Outputs:
- new_hist_outputs/
  - summary_messagelevel.csv
  - active_user_weeks_messagelevel.csv
  - histograms (all users + active users, per time range)

Dependencies:
  pip install psycopg2-binary pandas matplotlib tqdm numpy
"""

from __future__ import annotations

import ast
import json
import logging
import os
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================
# PASTE YOUR CONNECTION URL HERE (DO NOT COMMIT THIS FILE)
# Supported:
#   postgresql://HOST:5432/DB?user=USER&password=PASS&sslmode=require
#   postgresql://USER:PASS@HOST:5432/DB?sslmode=require
# NOTE: If password contains '&' or other special chars, URL-encode it.
# ============================================================
conn_str = "postgresql://YOUR_HOST:5432/YOUR_DB?user=YOUR_USER&password=YOUR_PASSWORD&sslmode=require"


# ----------------------------
# Config
# ----------------------------

SCHEMA = "public"
TABLE = "chat"
USER_COL = "user_id"
CREATED_COL = "created_at"   # bigint epoch
UPDATED_COL = "updated_at"   # bigint epoch
CHAT_JSON_COL = "chat"       # json/jsonb

# Only keep Monday-Thursday (Python weekday: 0=Mon..6=Sun)
INCLUDED_WEEKDAYS = {0, 1, 2, 3}  # Mon-Thu

# Session gap (manager requirement)
SESSION_GAP_MINUTES = 10

# Output directory
OUT_DIR = Path("new_hist_outputs")


# ----------------------------
# Logging
# ----------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ----------------------------
# Connection helpers
# ----------------------------

def parse_pg_url_to_kwargs(url: str) -> dict:
    if not url or url.strip() == "":
        raise ValueError("conn_str is empty. Paste your Postgres connection URL into conn_str.")

    u = urllib.parse.urlparse(url.strip())
    if u.scheme not in ("postgresql", "postgres"):
        raise ValueError(f"Unsupported URL scheme '{u.scheme}'. Use postgresql:// or postgres://")

    qs = urllib.parse.parse_qs(u.query)

    user = (qs.get("user") or [u.username] or [None])[0]
    password = (qs.get("password") or [u.password] or [None])[0]
    sslmode = (qs.get("sslmode") or ["require"])[0]

    dbname = (u.path or "").lstrip("/")
    host = u.hostname
    port = u.port or 5432

    missing = [k for k, v in {"host": host, "dbname": dbname, "user": user, "password": password}.items() if not v]
    if missing:
        raise ValueError(
            f"Connection URL missing required fields: {', '.join(missing)}.\n"
            "Expected: postgresql://HOST:5432/DB?user=USER&password=PASS&sslmode=require"
        )

    return {"dbname": dbname, "user": user, "password": password, "host": host, "port": port, "sslmode": sslmode}


def connect() -> psycopg2.extensions.connection:
    kwargs = parse_pg_url_to_kwargs(conn_str)
    logging.info(
        "Connecting to Postgres host=%s port=%s db=%s sslmode=%s user=%s",
        kwargs["host"], kwargs["port"], kwargs["dbname"], kwargs["sslmode"], kwargs["user"]
    )

    conn = psycopg2.connect(**kwargs)

    # Named cursors require a transaction.
    conn.autocommit = False

    # Ensure JSON fields decode
    psycopg2.extras.register_default_json(conn)
    psycopg2.extras.register_default_jsonb(conn)

    with conn.cursor() as cur:
        cur.execute("SET TIME ZONE 'UTC';")
        cur.execute("SET statement_timeout = '0';")
    conn.commit()
    return conn


# ----------------------------
# Time / epoch helpers
# ----------------------------

def ensure_utc(dt_obj: datetime) -> datetime:
    if dt_obj.tzinfo is None:
        return dt_obj.replace(tzinfo=timezone.utc)
    return dt_obj.astimezone(timezone.utc)


def monday_of(d: datetime) -> date:
    return (d.date() - timedelta(days=d.weekday()))


def detect_epoch_unit(conn, schema: str, table: str, col: str) -> str:
    """
    Detect whether bigint epoch is seconds or milliseconds using MAX(col).
    """
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX({col}) FROM {schema}.{table} WHERE {col} IS NOT NULL;")
        mx = cur.fetchone()[0]
    if mx is None:
        return "seconds"
    mx = int(mx)
    return "milliseconds" if mx >= 100_000_000_000 else "seconds"


def epoch_in_unit(dt: datetime, unit: str) -> int:
    ts = dt.timestamp()
    return int(ts * 1000) if unit == "milliseconds" else int(ts)


def from_epoch(val: float, unit: str) -> datetime:
    if unit == "milliseconds":
        return datetime.fromtimestamp(val / 1000.0, tz=timezone.utc)
    return datetime.fromtimestamp(val, tz=timezone.utc)


def guess_message_ts_unit(raw_ts: float) -> str:
    # Message-level heuristic: >1e11 looks like ms
    return "milliseconds" if raw_ts >= 100_000_000_000 else "seconds"


# ----------------------------
# JSON parsing: extract USER message timestamps
# ----------------------------

def extract_user_message_timestamps(chat_obj: Any) -> Tuple[List[datetime], Dict[str, int]]:
    """Extract USER message submission timestamps from the chat JSON.

    Your observed schema:
      chat.history.messages is a dict keyed by message UUID -> message object.

    Important:
    - USER messages may not always include a `timestamp` field.
    - When missing, we infer a submission timestamp from the earliest child message timestamp
      (childrenIds usually point to assistant responses that do carry a timestamp).

    Returns:
      (timestamps_utc, stats)
      where stats includes counts for direct timestamps vs inferred vs skipped.
    """

    stats = {
        "user_msgs_seen": 0,
        "user_msgs_direct_ts": 0,
        "user_msgs_inferred_ts": 0,
        "user_msgs_skipped_no_ts": 0,
    }

    def to_dt(ts_raw: Any) -> Optional[datetime]:
        if ts_raw is None:
            return None
        try:
            ts_val = float(ts_raw)
        except Exception:
            return None
        unit = guess_message_ts_unit(ts_val)
        return from_epoch(ts_val, unit)

    # Normalize input into a Python object.
    if chat_obj is None:
        return [], stats

    if isinstance(chat_obj, str):
        # Try strict JSON first
        try:
            chat_obj = json.loads(chat_obj)
        except Exception:
            # Fallback for JSON-ish strings (single quotes, None, True/False)
            try:
                chat_obj = ast.literal_eval(chat_obj)
            except Exception:
                return [], stats

    # Some drivers may return memoryview for json; handle defensively.
    if isinstance(chat_obj, (bytes, bytearray, memoryview)):
        try:
            chat_obj = json.loads(bytes(chat_obj).decode("utf-8"))
        except Exception:
            try:
                chat_obj = ast.literal_eval(bytes(chat_obj).decode("utf-8"))
            except Exception:
                return [], stats

    # Pull messages map.
    messages_map: Dict[str, Any] = {}

    if isinstance(chat_obj, dict):
        # Primary (your schema)
        history = chat_obj.get("history")
        if isinstance(history, dict):
            msgs = history.get("messages")
            if isinstance(msgs, dict):
                messages_map = msgs

        # Secondary shapes
        if not messages_map:
            msgs = chat_obj.get("messages")
            if isinstance(msgs, dict):
                messages_map = msgs

        # As a last resort, if the dict itself looks like message-id->message
        if not messages_map:
            # Heuristic: any value is dict with role/content
            if any(isinstance(v, dict) and ("role" in v or "content" in v) for v in chat_obj.values()):
                messages_map = chat_obj

    elif isinstance(chat_obj, list):
        # List of message dicts
        for idx, msg in enumerate(chat_obj):
            messages_map[str(idx)] = msg

    if not messages_map:
        return [], stats

    # Index messages by id for child lookups.
    # Some entries may have inconsistent keys; prefer the dict key, then message['id'].
    normalized: Dict[str, Dict[str, Any]] = {}
    for k, v in messages_map.items():
        if not isinstance(v, dict):
            continue
        mid = v.get("id") or k
        normalized[str(mid)] = v

    timestamps: List[datetime] = []

    for mid, msg in normalized.items():
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue

        stats["user_msgs_seen"] += 1

        # 1) Direct timestamp on user message
        dt = to_dt(msg.get("timestamp") or msg.get("createdAt") or msg.get("created_at") or msg.get("time"))
        if dt is not None:
            timestamps.append(dt)
            stats["user_msgs_direct_ts"] += 1
            continue

        # 2) Infer from earliest child timestamp
        children = msg.get("childrenIds") or msg.get("children_ids")
        if isinstance(children, list) and children:
            child_dts: List[datetime] = []
            for cid in children:
                child = normalized.get(str(cid))
                if not isinstance(child, dict):
                    continue
                cdt = to_dt(child.get("timestamp") or child.get("createdAt") or child.get("created_at") or child.get("time"))
                if cdt is not None:
                    child_dts.append(cdt)
            if child_dts:
                inferred = min(child_dts)
                timestamps.append(inferred)
                stats["user_msgs_inferred_ts"] += 1
                continue

        # 3) If we can't timestamp it, skip
        stats["user_msgs_skipped_no_ts"] += 1

    return timestamps, stats


# ----------------------------
# DB streaming: chat rows overlapping the time window
# ----------------------------

def count_chat_rows(conn, start_epoch: int, end_epoch: int) -> int:
    q = f"""
    SELECT COUNT(*)
    FROM {SCHEMA}.{TABLE}
    WHERE {USER_COL} IS NOT NULL
      AND {CHAT_JSON_COL} IS NOT NULL
      AND {UPDATED_COL} >= %s
      AND {CREATED_COL} < %s;
    """
    with conn.cursor() as cur:
        cur.execute(q, (start_epoch, end_epoch))
        return int(cur.fetchone()[0])


def stream_chat_rows(conn, start_epoch: int, end_epoch: int, itersize: int = 2000):
    """
    Stream chat rows that overlap [start, end):
    updated_at >= start AND created_at < end
    """
    q = f"""
    SELECT {USER_COL} AS user_id,
           {CREATED_COL} AS created_at,
           {UPDATED_COL} AS updated_at,
           {CHAT_JSON_COL} AS chat_json
    FROM {SCHEMA}.{TABLE}
    WHERE {USER_COL} IS NOT NULL
      AND {CHAT_JSON_COL} IS NOT NULL
      AND {UPDATED_COL} >= %s
      AND {CREATED_COL} < %s
    ORDER BY {USER_COL}, {CREATED_COL};
    """
    cur = conn.cursor(name=f"chat_rows_{int(datetime.now().timestamp())}")
    cur.itersize = itersize
    cur.execute(q, (start_epoch, end_epoch))
    try:
        for row in cur:
            yield row
    finally:
        cur.close()


# ----------------------------
# Analytics for one range
# ----------------------------

@dataclass
class RangeMetrics:
    range_label: str
    start_utc: datetime
    end_utc: datetime

    unique_users_total: int
    active_users_total: int

    total_sessions_total: int
    active_sessions_total: int

    mean_session_minutes_active: Optional[float]
    median_session_minutes_active: Optional[float]

    mean_sessions_per_active_week: Optional[float]
    median_sessions_per_active_week: Optional[float]


def analyze_time_range(
    conn,
    range_label: str,
    start_dt: datetime,
    end_dt: datetime,
    gap_minutes: int = SESSION_GAP_MINUTES,
    itersize: int = 2000,
) -> Tuple[RangeMetrics, pd.DataFrame, pd.DataFrame, Dict[str, List[float]]]:
    """
    Returns:
      - summary metrics
      - df_active_user_weeks (one row per active user-week)
      - df_all_user_weeks (one row per user-week with sessions/messages)
      - distributions dict used for histograms
    """
    start_dt = ensure_utc(start_dt)
    end_dt = ensure_utc(end_dt)

    # Detect epoch unit for created_at/updated_at
    epoch_unit = detect_epoch_unit(conn, SCHEMA, TABLE, CREATED_COL)
    logging.info("[%s] Detected chat epoch unit: %s", range_label, epoch_unit)

    start_epoch = epoch_in_unit(start_dt, epoch_unit)
    end_epoch = epoch_in_unit(end_dt, epoch_unit)

    total_chat_rows = count_chat_rows(conn, start_epoch, end_epoch)
    logging.info("[%s] Candidate chat rows (overlapping window): %d", range_label, total_chat_rows)

    gap = timedelta(minutes=gap_minutes)

    # Global sets/counters
    unique_users: Set[str] = set()

    extraction_stats = {
        "chat_rows_scanned": 0,
        "chat_rows_with_any_user_msg": 0,
        "user_msgs_seen": 0,
        "user_msgs_direct_ts": 0,
        "user_msgs_inferred_ts": 0,
        "user_msgs_skipped_no_ts": 0,
        "user_events_used_after_filters": 0,
    }

    # Session list (for filtering to active weeks later)
    # Each session record: (user_id, week_monday, duration_minutes)
    all_sessions: List[Tuple[str, date, float]] = []

    # Per user-week counts
    user_week_session_counts: Dict[Tuple[str, date], int] = {}
    user_week_days_used: Dict[Tuple[str, date], Set[int]] = {}

    # Sessionization state per current user
    current_user: Optional[str] = None
    last_msg_ts: Optional[datetime] = None
    session_start: Optional[datetime] = None
    session_end: Optional[datetime] = None

    def close_session():
        nonlocal session_start, session_end, last_msg_ts, current_user, all_sessions, user_week_session_counts
        if current_user is None or session_start is None or session_end is None:
            return

        dur_min = max(0.0, (session_end - session_start).total_seconds() / 60.0)
        wk = monday_of(session_start)

        all_sessions.append((current_user, wk, dur_min))
        user_week_session_counts[(current_user, wk)] = user_week_session_counts.get((current_user, wk), 0) + 1

        session_start = None
        session_end = None
        last_msg_ts = None

    pbar = tqdm(total=total_chat_rows, desc=f"[{range_label}] streaming chat rows", unit="rows")

    for user_id, created_at, updated_at, chat_json in stream_chat_rows(conn, start_epoch, end_epoch, itersize=itersize):
        pbar.update(1)
        extraction_stats["chat_rows_scanned"] += 1
        user_id = str(user_id)
        unique_users.add(user_id)

        # When user changes, close any open session for previous user
        if current_user is None:
            current_user = user_id
        elif user_id != current_user:
            close_session()
            current_user = user_id

        # Extract user-message timestamps (direct or inferred) from this chat JSON
        msg_ts, st = extract_user_message_timestamps(chat_json)
        extraction_stats["user_msgs_seen"] += st["user_msgs_seen"]
        extraction_stats["user_msgs_direct_ts"] += st["user_msgs_direct_ts"]
        extraction_stats["user_msgs_inferred_ts"] += st["user_msgs_inferred_ts"]
        extraction_stats["user_msgs_skipped_no_ts"] += st["user_msgs_skipped_no_ts"]

        if msg_ts:
            extraction_stats["chat_rows_with_any_user_msg"] += 1
        else:
            continue

        # Filter to window + Mon-Thu
        filtered: List[datetime] = []
        for ts in msg_ts:
            if ts < start_dt or ts >= end_dt:
                continue
            if ts.weekday() not in INCLUDED_WEEKDAYS:
                continue
            filtered.append(ts)

        if not filtered:
            continue

        filtered.sort()
        extraction_stats["user_events_used_after_filters"] += len(filtered)

        # Record days-used per user-week (based on message days, not sessions)
        for ts in filtered:
            wk = monday_of(ts)
            key = (user_id, wk)
            if key not in user_week_days_used:
                user_week_days_used[key] = set()
            user_week_days_used[key].add(ts.weekday())

        # Sessionize message timestamps for this user
        for ts in filtered:
            if session_start is None:
                session_start = ts
                session_end = ts
                last_msg_ts = ts
                continue

            # Same session if gap <= threshold
            if last_msg_ts is not None and (ts - last_msg_ts) <= gap:
                session_end = ts
                last_msg_ts = ts
            else:
                # close previous, start new
                close_session()
                session_start = ts
                session_end = ts
                last_msg_ts = ts

    # Close tail
    close_session()
    pbar.close()

    # Log extraction coverage so we can validate that message-level timestamps are working.
    logging.info(
        "[%s] Extraction coverage: chat_rows_scanned=%d, chat_rows_with_user_msgs=%d, user_msgs_seen=%d, direct_ts=%d, inferred_ts=%d, skipped_no_ts=%d, user_events_used_after_filters=%d",
        range_label,
        extraction_stats["chat_rows_scanned"],
        extraction_stats["chat_rows_with_any_user_msg"],
        extraction_stats["user_msgs_seen"],
        extraction_stats["user_msgs_direct_ts"],
        extraction_stats["user_msgs_inferred_ts"],
        extraction_stats["user_msgs_skipped_no_ts"],
        extraction_stats["user_events_used_after_filters"],
    )

    # Determine active user-weeks: at least one message on each day Mon-Thu
    required_days = set(INCLUDED_WEEKDAYS)  # {0,1,2,3}
    active_user_weeks = {k for k, days in user_week_days_used.items() if required_days.issubset(days)}
    active_users = {u for (u, wk) in active_user_weeks}

    # Split sessions into active vs all
    total_sessions_total = len(all_sessions)
    active_sessions = [(u, wk, d) for (u, wk, d) in all_sessions if (u, wk) in active_user_weeks]

    # Distributions
    all_session_durations = [d for (_, _, d) in all_sessions]
    active_session_durations = [d for (_, _, d) in active_sessions]

    # Sessions per week distributions
    all_sessions_per_week = list(user_week_session_counts.values())

    active_sessions_per_week = [
        user_week_session_counts.get((u, wk), 0)
        for (u, wk) in active_user_weeks
    ]

    # Summary stats for ACTIVE cohort (manager ask)
    def safe_mean(x: List[float]) -> Optional[float]:
        return float(np.mean(x)) if x else None

    def safe_median(x: List[float]) -> Optional[float]:
        return float(np.median(x)) if x else None

    metrics = RangeMetrics(
        range_label=range_label,
        start_utc=start_dt,
        end_utc=end_dt,
        unique_users_total=len(unique_users),
        active_users_total=len(active_users),
        total_sessions_total=total_sessions_total,
        active_sessions_total=len(active_sessions),
        mean_session_minutes_active=safe_mean(active_session_durations),
        median_session_minutes_active=safe_median(active_session_durations),
        mean_sessions_per_active_week=safe_mean([float(x) for x in active_sessions_per_week]) if active_sessions_per_week else None,
        median_sessions_per_active_week=safe_median([float(x) for x in active_sessions_per_week]) if active_sessions_per_week else None,
    )

    # Output frames (optional detail)
    df_all_user_weeks = pd.DataFrame([
        {
            "user_id": u,
            "week_monday": wk.isoformat(),
            "sessions_in_week": user_week_session_counts.get((u, wk), 0),
            "days_used": ",".join(str(d) for d in sorted(user_week_days_used.get((u, wk), set()))),
            "is_active_week": (u, wk) in active_user_weeks,
        }
        for (u, wk) in sorted(user_week_days_used.keys(), key=lambda x: (x[0], x[1]))
    ])

    df_active_user_weeks = df_all_user_weeks[df_all_user_weeks["is_active_week"] == True].copy()

    distributions = {
        "all_session_durations": all_session_durations,
        "active_session_durations": active_session_durations,
        "all_sessions_per_week": [float(x) for x in all_sessions_per_week],
        "active_sessions_per_week": [float(x) for x in active_sessions_per_week],
    }

    return metrics, df_active_user_weeks, df_all_user_weeks, distributions


# ----------------------------
# October detection
# ----------------------------

def find_october_range_with_data(conn, year_hint: Optional[int] = None) -> Optional[Tuple[str, datetime, datetime]]:
    """
    Returns (label, start_dt, end_dt) for an October that has overlapping chat rows.
    Checks year_hint, current year, previous year.
    """
    now = datetime.now(timezone.utc)
    candidates = []
    if year_hint:
        candidates.append(year_hint)
    candidates.extend([now.year, now.year - 1])

    epoch_unit = detect_epoch_unit(conn, SCHEMA, TABLE, CREATED_COL)

    for y in candidates:
        start_dt = datetime(y, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
        end_dt = datetime(y, 11, 1, 0, 0, 0, tzinfo=timezone.utc)
        start_epoch = epoch_in_unit(start_dt, epoch_unit)
        end_epoch = epoch_in_unit(end_dt, epoch_unit)

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT EXISTS(
                    SELECT 1
                    FROM {SCHEMA}.{TABLE}
                    WHERE {USER_COL} IS NOT NULL
                      AND {CHAT_JSON_COL} IS NOT NULL
                      AND {UPDATED_COL} >= %s
                      AND {CREATED_COL} < %s
                    LIMIT 1
                );
                """,
                (start_epoch, end_epoch),
            )
            if bool(cur.fetchone()[0]):
                return f"October_{y}", start_dt, end_dt

    return None


# ----------------------------
# Plotting
# ----------------------------

def save_histograms(range_label: str, dists: Dict[str, List[float]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Session duration histogram (cap at 200 for readability)
    cap_minutes = 200
    bins_dur = np.arange(0, cap_minutes + 5, 5)

    def plot_hist(values: List[float], title: str, out_name: str):
        plt.figure()
        if values:
            v = np.clip(np.array(values, dtype=float), 0, cap_minutes)
            plt.hist(v, bins=bins_dur)
        else:
            plt.text(0.5, 0.5, "No data", ha="center", va="center", transform=plt.gca().transAxes)
        plt.title(title)
        plt.xlabel(f"Session duration (minutes, capped at {cap_minutes})")
        plt.ylabel("Count")
        plt.xlim(0, cap_minutes)
        out_path = OUT_DIR / out_name
        plt.savefig(str(out_path), dpi=160, bbox_inches="tight")
        plt.close()
        logging.info("Wrote %s", out_path)

    plot_hist(
        dists["all_session_durations"],
        f"Session Duration (All Users) — {range_label}",
        f"{range_label.lower()}_session_duration_all_users.png",
    )
    plot_hist(
        dists["active_session_durations"],
        f"Session Duration (Active Users) — {range_label}",
        f"{range_label.lower()}_session_duration_active_users.png",
    )

    # Sessions per week histogram
    cap_x = 100
    bins_spw = np.arange(0, cap_x + 2, 1)

    def plot_hist_int(values: List[float], title: str, out_name: str):
        plt.figure()
        if values:
            v = np.clip(np.array(values, dtype=float), 0, cap_x)
            plt.hist(v, bins=bins_spw, align="left")
        else:
            plt.text(0.5, 0.5, "No data", ha="center", va="center", transform=plt.gca().transAxes)
        plt.title(title)
        plt.xlabel(f"Sessions per week (capped at {cap_x})")
        plt.ylabel("Count")
        plt.xlim(0, cap_x)
        out_path = OUT_DIR / out_name
        plt.savefig(str(out_path), dpi=160, bbox_inches="tight")
        plt.close()
        logging.info("Wrote %s", out_path)

    plot_hist_int(
        dists["all_sessions_per_week"],
        f"Sessions per Week (All Users) — {range_label}",
        f"{range_label.lower()}_sessions_per_week_all_users.png",
    )
    plot_hist_int(
        dists["active_sessions_per_week"],
        f"Sessions per Week (Active Users) — {range_label}",
        f"{range_label.lower()}_sessions_per_week_active_users.png",
    )


# ----------------------------
# Main
# ----------------------------

def main():
    setup_logging("INFO")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Output directory: %s", OUT_DIR.resolve())

    conn = connect()
    try:
        now = datetime.now(timezone.utc)
        last30_start = now - timedelta(days=30)

        ranges: List[Tuple[str, datetime, datetime]] = [
            ("Last_30_Days", last30_start, now),
        ]

        oct_found = find_october_range_with_data(conn, year_hint=None)
        if oct_found:
            ranges.append(oct_found)
        else:
            logging.warning("No October data found (checked current/previous year). Skipping October.")

        summary_rows = []
        all_active_weeks_frames = []
        all_user_weeks_frames = []

        for label, start_dt, end_dt in ranges:
            logging.info("==== Analyzing %s (%s -> %s) ====", label, start_dt.isoformat(), end_dt.isoformat())

            metrics, df_active_weeks, df_all_weeks, dists = analyze_time_range(
                conn,
                range_label=label,
                start_dt=start_dt,
                end_dt=end_dt,
                gap_minutes=SESSION_GAP_MINUTES,
                itersize=2000,
            )

            # Print manager-requested headline metrics (ACTIVE users)
            logging.info("[%s] Unique users (Mon–Thu): %d", label, metrics.unique_users_total)
            logging.info("[%s] Active users (Mon–Thu each day in a week): %d", label, metrics.active_users_total)
            logging.info("[%s] Mean session minutes (ACTIVE): %s", label, None if metrics.mean_session_minutes_active is None else round(metrics.mean_session_minutes_active, 3))
            logging.info("[%s] Median session minutes (ACTIVE): %s", label, None if metrics.median_session_minutes_active is None else round(metrics.median_session_minutes_active, 3))
            logging.info("[%s] Mean sessions/week (ACTIVE): %s", label, None if metrics.mean_sessions_per_active_week is None else round(metrics.mean_sessions_per_active_week, 3))
            logging.info("[%s] Median sessions/week (ACTIVE): %s", label, None if metrics.median_sessions_per_active_week is None else round(metrics.median_sessions_per_active_week, 3))

            summary_rows.append({
                "range_label": metrics.range_label,
                "range_start_utc": metrics.start_utc.isoformat(),
                "range_end_utc": metrics.end_utc.isoformat(),
                "session_gap_minutes": SESSION_GAP_MINUTES,
                "unique_users_total": metrics.unique_users_total,
                "active_users_total": metrics.active_users_total,
                "total_sessions_total_all_users": metrics.total_sessions_total,
                "active_sessions_total": metrics.active_sessions_total,
                "mean_session_minutes_active": metrics.mean_session_minutes_active,
                "median_session_minutes_active": metrics.median_session_minutes_active,
                "mean_sessions_per_active_week": metrics.mean_sessions_per_active_week,
                "median_sessions_per_active_week": metrics.median_sessions_per_active_week,
            })

            df_active_weeks["range_label"] = label
            df_all_weeks["range_label"] = label
            all_active_weeks_frames.append(df_active_weeks)
            all_user_weeks_frames.append(df_all_weeks)

            save_histograms(label, dists)

            conn.commit()

        # Write summary CSVs
        summary_df = pd.DataFrame(summary_rows)
        summary_path = OUT_DIR / "summary_messagelevel.csv"
        summary_df.to_csv(summary_path, index=False)
        logging.info("Wrote %s", summary_path)

        if all_active_weeks_frames:
            active_weeks_df = pd.concat(all_active_weeks_frames, ignore_index=True)
            active_weeks_path = OUT_DIR / "active_user_weeks_messagelevel.csv"
            active_weeks_df.to_csv(active_weeks_path, index=False)
            logging.info("Wrote %s", active_weeks_path)

        if all_user_weeks_frames:
            all_weeks_df = pd.concat(all_user_weeks_frames, ignore_index=True)
            all_weeks_path = OUT_DIR / "all_user_weeks_messagelevel.csv"
            all_weeks_df.to_csv(all_weeks_path, index=False)
            logging.info("Wrote %s", all_weeks_path)

        logging.info("Done.")

    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()


if __name__ == "__main__":
    main()