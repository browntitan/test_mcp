from __future__ import annotations

import argparse
import logging
import sys
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psycopg2
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
# Logging
# ----------------------------

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ----------------------------
# Connection parsing
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

    # IMPORTANT: psycopg2 named (server-side) cursors require an open transaction.
    # autocommit=True disables transactions and will raise:
    #   "can't use a named cursor outside of transactions"
    conn.autocommit = False

    with conn.cursor() as cur:
        # Make weekday extraction stable
        cur.execute("SET TIME ZONE 'UTC';")
        cur.execute("SET statement_timeout = '0';")

    # Commit session settings so we start analyses with a clean transaction state.
    conn.commit()
    return conn


# ----------------------------
# Schema introspection
# ----------------------------

@dataclass
class ChatSchema:
    schema: str
    table: str
    user_col: str
    ts_col: str
    ts_data_type: str  # info_schema data_type (e.g., bigint, timestamp with time zone, etc.)


def get_columns(conn, schema: str, table: str) -> Dict[str, str]:
    q = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = %s AND table_name = %s
    ORDER BY ordinal_position;
    """
    with conn.cursor() as cur:
        cur.execute(q, (schema, table))
        rows = cur.fetchall()
    return {r[0]: r[1] for r in rows}


def detect_schema(conn, schema: str, table: str,
                  user_col_override: Optional[str],
                  ts_col_override: Optional[str],
                  prefer_ts: str = "created_at") -> ChatSchema:
    cols = get_columns(conn, schema, table)
    if not cols:
        raise ValueError(f"Could not find table {schema}.{table} (no columns returned).")

    logging.info("Found %d columns on %s.%s", len(cols), schema, table)

    if user_col_override and user_col_override not in cols:
        raise ValueError(f"--user-col {user_col_override} not found on {schema}.{table}")
    if ts_col_override and ts_col_override not in cols:
        raise ValueError(f"--ts-col {ts_col_override} not found on {schema}.{table}")

    # OpenWebUI typically: user_id, created_at, updated_at (created_at/updated_at may be BIGINT)
    user_candidates = ["user_id", "userid", "userId", "user", "owner_id", "created_by"]
    ts_name_candidates = [prefer_ts, "created_at", "updated_at", "createdAt", "updatedAt", "timestamp", "ts"]

    def pick_user_col() -> str:
        if user_col_override:
            return user_col_override
        lower_map = {c.lower(): c for c in cols.keys()}
        for cand in user_candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        for c in cols.keys():
            if "user" in c.lower():
                return c
        raise ValueError("Could not auto-detect user id column. Use --user-col to specify it.")

    def pick_ts_col() -> str:
        if ts_col_override:
            return ts_col_override

        lower_map = {c.lower(): c for c in cols.keys()}

        # 1) Prefer known OpenWebUI timestamp column names, regardless of type
        for cand in ts_name_candidates:
            if cand and cand.lower() in lower_map:
                return lower_map[cand.lower()]

        # 2) If none matched, fall back to first timestamp-like type
        for c, t in cols.items():
            if "timestamp" in t.lower() or "date" in t.lower():
                return c

        # 3) Final fallback: any integer/bigint that looks like epoch by name
        for c, t in cols.items():
            if t.lower() in ("bigint", "integer") and ("created" in c.lower() or "updated" in c.lower() or "time" in c.lower()):
                return c

        raise ValueError("Could not auto-detect timestamp column. Use --ts-col to specify it.")

    user_col = pick_user_col()
    ts_col = pick_ts_col()
    ts_data_type = cols[ts_col]
    logging.info("Using user column: %s | timestamp column: %s (%s)", user_col, ts_col, ts_data_type)
    return ChatSchema(schema=schema, table=table, user_col=user_col, ts_col=ts_col, ts_data_type=ts_data_type)


# ----------------------------
# Epoch unit detection and SQL expression helpers
# ----------------------------

@dataclass
class TimeColumnPlan:
    # SQL expression that yields a timestamptz for weekday/session logic
    ts_expr: str
    # Range filter uses either timestamps (native) or numeric epoch (fast)
    range_where: str
    range_params_builder: callable  # (start_dt, end_dt) -> tuple(params)


def detect_epoch_unit_seconds_or_millis(conn, chat: ChatSchema) -> str:
    """
    Returns 'seconds' or 'milliseconds' for BIGINT/INTEGER timestamp columns.
    Uses MAX(value) heuristic:
      - epoch seconds ~ 1.7e9
      - epoch millis  ~ 1.7e12
    """
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT MAX({chat.ts_col}) FROM {chat.schema}.{chat.table} WHERE {chat.ts_col} IS NOT NULL;"
        )
        mx = cur.fetchone()[0]

    if mx is None:
        logging.warning("No non-null values found in %s; assuming epoch seconds.", chat.ts_col)
        return "seconds"

    try:
        mx_int = int(mx)
    except Exception:
        logging.warning("Could not parse MAX(%s)=%r as int; assuming epoch seconds.", chat.ts_col, mx)
        return "seconds"

    unit = "milliseconds" if mx_int >= 100_000_000_000 else "seconds"  # 1e11 threshold
    logging.info("Detected epoch unit for %s: %s (MAX=%d)", chat.ts_col, unit, mx_int)
    return unit


def build_time_plan(conn, chat: ChatSchema, ts_unit: str) -> TimeColumnPlan:
    t = chat.ts_data_type.lower()

    if "timestamp" in t or "date" in t:
        ts_expr = f"{chat.ts_col}"
        range_where = f"{chat.ts_col} >= %s AND {chat.ts_col} < %s"
        def params_builder(start_dt: datetime, end_dt: datetime):
            return (start_dt, end_dt)
        return TimeColumnPlan(ts_expr=ts_expr, range_where=range_where, range_params_builder=params_builder)

    # BIGINT/INTEGER epoch
    if ts_unit == "auto":
        ts_unit = detect_epoch_unit_seconds_or_millis(conn, chat)

    if ts_unit == "seconds":
        ts_expr = f"to_timestamp({chat.ts_col})"
        def params_builder(start_dt: datetime, end_dt: datetime):
            return (int(start_dt.timestamp()), int(end_dt.timestamp()))
        range_where = f"{chat.ts_col} >= %s AND {chat.ts_col} < %s"
        return TimeColumnPlan(ts_expr=ts_expr, range_where=range_where, range_params_builder=params_builder)

    if ts_unit == "milliseconds":
        ts_expr = f"to_timestamp({chat.ts_col} / 1000.0)"
        def params_builder(start_dt: datetime, end_dt: datetime):
            return (int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000))
        range_where = f"{chat.ts_col} >= %s AND {chat.ts_col} < %s"
        return TimeColumnPlan(ts_expr=ts_expr, range_where=range_where, range_params_builder=params_builder)

    raise ValueError(f"Unsupported --ts-unit {ts_unit}. Use auto|seconds|milliseconds.")


# ----------------------------
# Date helpers
# ----------------------------

def monday_of(d: datetime) -> date:
    return (d.date() - timedelta(days=d.weekday()))


def ensure_utc(dt_obj: datetime) -> datetime:
    if dt_obj.tzinfo is None:
        return dt_obj.replace(tzinfo=timezone.utc)
    return dt_obj.astimezone(timezone.utc)


# ----------------------------
# Core analytics
# ----------------------------

@dataclass
class WeekAgg:
    sessions: int = 0
    total_minutes: float = 0.0
    days_used: Set[int] = None  # 0=Mon..6=Sun
    durations: List[float] = None

    def __post_init__(self):
        if self.days_used is None:
            self.days_used = set()
        if self.durations is None:
            self.durations = []


@dataclass
class RangeResults:
    label: str
    start: datetime
    end: datetime
    active_users_total: int
    active_user_weeks_total: int
    session_durations_active: List[float]
    sessions_per_active_week: List[int]
    active_user_week_rows: List[dict]


def count_rows(conn, chat: ChatSchema, plan: TimeColumnPlan, start: datetime, end: datetime, dow_list_pg: List[int]) -> int:
    q = f"""
    SELECT COUNT(*)
    FROM {chat.schema}.{chat.table}
    WHERE {plan.range_where}
      AND {chat.user_col} IS NOT NULL
      AND EXTRACT(DOW FROM {plan.ts_expr}) = ANY(%s::int[]);
    """
    params = plan.range_params_builder(start, end) + (dow_list_pg,)
    with conn.cursor() as cur:
        cur.execute(q, params)
        return int(cur.fetchone()[0])


def stream_events(conn, chat: ChatSchema, plan: TimeColumnPlan, start: datetime, end: datetime,
                  dow_list_pg: List[int], itersize: int = 5000):
    """
    Streams (user_id, ts) ordered by (user_id, ts_raw/ts_col).
    If ts_col is BIGINT, ordering by ts_col is correct chronologically.
    """
    q = f"""
    SELECT {chat.user_col} AS user_id,
           {plan.ts_expr} AS ts
    FROM {chat.schema}.{chat.table}
    WHERE {plan.range_where}
      AND {chat.user_col} IS NOT NULL
      AND EXTRACT(DOW FROM {plan.ts_expr}) = ANY(%s::int[])
    ORDER BY {chat.user_col}, {chat.ts_col};
    """
    params = plan.range_params_builder(start, end) + (dow_list_pg,)

    cur = conn.cursor(name=f"evt_cursor_{int(datetime.now().timestamp())}")
    cur.itersize = itersize
    cur.execute(q, params)
    try:
        for user_id, ts in cur:
            yield user_id, ensure_utc(ts)
    finally:
        cur.close()


def analyze_range(conn, chat: ChatSchema, plan: TimeColumnPlan, label: str,
                  start: datetime, end: datetime,
                  gap_minutes: int,
                  required_weekdays: Set[int],
                  itersize: int = 5000) -> RangeResults:
    start = ensure_utc(start)
    end = ensure_utc(end)

    # Postgres DOW: 0=Sun, 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat
    dow_list_pg = [1, 2, 3, 4]  # Mon-Thu only
    total_rows = count_rows(conn, chat, plan, start, end, dow_list_pg)
    logging.info("[%s] Rows in range (Mon-Thu only): %d", label, total_rows)

    gap = timedelta(minutes=gap_minutes)

    active_users: Set[str] = set()
    active_user_week_rows: List[dict] = []
    durations_active: List[float] = []
    sessions_per_active_week: List[int] = []

    current_user = None
    session_start: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    per_week: Dict[date, WeekAgg] = {}

    def close_session():
        nonlocal session_start, last_ts, per_week
        if session_start is None or last_ts is None:
            return
        dur_min = max(0.0, (last_ts - session_start).total_seconds() / 60.0)
        wk = monday_of(session_start)
        wd = session_start.weekday()  # 0=Mon..6=Sun
        agg = per_week.get(wk)
        if agg is None:
            agg = WeekAgg()
            per_week[wk] = agg
        agg.sessions += 1
        agg.total_minutes += dur_min
        agg.days_used.add(wd)
        agg.durations.append(dur_min)
        session_start = None
        last_ts = None

    def finalize_user(user_id):
        nonlocal per_week, active_users, active_user_week_rows, durations_active, sessions_per_active_week
        if not per_week:
            return
        for wk_start, agg in per_week.items():
            is_active_week = required_weekdays.issubset(agg.days_used)
            if is_active_week:
                active_users.add(str(user_id))
                durations_active.extend(agg.durations)
                sessions_per_active_week.append(agg.sessions)
                active_user_week_rows.append({
                    "user_id": str(user_id),
                    "week_monday": wk_start.isoformat(),
                    "sessions": agg.sessions,
                    "total_session_minutes": round(agg.total_minutes, 3),
                    "avg_session_minutes": round((agg.total_minutes / agg.sessions) if agg.sessions else 0.0, 3),
                    "days_used": ",".join(str(d) for d in sorted(agg.days_used)),
                    "active_week": True,
                })
        per_week = {}

    pbar = tqdm(total=total_rows, desc=f"[{label}] streaming events", unit="rows")

    for user_id, ts in stream_events(conn, chat, plan, start, end, dow_list_pg, itersize=itersize):
        pbar.update(1)

        if current_user is None:
            current_user = user_id
            session_start = ts
            last_ts = ts
            continue

        if user_id != current_user:
            close_session()
            finalize_user(current_user)
            current_user = user_id
            session_start = ts
            last_ts = ts
            continue

        if last_ts is not None and (ts - last_ts) <= gap:
            last_ts = ts
        else:
            close_session()
            session_start = ts
            last_ts = ts

    close_session()
    if current_user is not None:
        finalize_user(current_user)

    pbar.close()

    logging.info("[%s] Active users=%d | Active user-weeks=%d | Active sessions=%d",
                 label, len(active_users), len(active_user_week_rows), len(durations_active))

    return RangeResults(
        label=label,
        start=start,
        end=end,
        active_users_total=len(active_users),
        active_user_weeks_total=len(active_user_week_rows),
        session_durations_active=durations_active,
        sessions_per_active_week=sessions_per_active_week,
        active_user_week_rows=active_user_week_rows,
    )


# ----------------------------
# Outputs
# ----------------------------

def summarize(label: str, durations: List[float], sessions_per_week: List[int], active_users_total: int) -> dict:
    if not durations:
        return {
            "label": label,
            "active_users_total": active_users_total,
            "active_sessions_total": 0,
            "mean_session_minutes": None,
            "median_session_minutes": None,
            "mean_sessions_per_active_week": None,
            "median_sessions_per_active_week": None,
        }

    d = np.array(durations, dtype=float)
    spw = np.array(sessions_per_week, dtype=int) if sessions_per_week else np.array([], dtype=int)

    return {
        "label": label,
        "active_users_total": int(active_users_total),
        "active_sessions_total": int(d.size),
        "mean_session_minutes": float(np.mean(d)),
        "median_session_minutes": float(np.median(d)),
        "mean_sessions_per_active_week": float(np.mean(spw)) if spw.size else None,
        "median_sessions_per_active_week": float(np.median(spw)) if spw.size else None,
    }


def save_histograms(label: str, durations: List[float], sessions_per_week: List[int]) -> None:
    if durations:
        d = np.array(durations, dtype=float)
        bins = list(np.arange(0, 181, 5)) + [240, 360, 480, 720, 1440]
        plt.figure()
        plt.hist(np.clip(d, 0, 1440), bins=bins)
        plt.title(f"Session Duration (minutes) — {label} (Active User-Weeks)")
        plt.xlabel("Session duration (minutes)")
        plt.ylabel("Count")
        out = f"{label.replace(' ', '_').lower()}_session_duration_hist.png"
        plt.savefig(out, dpi=160, bbox_inches="tight")
        plt.close()
        logging.info("[%s] Wrote %s", label, out)

    if sessions_per_week:
        spw = np.array(sessions_per_week, dtype=int)
        max_bin = max(10, int(np.max(spw)) + 2)
        bins = np.arange(0, max_bin + 1, 1)
        plt.figure()
        plt.hist(spw, bins=bins, align="left")
        plt.title(f"Sessions per Week — {label} (Active User-Weeks)")
        plt.xlabel("Sessions in week")
        plt.ylabel("Count")
        out = f"{label.replace(' ', '_').lower()}_sessions_per_week_hist.png"
        plt.savefig(out, dpi=160, bbox_inches="tight")
        plt.close()
        logging.info("[%s] Wrote %s", label, out)


def write_excel(results, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / f"{results.label.replace(' ', '_').lower()}_openwebui_stats.xlsx"

    df_weeks = pd.DataFrame(results.active_user_week_rows)
    summary_dict = summarize(
        results.label,
        results.session_durations_active,
        results.sessions_per_active_week,
        results.active_users_total,
    )
    df_summary = pd.DataFrame([{
        **summary_dict,
        "range_start_utc": results.start.isoformat(),
        "range_end_utc": results.end.isoformat(),
    }])

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="summary")
        df_weeks.to_excel(writer, index=False, sheet_name="active_user_weeks")

    logging.info("[%s] Wrote Excel: %s", results.label, str(xlsx_path))
    return xlsx_path


def print_terminal_summary(results) -> None:
    s = summarize(
        results.label,
        results.session_durations_active,
        results.sessions_per_active_week,
        results.active_users_total,
    )
    logging.info("======== SUMMARY (%s) ========", results.label)
    logging.info("UTC range: %s  ->  %s", results.start.isoformat(), results.end.isoformat())
    logging.info("Active users (distinct): %s", s["active_users_total"])
    logging.info("Active sessions (total): %s", s["active_sessions_total"])
    logging.info("Mean session minutes: %s", None if s["mean_session_minutes"] is None else round(s["mean_session_minutes"], 3))
    logging.info("Median session minutes: %s", None if s["median_session_minutes"] is None else round(s["median_session_minutes"], 3))
    logging.info("Mean sessions per active week: %s", None if s["mean_sessions_per_active_week"] is None else round(s["mean_sessions_per_active_week"], 3))
    logging.info("Median sessions per active week: %s", None if s["median_sessions_per_active_week"] is None else round(s["median_sessions_per_active_week"], 3))


# ----------------------------
# Time range selection
# ----------------------------

def pick_october_range(conn, chat: ChatSchema, plan: TimeColumnPlan, preferred_year: Optional[int]) -> Optional[Tuple[datetime, datetime, str]]:
    now = datetime.now(timezone.utc)
    candidates = []
    if preferred_year:
        candidates.append(preferred_year)
    candidates.extend([now.year, now.year - 1])

    with conn.cursor() as cur:
        for y in candidates:
            start = datetime(y, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
            end = datetime(y, 11, 1, 0, 0, 0, tzinfo=timezone.utc)

            q = f"""
            SELECT EXISTS(
              SELECT 1 FROM {chat.schema}.{chat.table}
              WHERE {plan.range_where}
                AND {chat.user_col} IS NOT NULL
              LIMIT 1
            );
            """
            params = plan.range_params_builder(start, end)
            cur.execute(q, params)
            if bool(cur.fetchone()[0]):
                return start, end, f"October_{y}"
    return None


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="OpenWebUI usage stats (sessions, active users, histograms, Excel).")
    ap.add_argument("--schema", default="public", help="Schema name (default: public).")
    ap.add_argument("--table", default="chat", help="Table name (default: chat).")
    ap.add_argument("--user-col", default=None, help="Override user id column name (auto-detected if omitted).")
    ap.add_argument("--ts-col", default=None, help="Override timestamp column name (auto-detected if omitted).")
    ap.add_argument("--prefer-ts", default="created_at", help="Prefer this ts column name when auto-detecting (default created_at).")
    ap.add_argument("--ts-unit", default="auto", choices=["auto", "seconds", "milliseconds"],
                    help="If ts_col is BIGINT/INTEGER: epoch unit. Default auto.")
    ap.add_argument("--gap-minutes", type=int, default=10, help="Session gap threshold (minutes). Default: 10.")
    ap.add_argument("--required-weekdays", default="0,1,2,3",
                    help="Required weekdays (Python: 0=Mon..6=Sun) for 'active' week. Default Mon-Thu: 0,1,2,3")
    ap.add_argument("--itersize", type=int, default=5000, help="Server-side cursor fetch size (default: 5000).")
    ap.add_argument("--october-year", type=int, default=None, help="Prefer this year for October range (e.g., 2025).")
    ap.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR. Default INFO.")
    ap.add_argument("--show-columns", action="store_true", help="Print columns for the chat table and exit.")
    args = ap.parse_args()

    setup_logging(args.log_level)

    # Validate conn_str early
    try:
        _ = parse_pg_url_to_kwargs(conn_str)
    except Exception as e:
        logging.error("Connection string parsing failed: %s", e)
        sys.exit(2)

    required_weekdays = set(int(x.strip()) for x in args.required_weekdays.split(",") if x.strip() != "")
    if not required_weekdays:
        raise ValueError("required_weekdays cannot be empty")

    conn = connect()
    try:
        chat = detect_schema(conn, args.schema, args.table, args.user_col, args.ts_col, prefer_ts=args.prefer_ts)

        cols = get_columns(conn, args.schema, args.table)
        if args.show_columns:
            logging.info("Columns for %s.%s:", args.schema, args.table)
            for c, t in cols.items():
                logging.info("  - %s : %s", c, t)
            return

        plan = build_time_plan(conn, chat, args.ts_unit)
        logging.info("Time plan: ts_expr=%s | range_where=%s", plan.ts_expr, plan.range_where)

        now = datetime.now(timezone.utc)
        last30_start = now - timedelta(days=30)
        ranges: List[Tuple[str, datetime, datetime]] = [("Last_30_Days", last30_start, now)]

        oct_range = pick_october_range(conn, chat, plan, args.october_year)
        if oct_range:
            ostart, oend, olabel = oct_range
            ranges.append((olabel, ostart, oend))
        else:
            logging.warning("No October data found (checked preferred/current/previous year). Skipping October analysis.")

        out_dir = Path("stats output")

        for label, start, end in ranges:
            logging.info("---- Running analysis: %s ----", label)
            results = analyze_range(
                conn=conn,
                chat=chat,
                plan=plan,
                label=label,
                start=start,
                end=end,
                gap_minutes=args.gap_minutes,
                required_weekdays=required_weekdays,
                itersize=args.itersize,
            )
            print_terminal_summary(results)
            save_histograms(label, results.session_durations_active, results.sessions_per_active_week)
            write_excel(results, out_dir)
            # Close the transaction used by the server-side cursor and release resources.
            conn.commit()

        logging.info("Done.")
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()


if __name__ == "__main__":
    main()