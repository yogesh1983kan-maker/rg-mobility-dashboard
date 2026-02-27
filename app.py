from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly
import plotly.graph_objects as go
import json
from datetime import timedelta
import re
import numpy as np
import traceback

app = Flask(__name__)

FILE_PATH = "Data.xlsx"

BM_LF_FIXED = 70.00
BM_IPKM_FIXED = 47.00


# =========================
# Helpers
# =========================
def norm_col(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def format_number(x):
    try:
        return "{:,.2f}".format(round(float(x), 2))
    except Exception:
        return ""


def json_safe(obj):
    """Replace NaN/Inf with None (valid JSON) recursively."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def normalize_lf_to_percent(lf_series: pd.Series) -> pd.Series:
    """
    Ensure Load Factor is always in 0–100 scale.

    If data is in ratio (0–1), convert to percent by *100.
    Uses robust rule based on mean/max.
    """
    s = pd.to_numeric(lf_series, errors="coerce")
    s_clean = s.dropna()
    if s_clean.empty:
        return s

    m = float(s_clean.mean())
    mx = float(s_clean.max())
    if m <= 1.5 or mx <= 1.5:
        return s * 100.0
    return s


def normalize_ipkm(ipkm_series: pd.Series) -> pd.Series:
    s = pd.to_numeric(ipkm_series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def sparse_labels(values, max_labels=35):
    """
    Show labels on some points only (keeps graph readable).
    """
    n = len(values)
    if n == 0:
        return []
    step = max(1, n // max_labels)
    out = []
    for i, v in enumerate(values):
        if i % step == 0 or i == n - 1:
            out.append(f"{v:.2f}")
        else:
            out.append("")
    return out


# =========================
# Load Data
# =========================
def load_data():
    raw_df = pd.read_excel(FILE_PATH, sheet_name="RawData")
    field_df = pd.read_excel(FILE_PATH, sheet_name="Field Description")

    raw_df.columns = [norm_col(c) for c in raw_df.columns]
    field_df.columns = [norm_col(c) for c in field_df.columns]

    if ("field_name_in_rawdata" not in field_df.columns or
            "field_description_to_be_shown_in_html_web_page" not in field_df.columns):
        raise Exception(
            "Field Description must have columns: "
            "'Field Name in RawData' and 'Field Description to be shown in HTML Web Page'"
        )

    field_df["field_name_in_rawdata"] = field_df["field_name_in_rawdata"].apply(norm_col)
    mapping = dict(zip(
        field_df["field_name_in_rawdata"],
        field_df["field_description_to_be_shown_in_html_web_page"]
    ))

    # Detect ops_date column
    possible_date_cols = ["ops_date", "op_date", "operation_date", "trip_date", "date"]
    date_col = None
    for c in possible_date_cols:
        if c in raw_df.columns:
            date_col = c
            break
    if date_col is None:
        for c in raw_df.columns:
            if "date" in c:
                date_col = c
                break
    if date_col is None:
        print("RAW DATA COLUMNS:", raw_df.columns.tolist())
        raise Exception("No valid date column found in RawData sheet.")

    if date_col != "ops_date":
        raw_df.rename(columns={date_col: "ops_date"}, inplace=True)

    raw_df["ops_date"] = pd.to_datetime(raw_df["ops_date"], errors="coerce")
    raw_df = raw_df.dropna(subset=["ops_date"])

    return raw_df, mapping


# =========================
# Filters (from-only / to-only supported)
# =========================
def apply_filters(df, f):
    f = f or {}

    from_date = f.get("from_date")
    to_date = f.get("to_date")

    if from_date:
        fd = pd.to_datetime(from_date, errors="coerce")
        if pd.notna(fd):
            df = df[df["ops_date"] >= fd]

    if to_date:
        td = pd.to_datetime(to_date, errors="coerce")
        if pd.notna(td):
            df = df[df["ops_date"] < (td + pd.Timedelta(days=1))]  # inclusive end date

    if f.get("bus_no") and "bus_no" in df.columns:
        df = df[df["bus_no"].astype(str) == str(f["bus_no"])]

    if f.get("reg_no") and "reg_no" in df.columns:
        df = df[df["reg_no"].astype(str) == str(f["reg_no"])]

    if f.get("route_trip") and "route_trip" in df.columns:
        df = df[df["route_trip"].astype(str) == str(f["route_trip"])]

    return df


# =========================
# Fixed FCI
# =========================
def get_fixed_fci_value(df):
    if "fixed_fci" in df.columns:
        v = pd.to_numeric(df["fixed_fci"], errors="coerce").dropna()
        return float(v.iloc[0]) if len(v) else 0.0
    if "fci" in df.columns:
        v = pd.to_numeric(df["fci"], errors="coerce").dropna()
        return float(v.iloc[0]) if len(v) else 0.0
    return 0.0


# =========================
# Current Avg IPKM & LF (match RawData / Excel)
# =========================
def compute_current_ipkm_lf(df):
    fixed_fci = get_fixed_fci_value(df)

    # Prefer RawData columns (match Excel)
    if "current_ipkm" in df.columns and "current_lf" in df.columns:
        ipkm = normalize_ipkm(df["current_ipkm"])
        lf = normalize_lf_to_percent(df["current_lf"])

        current_ipkm_avg = float(ipkm.dropna().mean()) if ipkm.notna().any() else 0.0
        current_lf_avg = float(lf.dropna().mean()) if lf.notna().any() else 0.0
        return current_ipkm_avg, current_lf_avg, fixed_fci

    # Fallback compute row-wise using your formula:
    # current_ipkm = ttl_bfm / odo_kms
    # current_lf = (ipkm * fixed_fci)/100
    ttl_bfm = pd.to_numeric(df["ttl_bfm"], errors="coerce") if "ttl_bfm" in df.columns else pd.Series(dtype=float)
    odo_kms = pd.to_numeric(df["odo_kms"], errors="coerce") if "odo_kms" in df.columns else pd.Series(dtype=float)

    if len(ttl_bfm) == 0 or len(odo_kms) == 0:
        return 0.0, 0.0, fixed_fci

    ipkm_row = np.where(odo_kms.fillna(0).to_numpy() == 0, np.nan, (ttl_bfm.to_numpy() / odo_kms.to_numpy()))
    ipkm_row = pd.Series(ipkm_row)

    lf_row = (ipkm_row * fixed_fci) / 100.0 if fixed_fci else pd.Series([np.nan] * len(ipkm_row))

    current_ipkm_avg = float(ipkm_row.dropna().mean()) if ipkm_row.notna().any() else 0.0
    current_lf_avg = float(lf_row.dropna().mean()) if lf_row.notna().any() else 0.0

    return current_ipkm_avg, current_lf_avg, fixed_fci


# =========================
# Aggregation (Day / Week / Month)
# =========================
def add_period_key(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    """
    Adds a 'period' column used for grouping.
    grain: 'day' | 'week' | 'month'
    """
    grain = (grain or "day").lower().strip()
    out = df.copy()

    if grain == "month":
        out["period"] = out["ops_date"].dt.to_period("M").dt.start_time
        return out

    if grain == "week":
        # Week starting Monday (more standard for reporting)
        out["period"] = out["ops_date"].dt.to_period("W-MON").dt.start_time
        return out

    # default: day
    out["period"] = out["ops_date"].dt.floor("D")
    return out


def period_label(period_series: pd.Series, grain: str) -> pd.Series:
    grain = (grain or "day").lower().strip()

    if grain == "month":
        return period_series.dt.strftime("%b-%y")  # Feb-26
    if grain == "week":
        # show week range: "24-Feb–02-Mar"
        start = period_series
        end = period_series + pd.Timedelta(days=6)
        return start.dt.strftime("%d-%b") + "–" + end.dt.strftime("%d-%b")
    return period_series.dt.strftime("%d-%m-%y (%a)")


# =========================
# Forecast / Projection (more logical)
# Week-1: last 15 days weighted avg + trend adjustment
# Week-2: Week-1 + 2.5%
# =========================
def projection_from_last15(daily: pd.DataFrame):
    """
    daily columns: ops_date, ipkm, lf (already normalized)
    Returns week1_ipkm, week1_lf, week2_ipkm, week2_lf
    """
    daily = daily.sort_values("ops_date")
    last15 = daily.tail(15).copy()

    # Weighted average (recent days heavier)
    n = len(last15)
    w = np.arange(1, n + 1, dtype=float)  # 1..n
    w = w / w.sum()

    ipkm_vals = last15["ipkm"].to_numpy(dtype=float)
    lf_vals = last15["lf"].to_numpy(dtype=float)

    w_ipkm = float(np.nansum(ipkm_vals * w))
    w_lf = float(np.nansum(lf_vals * w))

    # Trend adjustment using simple linear regression slope vs day index
    # Adds a small portion of last trend to week-1 (keeps it logical but not aggressive)
    x = np.arange(n, dtype=float)
    def slope(y):
        mask = np.isfinite(y)
        if mask.sum() < 5:
            return 0.0
        xs = x[mask]
        ys = y[mask]
        xm, ym = xs.mean(), ys.mean()
        denom = np.sum((xs - xm) ** 2)
        if denom == 0:
            return 0.0
        return float(np.sum((xs - xm) * (ys - ym)) / denom)

    s_ipkm = slope(ipkm_vals)
    s_lf = slope(lf_vals)

    # Apply half-week trend impact (very conservative)
    trend_boost_days = 3.0
    week1_ipkm = w_ipkm + (s_ipkm * trend_boost_days)
    week1_lf = w_lf + (s_lf * trend_boost_days)

    # Week-2 = Week-1 + 2.5%
    week2_ipkm = week1_ipkm * 1.025
    week2_lf = week1_lf * 1.025

    return week1_ipkm, week1_lf, week2_ipkm, week2_lf


def weekly_forecast_text(df):
    if "current_ipkm" not in df.columns or "current_lf" not in df.columns:
        return "Projection not available (current_ipkm/current_lf not found)."

    temp = df.copy()
    temp["current_ipkm"] = normalize_ipkm(temp["current_ipkm"])
    temp["current_lf"] = normalize_lf_to_percent(temp["current_lf"])

    daily = temp.groupby("ops_date", as_index=False).agg(
        ipkm=("current_ipkm", "mean"),
        lf=("current_lf", "mean")
    ).sort_values("ops_date")

    if len(daily) < 5:
        return "Not enough data for projection."

    last_date = daily["ops_date"].max()

    week1_ipkm, week1_lf, week2_ipkm, week2_lf = projection_from_last15(daily)

    w1s = last_date + timedelta(days=1)
    w1e = last_date + timedelta(days=7)
    w2s = last_date + timedelta(days=8)
    w2e = last_date + timedelta(days=14)

    def fmt(d):
        return d.strftime("%d %b'%y")

    return (
        f"Week-1 ({fmt(w1s)}–{fmt(w1e)}): Projected IPKM {format_number(week1_ipkm)} | Projected LF {format_number(week1_lf)}"
        f"   ||   "
        f"Week-2 ({fmt(w2s)}–{fmt(w2e)}): Projected IPKM {format_number(week2_ipkm)} | Projected LF {format_number(week2_lf)} (+2.5%)"
    )


# =========================
# Graphs (Day/Week/Month avg) with inline labels (like Excel)
# =========================
def create_lf_graph(df, grain="day"):
    fig = go.Figure()

    if "current_lf" not in df.columns:
        fig.update_layout(title="Load Factor Comparison (missing current_lf)")
        return json_safe(json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)))

    temp = df.copy()
    temp["cur_lf"] = normalize_lf_to_percent(temp["current_lf"])
    temp = add_period_key(temp, grain)

    g = temp.groupby("period", as_index=False).agg(cur_lf=("cur_lf", "mean")).sort_values("period")
    g["label"] = period_label(g["period"], grain)

    # BM LF reference
    fig.add_trace(go.Scatter(
        x=g["label"],
        y=[BM_LF_FIXED] * len(g),
        name=f"BM LF (Fixed {BM_LF_FIXED:.2f})",
        mode="lines",
        line=dict(color="blue", width=4, dash="dash")
    ))

    # Current LF line + inline labels
    yvals = g["cur_lf"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).tolist()
    labels = sparse_labels(yvals, max_labels=35)

    fig.add_trace(go.Scatter(
        x=g["label"],
        y=yvals,
        name=f"Current LF (Avg) [{grain.title()}]",
        mode="lines+markers+text",
        line=dict(color="red", width=3),
        marker=dict(size=6),
        text=labels,
        textposition="top center",
        textfont=dict(size=12),
        hovertemplate="Period: %{x}<br>Current LF: %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Load Factor Comparison ({grain.title()}-wise Avg)",
        xaxis_title="Period",
        yaxis_title="LF",
        autosize=True,
        hovermode="x unified",
        yaxis=dict(rangemode="tozero"),
        margin=dict(l=40, r=30, t=50, b=80)
    )

    return json_safe(json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)))


def create_ipkm_graph(df, grain="day"):
    fig = go.Figure()

    if "current_ipkm" not in df.columns:
        fig.update_layout(title="IPKM Comparison (missing current_ipkm)")
        return json_safe(json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)))

    temp = df.copy()
    temp["cur_ipkm"] = normalize_ipkm(temp["current_ipkm"])
    temp = add_period_key(temp, grain)

    g = temp.groupby("period", as_index=False).agg(cur_ipkm=("cur_ipkm", "mean")).sort_values("period")
    g["label"] = period_label(g["period"], grain)

    # BM IPKM reference
    fig.add_trace(go.Scatter(
        x=g["label"],
        y=[BM_IPKM_FIXED] * len(g),
        name=f"BM IPKM (Fixed {BM_IPKM_FIXED:.2f})",
        mode="lines",
        line=dict(color="blue", width=4, dash="dash")
    ))

    # Current IPKM line + inline labels
    yvals = g["cur_ipkm"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).tolist()
    labels = sparse_labels(yvals, max_labels=35)

    fig.add_trace(go.Scatter(
        x=g["label"],
        y=yvals,
        name=f"Current IPKM (Avg) [{grain.title()}]",
        mode="lines+markers+text",
        line=dict(color="red", width=3),
        marker=dict(size=6),
        text=labels,
        textposition="top center",
        textfont=dict(size=12),
        hovertemplate="Period: %{x}<br>Current IPKM: %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"IPKM Comparison ({grain.title()}-wise Avg)",
        xaxis_title="Period",
        yaxis_title="IPKM",
        autosize=True,
        hovermode="x unified",
        yaxis=dict(rangemode="tozero"),
        margin=dict(l=40, r=30, t=50, b=80)
    )

    return json_safe(json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)))


# =========================
# Routes
# =========================
@app.route("/")
def index():
    df, _ = load_data()
    return render_template(
        "dashboard.html",
        bus_list=sorted(df["bus_no"].dropna().unique()) if "bus_no" in df.columns else [],
        reg_list=sorted(df["reg_no"].dropna().unique()) if "reg_no" in df.columns else [],
        route_list=sorted(df["route_trip"].dropna().unique()) if "route_trip" in df.columns else []
    )


@app.route("/data", methods=["POST"])
def data():
    try:
        df, mapping = load_data()
        filters = request.json or {}
        grain = (filters.get("grain") or "day").lower().strip()  # day/week/month

        df_f = apply_filters(df, filters)

        if df_f.empty:
            return jsonify({
                "kpis": [],
                "lf_graph": {"data": [], "layout": {}},
                "ipkm_graph": {"data": [], "layout": {}},
                "table_columns": [],
                "table": [],
                "forecast": "No data available for selected filters."
            })

        # KPI totals
        trip_count = df_f["trip_no"].count() if "trip_no" in df_f.columns else 0
        total_tickets = df_f["ttl_no_of_tkt"].sum() if "ttl_no_of_tkt" in df_f.columns else 0
        total_odo = df_f["odo_kms"].sum() if "odo_kms" in df_f.columns else 0
        total_amt = df_f["tt_tkt_amt"].sum() if "tt_tkt_amt" in df_f.columns else 0
        total_tax = df_f["ttl_tax"].sum() if "ttl_tax" in df_f.columns else 0
        total_bfm = df_f["ttl_bfm"].sum() if "ttl_bfm" in df_f.columns else 0

        avg_tickets = df_f["ttl_no_of_tkt"].mean() if "ttl_no_of_tkt" in df_f.columns else 0
        avg_odo = df_f["odo_kms"].mean() if "odo_kms" in df_f.columns else 0
        avg_amt = df_f["tt_tkt_amt"].mean() if "tt_tkt_amt" in df_f.columns else 0
        avg_tax = df_f["ttl_tax"].mean() if "ttl_tax" in df_f.columns else 0
        avg_bfm = df_f["ttl_bfm"].mean() if "ttl_bfm" in df_f.columns else 0

        # Current Avg IPKM & LF (RawData/Excel aligned)
        current_ipkm_avg, current_lf_avg, fixed_fci = compute_current_ipkm_lf(df_f)

        kpis = [
            {"label": "Total Trip Count", "value": format_number(trip_count)},
            {"label": "Total Tickets", "value": format_number(total_tickets)},
            {"label": "Total ODO KMs", "value": format_number(total_odo)},
            {"label": "Total Ticket Amount", "value": format_number(total_amt)},
            {"label": "Total Taxes", "value": format_number(total_tax)},
            {"label": "Total BFM", "value": format_number(total_bfm)},
            {"label": "Avg Tickets", "value": format_number(avg_tickets)},
            {"label": "Avg ODO KMs", "value": format_number(avg_odo)},
            {"label": "Avg Ticket Amount", "value": format_number(avg_amt)},
            {"label": "Avg Taxes", "value": format_number(avg_tax)},
            {"label": "Avg BFM", "value": format_number(avg_bfm)},
            {"label": "Fixed FCI Used", "value": format_number(fixed_fci)},
            {"label": "Current Avg IPKM (RawData current_ipkm)", "value": format_number(current_ipkm_avg)},
            {"label": "Current Avg Load Factor (RawData current_lf)", "value": format_number(current_lf_avg)},
        ]

        # Table in RawData column order
        df_table = df_f.loc[:, df_f.columns].copy()
        df_table.rename(columns=mapping, inplace=True)

        ops_col_display = mapping.get("ops_date", "ops_date")
        if ops_col_display in df_table.columns:
            df_table[ops_col_display] = pd.to_datetime(df_table[ops_col_display], errors="coerce").dt.strftime("%d-%m-%y")

        # Format numeric cells for display
        for col in df_table.columns:
            if pd.api.types.is_numeric_dtype(df_table[col]):
                df_table[col] = df_table[col].apply(format_number)

        payload = {
            "kpis": kpis,
            "lf_graph": create_lf_graph(df_f, grain=grain),
            "ipkm_graph": create_ipkm_graph(df_f, grain=grain),
            "table_columns": list(df_table.columns),
            "table": df_table.to_dict("records"),
            "forecast": weekly_forecast_text(df_f),
            "grain": grain
        }

        return jsonify(json_safe(payload))

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)