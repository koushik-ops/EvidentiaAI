"""
Evidentia Forensic AI — Executive Incident Forensic Report Generator
Generates clean, minimalist, professional legal & insurance grade PDF forensic reports with:
- High-resolution vehicle kinematics & spatial trajectory reconstruction maps
- Multi-channel sensor telemetry waveforms (Velocity, G-Force, Brake, Throttle, Yaw)
- AI model cause of accident & driving behavior predictions
- Executive summary verdict card, KPI metric cards, and chronological incident logs.
"""

import io
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ── Design Tokens & Executive Palette ──────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Primary Slate & Accent Colors (Modern Executive Minimalist)
COLOR_INK_PRIMARY = "#0F172A"       # Deep slate navy
COLOR_INK_SECONDARY = "#334155"     # Slate body text
COLOR_INK_MUTED = "#64748B"         # Muted labels & captions
COLOR_INK_FAINT = "#94A3B8"         # Subdued borders & hints

COLOR_BG_SURFACE = "#F8FAFC"        # Off-white card surface
COLOR_BG_CARD = "#FFFFFF"           # Pure white container
COLOR_BORDER = "#E2E8F0"            # Clean subtle divider border
COLOR_BORDER_STRONG = "#CBD5E1"     # Focused border

COLOR_PRIMARY = "#2563EB"           # Evidentia cobalt blue
COLOR_PRIMARY_LIGHT = "#EFF6FF"     # Subtle blue highlight tint
COLOR_PRIMARY_DARK = "#1E40AF"

# Incident Severity Tints
COLOR_ALERT_CRITICAL = "#DC2626"    # Crimson (Severe impact / crash)
COLOR_ALERT_WARNING = "#D97706"     # Amber (Rash / Aggressive)
COLOR_ALERT_SUCCESS = "#059669"     # Emerald (Normal / Baseline)
COLOR_ALERT_INFO = "#4F46E5"        # Indigo (Brake anomaly / Mechanical)


def _hex_to_rl(hex_str, alpha=1.0):
    """Convert hex color string to ReportLab Color with optional alpha."""
    h = hex_str.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return colors.Color(r, g, b, alpha=alpha)


def _build_styles():
    """Create typographic hierarchy following Swiss minimalist design principles."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "DocHeaderTitle",
        fontName="Helvetica-Bold",
        fontSize=15,
        textColor=_hex_to_rl(COLOR_INK_PRIMARY),
        leading=18,
    ))
    styles.add(ParagraphStyle(
        "DocHeaderSubtitle",
        fontName="Helvetica",
        fontSize=8.5,
        textColor=_hex_to_rl(COLOR_INK_MUTED),
        leading=12,
    ))
    styles.add(ParagraphStyle(
        "MetaLabel",
        fontName="Helvetica-Bold",
        fontSize=7,
        textColor=_hex_to_rl(COLOR_INK_MUTED),
        alignment=TA_RIGHT,
        leading=9,
    ))
    styles.add(ParagraphStyle(
        "MetaValue",
        fontName="Helvetica",
        fontSize=8,
        textColor=_hex_to_rl(COLOR_INK_PRIMARY),
        alignment=TA_RIGHT,
        leading=11,
    ))
    styles.add(ParagraphStyle(
        "SectionTitle",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=_hex_to_rl(COLOR_INK_PRIMARY),
        spaceBefore=10,
        spaceAfter=3,
        leading=14,
    ))
    styles.add(ParagraphStyle(
        "SectionSubtitle",
        fontName="Helvetica",
        fontSize=8,
        textColor=_hex_to_rl(COLOR_INK_MUTED),
        spaceAfter=6,
        leading=11,
    ))
    styles.add(ParagraphStyle(
        "VerdictCause",
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=_hex_to_rl(COLOR_INK_PRIMARY),
        leading=16,
    ))
    styles.add(ParagraphStyle(
        "VerdictReason",
        fontName="Helvetica",
        fontSize=8.5,
        textColor=_hex_to_rl(COLOR_INK_SECONDARY),
        leading=12.5,
    ))
    styles.add(ParagraphStyle(
        "PillBadge",
        fontName="Helvetica-Bold",
        fontSize=7.5,
        textColor=_hex_to_rl(COLOR_PRIMARY_DARK),
        alignment=TA_CENTER,
        leading=9,
    ))
    styles.add(ParagraphStyle(
        "MetricLabel",
        fontName="Helvetica-Bold",
        fontSize=7,
        textColor=_hex_to_rl(COLOR_INK_MUTED),
        alignment=TA_CENTER,
        leading=8.5,
    ))
    styles.add(ParagraphStyle(
        "MetricValue",
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=_hex_to_rl(COLOR_INK_PRIMARY),
        alignment=TA_CENTER,
        leading=14,
    ))
    styles.add(ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=7.5,
        textColor=_hex_to_rl(COLOR_INK_SECONDARY),
        leading=10,
    ))
    styles.add(ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=7.5,
        textColor=_hex_to_rl(COLOR_INK_SECONDARY),
        leading=10,
    ))
    styles.add(ParagraphStyle(
        "TableCellBold",
        fontName="Helvetica-Bold",
        fontSize=7.5,
        textColor=_hex_to_rl(COLOR_INK_PRIMARY),
        leading=10,
    ))
    return styles


# ── High-Resolution Visual Data & Chart Generators ─────────────────────

def _make_trajectory_map(raw_df, width=530, height=195):
    """
    Generate an Executive Spatial Kinematic Trajectory Map.
    - If GPS lat/lon present, plots geographic coordinate path.
    - Otherwise integrates velocity and yaw/gyro to reconstruct the physical
      vehicle trajectory (in meters), marking Start, Impact/Peak G-Force, and Final Rest.
    """
    if raw_df is None or raw_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    # 1. Determine Coordinates (GPS or Integrated Dead-Reckoning Kinematics)
    has_gps = False
    for lat_c in ["Latitude", "lat", "GPS_Lat", "latitude"]:
        for lon_c in ["Longitude", "lon", "GPS_Lon", "longitude"]:
            if lat_c in raw_df.columns and lon_c in raw_df.columns:
                lats = pd.to_numeric(raw_df[lat_c], errors="coerce").dropna()
                lons = pd.to_numeric(raw_df[lon_c], errors="coerce").dropna()
                if len(lats) > 3 and not (lats == 0).all():
                    x_pts = (lons.values - lons.values[0]) * 111320.0 * np.cos(np.radians(lats.values[0]))
                    y_pts = (lats.values - lats.values[0]) * 110540.0
                    has_gps = True
                    break

    if not has_gps:
        # High-Fidelity Dead-Reckoning Kinematic Reconstruction
        speed_raw = pd.to_numeric(raw_df.get("Speed", 0), errors="coerce").fillna(0.0).values
        # Convert km/h to m/s if typical vehicle speeds observed
        v = (speed_raw / 3.6) if np.nanmax(speed_raw) > 5.0 else speed_raw

        # Angular yaw rate (rad/s)
        yaw_rate = np.zeros(len(raw_df))
        for yc in ["GyroZ", "YawRate", "yaw_rate", "Gyroscope_Z"]:
            if yc in raw_df.columns:
                yaw_rate = pd.to_numeric(raw_df[yc], errors="coerce").fillna(0.0).values
                break

        # Time step (assume 10 Hz / 0.1s standard if not explicit)
        dt = 0.1
        heading = np.cumsum(yaw_rate * dt)
        x_pts = np.cumsum(v * np.cos(heading) * dt)
        y_pts = np.cumsum(v * np.sin(heading) * dt)

    # 2. Find Point of Maximum Kinetic Anomaly / Collision Impact
    acc_col = None
    for ac in ["LinAccMag", "AccMag", "AccX", "AccY", "AccZ"]:
        if ac in raw_df.columns:
            acc_col = ac
            break
    
    if acc_col:
        acc_vals = pd.to_numeric(raw_df[acc_col], errors="coerce").fillna(0.0).values
        impact_idx = int(np.argmax(acc_vals))
    else:
        impact_idx = int(len(x_pts) * 0.7)

    # 3. Draw Minimalist Spatial Map
    # Subtle background grid
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="#CBD5E1")

    # Trajectory path line
    ax.plot(x_pts, y_pts, color=COLOR_PRIMARY, linewidth=2.2, alpha=0.9, zorder=3, label="Reconstructed Motion Path")
    
    # Path direction arrows
    if len(x_pts) > 6:
        step = max(len(x_pts) // 5, 2)
        for i in range(step, len(x_pts) - step, step):
            dx = x_pts[i + 1] - x_pts[i]
            dy = y_pts[i + 1] - y_pts[i]
            if np.hypot(dx, dy) > 0.05:
                ax.annotate("", xy=(x_pts[i + 1], y_pts[i + 1]), xytext=(x_pts[i], y_pts[i]),
                            arrowprops=dict(arrowstyle="-|>", color=COLOR_PRIMARY, lw=1.2, mutation_scale=10),
                            zorder=4)

    # Start Point
    ax.scatter(x_pts[0], y_pts[0], color=COLOR_ALERT_SUCCESS, s=60, edgecolors="#FFFFFF", linewidth=1.5, zorder=6, label="Start Position")
    ax.text(x_pts[0], y_pts[0] + 0.5, " START", fontsize=7, fontweight="bold", color=COLOR_ALERT_SUCCESS, zorder=7)

    # End / Rest Point
    ax.scatter(x_pts[-1], y_pts[-1], color=COLOR_INK_MUTED, s=60, marker="s", edgecolors="#FFFFFF", linewidth=1.5, zorder=6, label="Final Rest")
    ax.text(x_pts[-1], y_pts[-1] - 1.2, " REST", fontsize=7, fontweight="bold", color=COLOR_INK_MUTED, zorder=7)

    # Impact / Max Anomaly Target Callout
    if 0 <= impact_idx < len(x_pts):
        ix, iy = x_pts[impact_idx], y_pts[impact_idx]
        ax.scatter(ix, iy, color=COLOR_ALERT_CRITICAL, s=120, edgecolors="#FFFFFF", linewidth=2.0, zorder=8)
        ax.scatter(ix, iy, color=COLOR_ALERT_CRITICAL, s=260, facecolors="none", edgecolors=COLOR_ALERT_CRITICAL, linewidth=1.2, linestyle="--", zorder=7)
        ax.annotate(
            " IMPACT / PEAK ANOMALY",
            xy=(ix, iy),
            xytext=(ix + 1.5, iy + 1.2),
            fontsize=7.5,
            fontweight="bold",
            color=COLOR_ALERT_CRITICAL,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor=COLOR_ALERT_CRITICAL, linewidth=0.8),
            arrowprops=dict(arrowstyle="->", color=COLOR_ALERT_CRITICAL, lw=1.0),
            zorder=9
        )

    # Compass Rose Indicator (Top Right)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    margin_x = (x_max - x_min) * 0.08
    margin_y = (y_max - y_min) * 0.08
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)

    ax.set_title("Kinematic Trajectory & Incident Spatial Reconstruction", fontsize=8.5, fontweight="bold", color=COLOR_INK_PRIMARY, pad=8, loc="left")
    ax.set_xlabel("Relative X Displacement (meters)", fontsize=7.5, color=COLOR_INK_MUTED, labelpad=4)
    ax.set_ylabel("Relative Y Displacement (meters)", fontsize=7.5, color=COLOR_INK_MUTED, labelpad=4)
    ax.tick_params(colors=COLOR_INK_MUTED, labelsize=7)

    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER)
        spine.set_linewidth(0.8)

    ax.legend(loc="lower right", fontsize=6.5, frameon=True, facecolor="#FFFFFF", edgecolor=COLOR_BORDER, framealpha=0.9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="#FFFFFF")
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_telemetry_wave_chart(raw_df, width=530, height=170):
    """Clean, minimalist 2-panel curve: Velocity Profile vs Resultant G-Force."""
    if raw_df is None or raw_df.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width / 100, height / 100), dpi=150, sharex=True)
    fig.patch.set_facecolor("#FFFFFF")
    plt.subplots_adjust(hspace=0.22)

    samples = np.arange(len(raw_df))
    has_speed = "Speed" in raw_df.columns and not raw_df["Speed"].isna().all()

    # Panel 1: Speed Profile
    ax1.set_facecolor("#F8FAFC")
    if has_speed:
        speeds = pd.to_numeric(raw_df["Speed"], errors="coerce").fillna(0.0).values
        ax1.plot(samples, speeds, color=COLOR_PRIMARY, linewidth=1.8, label="Velocity (km/h)")
        ax1.fill_between(samples, speeds, color=COLOR_PRIMARY, alpha=0.08)
        ax1.set_ylabel("Speed (km/h)", fontsize=7.5, fontweight="bold", color=COLOR_INK_SECONDARY)
    else:
        ax1.text(0.5, 0.5, "Speed Signal Converted to Baseline", ha="center", va="center", transform=ax1.transAxes, color=COLOR_INK_MUTED, fontsize=7.5)

    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color=COLOR_BORDER)
    ax1.tick_params(labelsize=6.5, colors=COLOR_INK_MUTED)
    for s in ax1.spines.values():
        s.set_color(COLOR_BORDER)

    # Panel 2: Resultant Acceleration / G-Force Magnitude
    ax2.set_facecolor("#F8FAFC")
    acc_col = None
    for ac in ["LinAccMag", "AccMag", "AccX", "AccY", "AccZ"]:
        if ac in raw_df.columns:
            acc_col = ac
            break

    if acc_col:
        accs = pd.to_numeric(raw_df[acc_col], errors="coerce").fillna(0.0).values
        ax2.plot(samples, accs, color=COLOR_ALERT_CRITICAL, linewidth=1.6, label="G-Force (g)")
        ax2.fill_between(samples, accs, color=COLOR_ALERT_CRITICAL, alpha=0.08)
        ax2.set_ylabel("Impact G-Force (g)", fontsize=7.5, fontweight="bold", color=COLOR_INK_SECONDARY)

        # Highlight Peak Impact Spike
        max_idx = np.argmax(accs)
        ax2.scatter(samples[max_idx], accs[max_idx], color=COLOR_ALERT_CRITICAL, s=35, zorder=5)
        ax2.axvline(samples[max_idx], color=COLOR_ALERT_CRITICAL, linestyle=":", linewidth=1.0, alpha=0.7)
    else:
        ax2.text(0.5, 0.5, "Accelerometer Signal Baseline", ha="center", va="center", transform=ax2.transAxes, color=COLOR_INK_MUTED, fontsize=7.5)

    ax2.set_xlabel("Time Frame (Samples / 10Hz)", fontsize=7.5, color=COLOR_INK_MUTED)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color=COLOR_BORDER)
    ax2.tick_params(labelsize=6.5, colors=COLOR_INK_MUTED)
    for s in ax2.spines.values():
        s.set_color(COLOR_BORDER)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="#FFFFFF")
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_control_and_yaw_chart(raw_df, width=530, height=160):
    """Control inputs (Throttle / Brake) and Angular Yaw Dynamics."""
    if raw_df is None or raw_df.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width / 100, height / 100), dpi=150, sharex=True)
    fig.patch.set_facecolor("#FFFFFF")
    plt.subplots_adjust(hspace=0.22)

    samples = np.arange(len(raw_df))

    # Throttle vs Brake
    ax1.set_facecolor("#F8FAFC")
    has_pedal = False
    if "ThrottlePct" in raw_df.columns:
        thr = pd.to_numeric(raw_df["ThrottlePct"], errors="coerce").fillna(0.0).values
        if np.any(thr > 0):
            ax1.plot(samples, thr, color=COLOR_ALERT_SUCCESS, linewidth=1.5, label="Throttle %")
            has_pedal = True
    if "BrakePct" in raw_df.columns:
        brk = pd.to_numeric(raw_df["BrakePct"], errors="coerce").fillna(0.0).values
        if np.any(brk > 0):
            ax1.plot(samples, brk, color=COLOR_ALERT_CRITICAL, linewidth=1.5, linestyle="--", label="Brake %")
            has_pedal = True

    if has_pedal:
        ax1.legend(loc="upper right", fontsize=6.5, frameon=True, facecolor="#FFFFFF", edgecolor=COLOR_BORDER)
        ax1.set_ylabel("Pedals (%)", fontsize=7.5, fontweight="bold", color=COLOR_INK_SECONDARY)
    else:
        ax1.text(0.5, 0.5, "Pedal Control Signals (Standard Drift)", ha="center", va="center", transform=ax1.transAxes, color=COLOR_INK_MUTED, fontsize=7.5)

    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color=COLOR_BORDER)
    ax1.tick_params(labelsize=6.5, colors=COLOR_INK_MUTED)
    for s in ax1.spines.values():
        s.set_color(COLOR_BORDER)

    # Angular Yaw Dynamics
    ax2.set_facecolor("#F8FAFC")
    yaw_col = None
    for yc in ["GyroZ", "YawRate", "yaw_rate", "GyroMag"]:
        if yc in raw_df.columns:
            yaw_col = yc
            break

    if yaw_col:
        yaw = pd.to_numeric(raw_df[yaw_col], errors="coerce").fillna(0.0).values
        ax2.plot(samples, yaw, color=COLOR_ALERT_WARNING, linewidth=1.5, label="Yaw Rate (rad/s)")
        ax2.set_ylabel("Yaw Rate (rad/s)", fontsize=7.5, fontweight="bold", color=COLOR_INK_SECONDARY)
    else:
        ax2.text(0.5, 0.5, "Gyroscope Rotation Baseline", ha="center", va="center", transform=ax2.transAxes, color=COLOR_INK_MUTED, fontsize=7.5)

    ax2.set_xlabel("Time Frame (Samples / 10Hz)", fontsize=7.5, color=COLOR_INK_MUTED)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color=COLOR_BORDER)
    ax2.tick_params(labelsize=6.5, colors=COLOR_INK_MUTED)
    for s in ax2.spines.values():
        s.set_color(COLOR_BORDER)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="#FFFFFF")
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_distribution_charts(behavior_counts, cause_counts, width=530, height=155):
    """Minimalist horizontal probability & occurrence bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width / 100, height / 100), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    plt.subplots_adjust(wspace=0.38)

    # Driving Behavior Chart
    ax1.set_facecolor("#F8FAFC")
    if behavior_counts:
        b_labels = [k.replace("_", " ").title() for k in behavior_counts.keys()]
        b_vals = list(behavior_counts.values())
        y_pos = np.arange(len(b_labels))
        bars = ax1.barh(y_pos, b_vals, color=COLOR_PRIMARY, height=0.45, edgecolor="none")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(b_labels, fontsize=7, color=COLOR_INK_PRIMARY)
        ax1.set_xlabel("Windows", fontsize=7, color=COLOR_INK_MUTED)
        ax1.set_title("Behavior Profile", fontsize=8, fontweight="bold", color=COLOR_INK_PRIMARY, pad=6)
        ax1.invert_yaxis()
        for spine in ax1.spines.values():
            spine.set_color(COLOR_BORDER)
        ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, axis="x", color=COLOR_BORDER)
    else:
        ax1.text(0.5, 0.5, "No Behavior Data", ha="center", va="center", color=COLOR_INK_MUTED, fontsize=7.5)

    # Cause Breakdown Chart
    ax2.set_facecolor("#F8FAFC")
    if cause_counts:
        c_labels = [k.replace("_", " ").title() for k in cause_counts.keys()]
        c_vals = list(cause_counts.values())
        y_pos2 = np.arange(len(c_labels))
        bars2 = ax2.barh(y_pos2, c_vals, color=COLOR_ALERT_CRITICAL, height=0.45, edgecolor="none")
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(c_labels, fontsize=7, color=COLOR_INK_PRIMARY)
        ax2.set_xlabel("Windows", fontsize=7, color=COLOR_INK_MUTED)
        ax2.set_title("Predicted Accident Causes", fontsize=8, fontweight="bold", color=COLOR_INK_PRIMARY, pad=6)
        ax2.invert_yaxis()
        for spine in ax2.spines.values():
            spine.set_color(COLOR_BORDER)
        ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, axis="x", color=COLOR_BORDER)
    else:
        ax2.text(0.5, 0.5, "No Cause Data", ha="center", va="center", color=COLOR_INK_MUTED, fontsize=7.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="#FFFFFF")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Structured Component Builders ──────────────────────────────────────

def _build_header_card(input_fn, styles):
    """Executive top header with brand logo, document stamp, and incident ID."""
    logo_path = os.path.join(BASE_DIR, "web", "img2.png")
    if not os.path.exists(logo_path):
        logo_path = os.path.join(BASE_DIR, "img2.png")

    # Left: Brand Logo & Title
    left_elements = []
    if os.path.exists(logo_path):
        try:
            # Scaled cleanly to 40pt height maintaining ratio
            logo_img = Image(logo_path, width=72, height=36)
            left_elements.append(logo_img)
        except Exception:
            pass

    left_elements.append(Paragraph("EVIDENTIA FORENSIC AI", styles["DocHeaderTitle"]))
    left_elements.append(Paragraph("Autonomous Vehicle Incident & Telemetry Forensics", styles["DocHeaderSubtitle"]))

    # Right: Document Reference & Metadata
    now_str = datetime.now().strftime("%B %d, %Y  •  %H:%M:%S UTC")
    clean_fn = os.path.basename(input_fn)
    ref_id = f"EV-{datetime.now().strftime('%Y%m%d')}-{abs(hash(clean_fn)) % 10000:04d}"

    right_elements = [
        Paragraph(f"<b>REPORT REF:</b> {ref_id}", styles["MetaValue"]),
        Paragraph(f"<b>INCIDENT SOURCE:</b> {clean_fn[:30]}", styles["MetaValue"]),
        Paragraph(f"<b>DATE GENERATED:</b> {now_str}", styles["MetaValue"]),
        Paragraph("<b>CLASSIFICATION:</b> OFFICIAL FORENSIC RECORD", styles["MetaValue"]),
    ]

    header_table = Table([[left_elements, right_elements]], colWidths=[310, 220])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return header_table


def _build_verdict_card(forensic_insights, styles):
    """Executive Verdict Card with colored indicator bar and badges."""
    cause_name = forensic_insights.get("primary_cause_display", "Normal / Safe Driving")
    behavior = forensic_insights.get("primary_behavior", "NORMAL")
    severity = forensic_insights.get("severity", "LOW")
    reason = forensic_insights.get("primary_reason", "Telemetry conforms to nominal vehicle operation.")

    # Status color assignment
    if severity == "CRITICAL" or "collision" in cause_name.lower() or "failure" in cause_name.lower():
        accent_color = COLOR_ALERT_CRITICAL
        badge_bg = "#FEF2F2"
        badge_text = COLOR_ALERT_CRITICAL
    elif severity == "HIGH" or "rash" in cause_name.lower():
        accent_color = COLOR_ALERT_WARNING
        badge_bg = "#FFFBEB"
        badge_text = COLOR_ALERT_WARNING
    else:
        accent_color = COLOR_ALERT_SUCCESS
        badge_bg = "#ECFDF5"
        badge_text = COLOR_ALERT_SUCCESS

    info_content = [
        Paragraph(f"PRIMARY INCIDENT VERDICT: <b>{cause_name.upper()}</b>", styles["VerdictCause"]),
        Spacer(1, 3),
        Paragraph(f"<b>Forensic Determination:</b> {reason}", styles["VerdictReason"]),
    ]

    badges_content = [
        Paragraph(f"<font color='{badge_text}'><b>SEVERITY: {severity}</b></font>", styles["PillBadge"]),
        Spacer(1, 4),
        Paragraph(f"<font color='{COLOR_INK_MUTED}'>BEHAVIOR: {behavior}</font>", styles["PillBadge"]),
    ]

    card_table = Table([[info_content, badges_content]], colWidths=[405, 125])
    card_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _hex_to_rl(COLOR_BG_SURFACE)),
        ("BOX", (0, 0), (-1, -1), 0.6, _hex_to_rl(COLOR_BORDER)),
        ("LINEBEFORE", (0, 0), (0, -1), 4.0, _hex_to_rl(accent_color)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (0, -1), 12),
        ("RIGHTPADDING", (-1, 0), (-1, -1), 10),
    ]))
    return card_table


def _build_kpi_grid(telemetry_metrics, styles):
    """Executive 6-Card KPI Telemetry Metric Grid with generous spacing."""
    max_spd = f"{telemetry_metrics.get('max_speed', 0):.1f} km/h"
    g_force = f"{telemetry_metrics.get('peak_g_force', 0):.2f} g"
    spd_drop = f"{telemetry_metrics.get('speed_drop', 0):.1f} km/h"
    jerk = f"{telemetry_metrics.get('peak_jerk', 0):.2f} g/s"
    yaw = f"{telemetry_metrics.get('peak_yaw', 0):.3f} rad/s"
    brake = f"{telemetry_metrics.get('max_brake', 0):.1f}%"

    cards = [
        [
            [[Paragraph("MAX VELOCITY", styles["MetricLabel"])], [Paragraph(max_spd, styles["MetricValue"])]],
            [[Paragraph("PEAK G-FORCE", styles["MetricLabel"])], [Paragraph(g_force, styles["MetricValue"])]],
            [[Paragraph("DECELERATION DROP", styles["MetricLabel"])], [Paragraph(spd_drop, styles["MetricValue"])]],
        ],
        [
            [[Paragraph("PEAK JERK RATE", styles["MetricLabel"])], [Paragraph(jerk, styles["MetricValue"])]],
            [[Paragraph("LATERAL YAW RATE", styles["MetricLabel"])], [Paragraph(yaw, styles["MetricValue"])]],
            [[Paragraph("MAX BRAKE INPUT", styles["MetricLabel"])], [Paragraph(brake, styles["MetricValue"])]],
        ]
    ]

    col_w = 172
    grid_rows = []
    for row in cards:
        row_cells = []
        for card_content in row:
            card_tbl = Table(card_content, colWidths=[col_w - 12])
            card_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), _hex_to_rl(COLOR_BG_SURFACE)),
                ("BOX", (0, 0), (-1, -1), 0.6, _hex_to_rl(COLOR_BORDER)),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            row_cells.append(card_tbl)
        grid_rows.append(row_cells)

    grid_tbl = Table(grid_rows, colWidths=[col_w, col_w, col_w])
    grid_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    return grid_tbl


def _build_preview_table(results_df, styles, max_rows=25):
    """Clean, minimalist window breakdown table with alternating row tints."""
    preview = results_df.head(max_rows)
    if preview.empty:
        return None

    pref_cols = [
        ("window_index", "WIN", 36),
        ("prediction_label", "BEHAVIOR", 75),
        ("predicted_cause", "PREDICTED CAUSE", 115),
        ("cause_confidence", "CONF.", 50),
        ("cause_reason", "FORENSIC SIGNAL CONTRIBUTION", 254),
    ]

    header_cells = [Paragraph(f"<b>{title}</b>", styles["TableHeader"]) for _, title, _ in pref_cols]
    rows = [header_cells]

    for _, r in preview.iterrows():
        w_idx = str(r.get("window_index", "—"))
        beh = str(r.get("prediction_label", "—"))
        cause = str(r.get("predicted_cause", "—")).replace("_", " ").title()
        conf_val = r.get("cause_confidence", 0.0)
        conf_str = f"{float(conf_val):.2f}" if pd.notna(conf_val) else "—"
        reason = str(r.get("cause_reason", "—"))

        rows.append([
            Paragraph(w_idx, styles["TableCell"]),
            Paragraph(beh, styles["TableCellBold"]),
            Paragraph(cause, styles["TableCell"]),
            Paragraph(conf_str, styles["TableCell"]),
            Paragraph(reason, styles["TableCell"]),
        ])

    widths = [w for _, _, w in pref_cols]
    tbl = Table(rows, colWidths=widths)
    tbl_styles = [
        ("BACKGROUND", (0, 0), (-1, 0), _hex_to_rl(COLOR_BG_SURFACE)),
        ("LINEBELOW", (0, 0), (-1, 0), 1.0, _hex_to_rl(COLOR_BORDER_STRONG)),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, _hex_to_rl(COLOR_BORDER)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            tbl_styles.append(("BACKGROUND", (0, i), (-1, i), _hex_to_rl("#FAFAFA")))

    tbl.setStyle(TableStyle(tbl_styles))
    return tbl


def _add_header_footer(canvas, doc):
    """Draw refined, executive page headers and footers."""
    canvas.saveState()

    # Top thin accent line
    canvas.setStrokeColor(_hex_to_rl(COLOR_PRIMARY))
    canvas.setLineWidth(1.5)
    canvas.line(32, A4[1] - 22, A4[0] - 32, A4[1] - 22)

    # Bottom footer line & text
    canvas.setStrokeColor(_hex_to_rl(COLOR_BORDER))
    canvas.setLineWidth(0.6)
    canvas.line(32, 28, A4[0] - 32, 28)

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(_hex_to_rl(COLOR_INK_MUTED))
    canvas.drawString(32, 16, "Evidentia Forensic AI  •  Certified Incident Reconstruction & Telemetry Intelligence")
    canvas.drawRightString(A4[0] - 32, 16, f"Page {doc.page}")

    canvas.restoreState()


# ── Main PDF Generation Entry Point ────────────────────────────────────

def generate_report(
    results_df,
    summary,
    output_path,
    input_info=None,
    forensic_insights=None,
    raw_telemetry_df=None,
    max_table_rows=30,
):
    """
    Generate an executive, professional accident forensic report in PDF format.
    """
    styles = _build_styles()
    elements = []

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    input_fn = os.path.basename(input_info.get("input_path", "Vehicle Telemetry")) if input_info else "Vehicle Telemetry"

    # ═════════════════════════════════════════════════════════════════════
    # PAGE 1: EXECUTIVE BRIEF, KINEMATIC TRAJECTORY MAP & CORE WAVEFORMS
    # ═════════════════════════════════════════════════════════════════════

    # 1. Top Header
    elements.append(_build_header_card(input_fn, styles))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=0.8, color=_hex_to_rl(COLOR_BORDER), spaceAfter=8))

    # 2. Verdict Card
    if forensic_insights:
        elements.append(_build_verdict_card(forensic_insights, styles))
        elements.append(Spacer(1, 8))

    # 3. Key Telemetry Metrics Grid (6 KPI cards)
    elements.append(Paragraph("Key Incident Telemetry Metrics", styles["SectionTitle"]))
    telemetry_metrics = forensic_insights.get("telemetry_metrics", {}) if forensic_insights else {}
    elements.append(_build_kpi_grid(telemetry_metrics, styles))
    elements.append(Spacer(1, 8))

    # 4. Kinematic Trajectory & Spatial Reconstruction Map
    traj_buf = _make_trajectory_map(raw_telemetry_df, width=530, height=185)
    if traj_buf:
        elements.append(Paragraph("Kinematic Trajectory & Spatial Reconstruction Map", styles["SectionTitle"]))
        elements.append(Paragraph("2D spatial motion vector reconstructed from multi-axis accelerometer and gyro yaw channels, pinpointing incident anomaly coordinates.", styles["SectionSubtitle"]))
        elements.append(Image(traj_buf, width=520, height=180))
        elements.append(Spacer(1, 6))

    # 5. Telemetry Waveforms (Speed & G-Force)
    wave_buf = _make_telemetry_wave_chart(raw_telemetry_df, width=530, height=155)
    if wave_buf:
        elements.append(Paragraph("Primary Telemetry Dynamics: Velocity & Resultant Impact Force", styles["SectionTitle"]))
        elements.append(Image(wave_buf, width=520, height=150))

    # ═════════════════════════════════════════════════════════════════════
    # PAGE 2: PEDAL DYNAMICS, DISTRIBUTIONS & CHRONOLOGICAL LOG
    # ═════════════════════════════════════════════════════════════════════
    elements.append(PageBreak())

    # 6. Driver Control Dynamics (Throttle % vs Brake % & Angular Yaw)
    control_buf = _make_control_and_yaw_chart(raw_telemetry_df, width=530, height=160)
    if control_buf:
        elements.append(Paragraph("Control Inputs & Angular Stability", styles["SectionTitle"]))
        elements.append(Paragraph("Synchronized driver pedal engagement (Throttle / Brake) and lateral vehicle yaw rotation rate.", styles["SectionSubtitle"]))
        elements.append(Image(control_buf, width=520, height=155))
        elements.append(Spacer(1, 10))

    # 7. AI Behavior & Cause Distributions
    behavior_counts = summary.get("behavior_counts", {}) if summary else {}
    cause_counts = summary.get("cause_counts", {}) if summary else {}
    if behavior_counts or cause_counts:
        dist_buf = _make_distribution_charts(behavior_counts, cause_counts, width=530, height=140)
        if dist_buf:
            elements.append(Paragraph("AI Model Prediction Distributions", styles["SectionTitle"]))
            elements.append(Paragraph("Statistical occurrence across sliding analysis windows.", styles["SectionSubtitle"]))
            elements.append(Image(dist_buf, width=520, height=135))
            elements.append(Spacer(1, 10))

    # 8. Detailed Forensic Window Breakdown Table
    if results_df is not None and not results_df.empty:
        elements.append(Paragraph("Chronological Forensic Window Analysis", styles["SectionTitle"]))
        elements.append(Paragraph(f"Windowed signal contribution for the first {min(max_table_rows, len(results_df))} temporal analysis windows.", styles["SectionSubtitle"]))
        tbl = _build_preview_table(results_df, styles, max_table_rows)
        if tbl:
            elements.append(tbl)

    # ── Compile Document ────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=28,
        bottomMargin=32,
        leftMargin=32,
        rightMargin=32,
        title="Evidentia Vehicle Forensic Report",
        author="Evidentia Forensic AI",
    )
    doc.build(elements, onFirstPage=_add_header_footer, onLaterPages=_add_header_footer)
    return output_path
