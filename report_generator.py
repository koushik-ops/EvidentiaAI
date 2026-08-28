"""
DriveOps Report Generator
Generates comprehensive professional PDF accident forensic reports with multiple telemetry graphs,
behavior analysis, cause of accident predictions, and key metrics.
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
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ── Color Palette ───────────────────────────────────────────────────────
BLUE_PRIMARY = "#1A73E8"
BLUE_DARK = "#174EA6"
BLUE_LIGHT = "#E8F0FE"
BLUE_ACCENT = "#D2E3FC"
TEXT_DARK = "#202124"
TEXT_MUTED = "#5F6368"
WHITE = "#FFFFFF"
BORDER_LIGHT = "#DADCE0"

RED_ALERT = "#D93025"
YELLOW_ALERT = "#F9AB00"
GREEN_SAFE = "#1E8E3E"

CHART_COLORS = [
    "#1A73E8", "#34A853", "#FBBC04", "#EA4335", "#8AB4F8",
    "#81C995", "#FDE293", "#F28B82", "#AECBFA", "#A8DAB5",
]


def _hex_to_reportlab(hex_str):
    """Convert hex color string to reportlab Color."""
    h = hex_str.lstrip("#")
    return colors.Color(int(h[:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:], 16) / 255)


def _build_styles():
    """Create custom paragraph styles for the report."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=_hex_to_reportlab(TEXT_DARK),
        alignment=TA_LEFT,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=_hex_to_reportlab(TEXT_MUTED),
        alignment=TA_LEFT,
        spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=_hex_to_reportlab(BLUE_PRIMARY),
        spaceBefore=14,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=_hex_to_reportlab(TEXT_DARK),
        spaceAfter=6,
        leading=13,
    ))
    styles.add(ParagraphStyle(
        "ForensicReason",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        textColor=_hex_to_reportlab(TEXT_DARK),
        leading=13,
    ))
    styles.add(ParagraphStyle(
        "VerdictTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=_hex_to_reportlab(WHITE),
        alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        "VerdictSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=_hex_to_reportlab(WHITE),
        alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=_hex_to_reportlab(TEXT_DARK),
        alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        textColor=_hex_to_reportlab(TEXT_DARK),
        alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=_hex_to_reportlab(TEXT_DARK),
        alignment=TA_LEFT,
    ))
    return styles


# ── Chart Generation Functions ──────────────────────────────────────────

def _make_telemetry_wave_chart(raw_df, width=520, height=190):
    """
    Generate a 2-panel or twin-axis telemetry curve:
    Panel 1: Speed over time
    Panel 2: Acceleration (G-Force) & Jerk over time
    """
    if raw_df is None or raw_df.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width / 100, height / 100), dpi=120, sharex=True)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(hspace=0.25)

    samples = np.arange(len(raw_df))
    has_speed = "Speed" in raw_df.columns and not raw_df["Speed"].isna().all()
    
    # Speed Panel
    ax1.set_facecolor("#FAFBFC")
    if has_speed:
        speeds = pd.to_numeric(raw_df["Speed"], errors="coerce").fillna(0.0).values
        ax1.plot(samples, speeds, color=BLUE_PRIMARY, linewidth=1.8, label="Speed (km/h)")
        ax1.fill_between(samples, speeds, color=BLUE_PRIMARY, alpha=0.1)
        ax1.set_ylabel("Speed", fontsize=8, color=TEXT_MUTED, fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "Speed Signal Not Present", ha="center", va="center", transform=ax1.transAxes, color=TEXT_MUTED, fontsize=8)
    
    ax1.grid(True, linestyle="--", alpha=0.4, color=BORDER_LIGHT)
    ax1.tick_params(labelsize=7, colors=TEXT_MUTED)
    for spine in ax1.spines.values():
        spine.set_color(BORDER_LIGHT)

    # Acceleration / G-Force Panel
    ax2.set_facecolor("#FAFBFC")
    acc_col = "LinAccMag" if "LinAccMag" in raw_df.columns else "AccMag" if "AccMag" in raw_df.columns else "AccX" if "AccX" in raw_df.columns else None
    if acc_col and acc_col in raw_df.columns:
        accs = pd.to_numeric(raw_df[acc_col], errors="coerce").fillna(0.0).values
        ax2.plot(samples, accs, color="#EA4335", linewidth=1.6, label="Acc Magnitude (g)")
        ax2.fill_between(samples, accs, color="#EA4335", alpha=0.1)
        ax2.set_ylabel("G-Force (g)", fontsize=8, color=TEXT_MUTED, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "Acceleration Signal Not Present", ha="center", va="center", transform=ax2.transAxes, color=TEXT_MUTED, fontsize=8)

    ax2.set_xlabel("Sample / Time Step", fontsize=8, color=TEXT_MUTED, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4, color=BORDER_LIGHT)
    ax2.tick_params(labelsize=7, colors=TEXT_MUTED)
    for spine in ax2.spines.values():
        spine.set_color(BORDER_LIGHT)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_control_and_yaw_chart(raw_df, width=520, height=180):
    """
    Generate control dynamics chart:
    Panel 1: Throttle % vs Brake %
    Panel 2: Yaw Rate / Gyroscope Z
    """
    if raw_df is None or raw_df.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width / 100, height / 100), dpi=120, sharex=True)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(hspace=0.25)

    samples = np.arange(len(raw_df))

    # Throttle vs Brake
    ax1.set_facecolor("#FAFBFC")
    has_b_or_t = False
    if "ThrottlePct" in raw_df.columns:
        throttle = pd.to_numeric(raw_df["ThrottlePct"], errors="coerce").fillna(0.0).values
        if np.any(throttle > 0):
            ax1.plot(samples, throttle, color="#34A853", linewidth=1.5, label="Throttle %")
            has_b_or_t = True
    if "BrakePct" in raw_df.columns:
        brake = pd.to_numeric(raw_df["BrakePct"], errors="coerce").fillna(0.0).values
        if np.any(brake > 0):
            ax1.plot(samples, brake, color="#EA4335", linewidth=1.5, linestyle="--", label="Brake %")
            has_b_or_t = True

    if has_b_or_t:
        ax1.legend(loc="upper right", fontsize=7, framealpha=0.8)
        ax1.set_ylabel("Pedal %", fontsize=8, color=TEXT_MUTED, fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "Pedal Inputs (Brake/Throttle) Baseline (0%)", ha="center", va="center", transform=ax1.transAxes, color=TEXT_MUTED, fontsize=8)

    ax1.grid(True, linestyle="--", alpha=0.4, color=BORDER_LIGHT)
    ax1.tick_params(labelsize=7, colors=TEXT_MUTED)
    for spine in ax1.spines.values():
        spine.set_color(BORDER_LIGHT)

    # Yaw / Gyro
    ax2.set_facecolor("#FAFBFC")
    has_gyro = False
    if "GyroZ" in raw_df.columns:
        gyroz = pd.to_numeric(raw_df["GyroZ"], errors="coerce").fillna(0.0).values
        ax2.plot(samples, gyroz, color="#FBBC04", linewidth=1.5, label="Yaw Rate (GyroZ)")
        ax2.set_ylabel("Yaw Rate (rad/s)", fontsize=8, color=TEXT_MUTED, fontweight="bold")
        has_gyro = True
    elif "GyroMag" in raw_df.columns:
        gyromag = pd.to_numeric(raw_df["GyroMag"], errors="coerce").fillna(0.0).values
        ax2.plot(samples, gyromag, color="#FBBC04", linewidth=1.5, label="Gyro Mag")
        ax2.set_ylabel("Rotation Mag", fontsize=8, color=TEXT_MUTED, fontweight="bold")
        has_gyro = True
    else:
        ax2.text(0.5, 0.5, "Gyroscope Signal Not Present", ha="center", va="center", transform=ax2.transAxes, color=TEXT_MUTED, fontsize=8)

    ax2.set_xlabel("Sample / Time Step", fontsize=8, color=TEXT_MUTED, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4, color=BORDER_LIGHT)
    ax2.tick_params(labelsize=7, colors=TEXT_MUTED)
    for spine in ax2.spines.values():
        spine.set_color(BORDER_LIGHT)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_distribution_charts(behavior_counts, cause_counts, width=520, height=180):
    """
    Generate side-by-side distribution charts:
    Left: Driving Behavior Pie Chart
    Right: Incident / Accident Cause Bar Chart
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width / 100, height / 100), dpi=120)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(wspace=0.35)

    # Behavior Pie Chart
    if behavior_counts:
        labels = list(behavior_counts.keys())
        values = list(behavior_counts.values())
        colors_subset = [BLUE_PRIMARY, "#34A853", "#EA4335"][: len(labels)]
        ax1.pie(
            values,
            labels=labels,
            autopct="%1.0f%%",
            startangle=140,
            colors=colors_subset,
            textprops={"fontsize": 7, "color": TEXT_DARK},
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        )
        ax1.set_title("Driving Behavior", fontsize=9, fontweight="bold", color=TEXT_DARK, pad=8)
    else:
        ax1.text(0.5, 0.5, "No Behavior Data", ha="center", va="center", color=TEXT_MUTED, fontsize=8)

    # Cause Bar Chart
    ax2.set_facecolor("#FAFBFC")
    if cause_counts:
        # Clean labels
        clean_labels = [k.replace("_", " ").title() for k in cause_counts.keys()]
        values = list(cause_counts.values())
        y_pos = np.arange(len(clean_labels))
        bars = ax2.barh(y_pos, values, color=BLUE_PRIMARY, height=0.55, edgecolor="white")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(clean_labels, fontsize=7, color=TEXT_DARK)
        ax2.set_xlabel("Windows", fontsize=7, color=TEXT_MUTED)
        ax2.set_title("Accident Cause Breakdown", fontsize=9, fontweight="bold", color=TEXT_DARK, pad=8)
        ax2.invert_yaxis()
        for spine in ax2.spines.values():
            spine.set_color(BORDER_LIGHT)
        ax2.tick_params(left=False, labelsize=7, colors=TEXT_MUTED)
        ax2.grid(True, linestyle="--", alpha=0.3, axis="x", color=BORDER_LIGHT)
        for bar in bars:
            w = bar.get_width()
            ax2.text(w + max(values) * 0.03, bar.get_y() + bar.get_height() / 2, str(int(w)),
                     va="center", fontsize=7, color=TEXT_DARK, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No Cause Data", ha="center", va="center", color=TEXT_MUTED, fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf


def _build_verdict_banner(forensic_insights, styles):
    """Create a high-impact verdict box at the top of the report."""
    cause_name = forensic_insights.get("primary_cause_display", "Normal / Safe Driving")
    behavior = forensic_insights.get("primary_behavior", "NORMAL")
    severity = forensic_insights.get("severity", "LOW")
    reason = forensic_insights.get("primary_reason", "Telemetry conforms to normal driving standards.")
    
    bg_color_hex = BLUE_PRIMARY if severity == "LOW" else "#D93025" if severity == "CRITICAL" else "#E37400"
    
    banner_data = [
        [
            Paragraph(f"<b>PRIMARY CAUSE: {cause_name.upper()}</b>", styles["VerdictTitle"]),
            Paragraph(f"<b>SEVERITY: {severity}</b>  |  <b>BEHAVIOR: {behavior}</b>", styles["VerdictSubtitle"]),
        ],
        [
            Paragraph(f"<b>Forensic Diagnosis:</b> {reason}", ParagraphStyle("v_reason", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, textColor=_hex_to_reportlab(WHITE), leading=12)),
            Paragraph("", styles["Normal"]),
        ]
    ]

    banner_table = Table(banner_data, colWidths=[360, 160])
    banner_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _hex_to_reportlab(bg_color_hex)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("SPAN", (0, 1), (1, 1)),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    return banner_table


def _build_metrics_grid(telemetry_metrics, styles):
    """Build a 2-column or 4-cell key metrics table."""
    headers = [
        Paragraph("<b>Telemetry Metric</b>", styles["TableHeader"]),
        Paragraph("<b>Peak / Observed Value</b>", styles["TableHeader"]),
        Paragraph("<b>Telemetry Metric</b>", styles["TableHeader"]),
        Paragraph("<b>Peak / Observed Value</b>", styles["TableHeader"]),
    ]

    max_spd = f"{telemetry_metrics.get('max_speed', 0):.1f} km/h"
    avg_spd = f"{telemetry_metrics.get('avg_speed', 0):.1f} km/h"
    g_force = f"{telemetry_metrics.get('peak_g_force', 0):.2f} g"
    jerk = f"{telemetry_metrics.get('peak_jerk', 0):.2f} g/s"
    yaw = f"{telemetry_metrics.get('peak_yaw', 0):.3f} rad/s"
    spd_drop = f"{telemetry_metrics.get('speed_drop', 0):.1f} km/h"
    brake = f"{telemetry_metrics.get('max_brake', 0):.1f}%"
    throttle = f"{telemetry_metrics.get('max_throttle', 0):.1f}%"

    rows = [
        headers,
        [Paragraph("Maximum Speed", styles["TableCell"]), Paragraph(max_spd, styles["TableCellBold"]),
         Paragraph("Peak G-Force (Acc)", styles["TableCell"]), Paragraph(g_force, styles["TableCellBold"])],
        [Paragraph("Average Speed", styles["TableCell"]), Paragraph(avg_spd, styles["TableCellBold"]),
         Paragraph("Peak Jerk (Rate of Acc)", styles["TableCell"]), Paragraph(jerk, styles["TableCellBold"])],
        [Paragraph("Maximum Speed Drop", styles["TableCell"]), Paragraph(spd_drop, styles["TableCellBold"]),
         Paragraph("Peak Yaw (Lateral Turn)", styles["TableCell"]), Paragraph(yaw, styles["TableCellBold"])],
        [Paragraph("Peak Brake Input", styles["TableCell"]), Paragraph(brake, styles["TableCellBold"]),
         Paragraph("Peak Throttle Input", styles["TableCell"]), Paragraph(throttle, styles["TableCellBold"])],
    ]

    tbl = Table(rows, colWidths=[140, 120, 140, 120])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _hex_to_reportlab(BLUE_LIGHT)),
        ("GRID", (0, 0), (-1, -1), 0.5, _hex_to_reportlab(BORDER_LIGHT)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


def _build_preview_table(results_df, styles, max_rows=30):
    """Build detailed predictions table."""
    preview = results_df.head(max_rows)
    if preview.empty:
        return None

    pref_cols = [
        ("window_index", "Window", 45),
        ("prediction_label", "Behavior", 65),
        ("predicted_cause", "Predicted Cause", 110),
        ("cause_confidence", "Confidence", 65),
        ("cause_reason", "Forensic Reasoning / Strongest Signals", 235),
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
        ("BACKGROUND", (0, 0), (-1, 0), _hex_to_reportlab(BLUE_LIGHT)),
        ("GRID", (0, 0), (-1, -1), 0.4, _hex_to_reportlab(BORDER_LIGHT)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            tbl_styles.append(("BACKGROUND", (0, i), (-1, i), _hex_to_reportlab("#F9FAFC")))

    tbl.setStyle(TableStyle(tbl_styles))
    return tbl


def _add_header_footer(canvas, doc):
    """Draw decorative top line and footer on each page."""
    canvas.saveState()
    canvas.setStrokeColor(_hex_to_reportlab(BLUE_PRIMARY))
    canvas.setLineWidth(2.5)
    canvas.line(30, A4[1] - 25, A4[0] - 30, A4[1] - 25)

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(_hex_to_reportlab(TEXT_MUTED))
    canvas.drawString(30, 18, "DriveOps Vehicle Forensic AI System  •  Confidential Accident Analysis")
    canvas.drawRightString(A4[0] - 30, 18, f"Page {doc.page}")
    canvas.restoreState()


# ── Main Entry Point ───────────────────────────────────────────────────

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
    Generate a complete, multi-graph accident forensic report in PDF format.
    """
    styles = _build_styles()
    elements = []

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ── 1. Document Title Header ────────────────────────────────────────
    elements.append(Paragraph("DriveOps Vehicle Forensic & Accident Report", styles["ReportTitle"]))
    timestamp = datetime.now().strftime("%B %d, %Y  •  %I:%M %p")
    input_fn = os.path.basename(input_info.get("input_path", "Uploaded Telemetry")) if input_info else "Uploaded Telemetry"
    elements.append(Paragraph(f"Analyzed File: <b>{input_fn}</b>  |  Generated: {timestamp}", styles["ReportSubtitle"]))

    # ── 2. Accident Cause Verdict Banner ────────────────────────────────
    if forensic_insights:
        elements.append(_build_verdict_banner(forensic_insights, styles))
        elements.append(Spacer(1, 10))

    # ── 3. Key Telemetry Metrics ────────────────────────────────────────
    elements.append(Paragraph("Key Telemetry & Forensic Metrics", styles["SectionHeading"]))
    telemetry_metrics = forensic_insights.get("telemetry_metrics", {}) if forensic_insights else {}
    elements.append(_build_metrics_grid(telemetry_metrics, styles))
    elements.append(Spacer(1, 10))

    # ── 4. Graph Section 1: Telemetry Curves (Speed & G-Force) ──────────
    elements.append(Paragraph("Sensor Telemetry & Motion Waveforms", styles["SectionHeading"]))
    elements.append(Paragraph("Time-series tracking of vehicle velocity profile and resultant G-force acceleration magnitude.", styles["BodyTextCustom"]))

    wave_buf = _make_telemetry_wave_chart(raw_telemetry_df, width=520, height=170)
    if wave_buf:
        elements.append(Image(wave_buf, width=515, height=165))
        elements.append(Spacer(1, 8))

    # ── 5. Graph Section 2: Control Inputs & Lateral Dynamics ───────────
    control_buf = _make_control_and_yaw_chart(raw_telemetry_df, width=520, height=160)
    if control_buf:
        elements.append(Paragraph("Control Inputs & Angular Dynamics", styles["SectionHeading"]))
        elements.append(Paragraph("Driver pedal engagement (Throttle % / Brake %) and lateral yaw rotation rate.", styles["BodyTextCustom"]))
        elements.append(Image(control_buf, width=515, height=155))
        elements.append(Spacer(1, 8))

    # ── 6. Graph Section 3: ML Behavior & Cause Distributions ───────────
    behavior_counts = summary.get("behavior_counts", {})
    cause_counts = summary.get("cause_counts", {})
    if behavior_counts or cause_counts:
        dist_buf = _make_distribution_charts(behavior_counts, cause_counts, width=520, height=160)
        if dist_buf:
            elements.append(Paragraph("ML Model Incident & Behavior Distributions", styles["SectionHeading"]))
            elements.append(Image(dist_buf, width=515, height=155))
            elements.append(Spacer(1, 10))

    # ── 7. Detailed Forensic Window Breakdown Table ──────────────────────
    elements.append(PageBreak())
    elements.append(Paragraph("Forensic Window-by-Window Log", styles["SectionHeading"]))
    elements.append(Paragraph(f"Detailed classification and signal contribution for the first {min(max_table_rows, len(results_df))} sliding analysis windows.", styles["BodyTextCustom"]))

    tbl = _build_preview_table(results_df, styles, max_table_rows)
    if tbl:
        elements.append(tbl)

    # ── Build Document ──────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=32,
        bottomMargin=30,
        leftMargin=30,
        rightMargin=30,
        title="DriveOps Vehicle Forensic Report",
        author="DriveOps AI",
    )
    doc.build(elements, onFirstPage=_add_header_footer, onLaterPages=_add_header_footer)
    return output_path
