"""
DriveOps Forensic AI — Web Backend Server
Serves the modern web dashboard and provides REST APIs for:
- Telemetry file analysis (.xlsx, .csv, .json)
- Instant AI cause of accident & driving behavior prediction
- Automated professional PDF forensic report generation
- Interactive multi-channel telemetry visualization
"""

import os
import sys
import glob
import json
import time
import uuid
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import test_model
import report_generator
import supabase_client

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DriveOpsServer")

# Initialize Flask app
WEB_DIR = os.path.join(BASE_DIR, "web")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
UPLOADS_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB max upload

# In-memory history cache
INCIDENT_HISTORY = []

# Pre-load prediction assets once for blazing fast inference
PREDICTION_ASSETS = None

def get_prediction_assets():
    global PREDICTION_ASSETS
    if PREDICTION_ASSETS is None:
        try:
            logger.info("Pre-loading DriveOps AI prediction assets...")
            PREDICTION_ASSETS = test_model.load_prediction_assets()
            logger.info("AI prediction assets loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading prediction assets: {e}")
    return PREDICTION_ASSETS


def extract_telemetry_series(processed_raw_df, max_points=250):
    """Extract and downsample telemetry time-series for frontend interactive charting."""
    if processed_raw_df is None or processed_raw_df.empty:
        return {"timestamps": [], "speed": [], "acc_mag": [], "acc_x": [], "acc_y": [], "acc_z": [], "yaw": [], "jerk": []}

    df = processed_raw_df.copy()
    n_rows = len(df)
    
    # Downsample if very long to keep browser snappy
    step = max(1, n_rows // max_points)
    sampled = df.iloc[::step].copy()

    # Time / Index
    if "Timestamp" in sampled.columns:
        times = [str(t) for t in sampled["Timestamp"]]
    elif "Time" in sampled.columns:
        times = [f"{float(t):.2f}s" if isinstance(t, (int, float)) else str(t) for t in sampled["Time"]]
    else:
        times = [f"{i * 0.1:.1f}s" for i in range(len(sampled))]

    # Speed
    speed_col = None
    for c in ["Speed", "speed", "Speed_kmh", "speed_kmh", "vehicle_speed"]:
        if c in sampled.columns:
            speed_col = c
            break
    speeds = [float(v) if pd.notnull(v) else 0.0 for v in sampled[speed_col]] if speed_col else [0.0] * len(sampled)

    # Accelerations
    acc_mag_col = "LinAccMag" if "LinAccMag" in sampled.columns else ("AccMag" if "AccMag" in sampled.columns else None)
    acc_mag = [float(v) if pd.notnull(v) else 0.0 for v in sampled[acc_mag_col]] if acc_mag_col else [0.0] * len(sampled)

    acc_x = [float(v) if pd.notnull(v) else 0.0 for v in sampled["AccX"]] if "AccX" in sampled.columns else [0.0] * len(sampled)
    acc_y = [float(v) if pd.notnull(v) else 0.0 for v in sampled["AccY"]] if "AccY" in sampled.columns else [0.0] * len(sampled)
    acc_z = [float(v) if pd.notnull(v) else 0.0 for v in sampled["AccZ"]] if "AccZ" in sampled.columns else [0.0] * len(sampled)

    # Gyro / Yaw
    yaw = [float(v) if pd.notnull(v) else 0.0 for v in sampled["GyroZ"]] if "GyroZ" in sampled.columns else [0.0] * len(sampled)

    # Jerk (differential of acc_mag)
    if len(acc_mag) > 1:
        jerk = [0.0] + [float(abs(acc_mag[i] - acc_mag[i-1]) * 10) for i in range(1, len(acc_mag))]
    else:
        jerk = [0.0] * len(acc_mag)

    return {
        "timestamps": times,
        "speed": speeds,
        "acc_mag": acc_mag,
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "yaw": yaw,
        "jerk": jerk
    }


def process_telemetry_file(filepath, original_filename):
    """Run full AI telemetry analysis and generate PDF report."""
    assets = get_prediction_assets()
    
    # 1. Run AI analysis
    payload = test_model.auto_analyze_telemetry(filepath, assets=assets)
    
    # 2. Generate PDF report
    raw_base_name = os.path.splitext(original_filename)[0]
    import re
    clean_base = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', raw_base_name).strip('_')
    if not clean_base:
        clean_base = "Incident"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"{clean_base}_Forensic_Report_{timestamp_str}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
    
    cloud_pdf_url = None
    try:
        report_generator.generate_report(
            results_df=payload["results"],
            summary=payload["summary"],
            output_path=pdf_path,
            input_info={"input_path": payload.get("input_path", original_filename)},
            forensic_insights=payload.get("forensic_insights"),
            raw_telemetry_df=payload.get("processed_raw_df"),
        )
        # Automatically upload ONLY the PDF report to target Supabase Storage (evidentia bucket)
        # Format: report / {vehicle_id} / {report_filename}.pdf
        if os.path.exists(pdf_path) and pdf_path.lower().endswith(".pdf"):
            vehicle_id = supabase_client.extract_vehicle_id_from_telemetry(filepath)
            upload_res = supabase_client.upload_report_to_supabase(pdf_path, remote_filename=pdf_filename, vehicle_id=vehicle_id)
            if upload_res.get("success"):
                cloud_pdf_url = upload_res.get("cloud_url")
                logger.info(f"✅ PDF Report auto-synced to Cloud Storage: {upload_res.get('remote_path')}")
    except Exception as e:
        logger.error(f"Error generating or syncing PDF report: {e}", exc_info=True)
        pdf_filename = None

    # 3. Format structured response
    forensic = payload.get("forensic_insights", {})
    summary = payload.get("summary", {})
    metrics = forensic.get("telemetry_metrics", {})
    
    # Build clean cause probability breakdown
    cause_pcts = summary.get("cause_percentages", {})
    cause_breakdown = []
    for k, v in sorted(cause_pcts.items(), key=lambda x: x[1], reverse=True):
        display_name = test_model.CAUSE_DISPLAY_NAMES.get(k, k.replace("_", " ").title())
        cause_breakdown.append({
            "key": k,
            "name": display_name,
            "percentage": round(float(v), 1)
        })

    # Build behavior breakdown
    behavior_pcts = summary.get("behavior_percentages", {})
    behavior_breakdown = []
    for k, v in sorted(behavior_pcts.items(), key=lambda x: x[1], reverse=True):
        behavior_breakdown.append({
            "key": k,
            "name": k.replace("_", " ").title(),
            "percentage": round(float(v), 1)
        })

    # Extract time series for interactive chart
    processed_df = payload.get("processed_raw_df")
    telemetry_chart_data = extract_telemetry_series(processed_df)

    # Safety Score calculation (0 - 100)
    primary_cause = forensic.get("primary_cause_key", "normal_driving")
    peak_g = metrics.get("peak_g_force", 0.0)
    speed_drop = metrics.get("speed_drop", 0.0)
    
    if primary_cause == "normal_driving":
        safety_score = max(75, int(98 - peak_g * 5))
    elif primary_cause in ["rash_driving", "sharp_turning_or_skid"]:
        safety_score = max(35, int(65 - peak_g * 6))
    elif primary_cause in ["hard_braking"]:
        safety_score = max(40, int(70 - speed_drop * 0.4))
    else:  # possible_collision_impact or possible_brake_failure
        safety_score = max(5, int(30 - peak_g * 4))

    result_data = {
        "id": str(uuid.uuid4())[:8],
        "filename": original_filename,
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_kind": payload.get("input_kind", "Telemetry Stream"),
        "primary_cause": forensic.get("primary_cause_display", "Unknown"),
        "primary_cause_key": primary_cause,
        "primary_behavior": forensic.get("primary_behavior", "NORMAL"),
        "severity": forensic.get("severity", "LOW"),
        "severity_color": forensic.get("severity_color", "#34A853"),
        "primary_reason": forensic.get("primary_reason", "No critical anomaly detected."),
        "safety_score": safety_score,
        "confidence": round(float(forensic.get("cause_confidence", 0.85)) * 100, 1),
        "metrics": {
            "peak_g_force": round(float(metrics.get("peak_g_force", 0.0)), 2),
            "avg_g_force": round(float(metrics.get("avg_g_force", 0.0)), 2),
            "max_speed": round(float(metrics.get("max_speed", 0.0)), 1),
            "avg_speed": round(float(metrics.get("avg_speed", 0.0)), 1),
            "speed_drop": round(float(metrics.get("speed_drop", 0.0)), 1),
            "peak_jerk": round(float(metrics.get("peak_jerk", 0.0)), 2),
            "peak_yaw": round(float(metrics.get("peak_yaw", 0.0)), 2),
            "max_brake": round(float(metrics.get("max_brake", 0.0)), 1),
            "max_throttle": round(float(metrics.get("max_throttle", 0.0)), 1),
            "total_samples": metrics.get("total_duration_samples", 0),
        },
        "causes": cause_breakdown,
        "behaviors": behavior_breakdown,
        "chart_data": telemetry_chart_data,
        "pdf_report": f"/api/reports/{pdf_filename}" if pdf_filename else None,
        "pdf_filename": pdf_filename,
        "cloud_pdf_url": cloud_pdf_url
    }

    # Store in history
    INCIDENT_HISTORY.insert(0, {
        "id": result_data["id"],
        "filename": result_data["filename"],
        "analyzed_at": result_data["analyzed_at"],
        "primary_cause": result_data["primary_cause"],
        "severity": result_data["severity"],
        "safety_score": result_data["safety_score"],
        "peak_g_force": result_data["metrics"]["peak_g_force"],
        "pdf_filename": pdf_filename
    })

    return result_data


# ── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve frontend index."""
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Upload telemetry file (.xlsx, .csv, .json) and run analysis."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Selected file is empty"}), 400

    filename = secure_filename(file.filename)
    if not filename:
        filename = f"upload_{int(time.time())}.csv"

    # Save uploaded file
    save_path = os.path.join(UPLOADS_DIR, f"{int(time.time())}_{filename}")
    file.save(save_path)

    try:
        analysis_result = process_telemetry_file(save_path, filename)
        return jsonify(analysis_result)
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({"error": f"Failed to analyze telemetry: {str(e)}"}), 500


@app.route("/api/demo", methods=["GET"])
def api_demo():
    """Load a sample accident file for instant testing."""
    sample_type = request.args.get("type", "collision")
    sample_files = {
        "collision": os.path.join(BASE_DIR, "data", "sample_raw_input.csv"),
        "rash": os.path.join(BASE_DIR, "data", "sample_rash_input.csv"),
        "brake": os.path.join(BASE_DIR, "data", "sample_brake_failure_input.csv")
    }

    sample_path = sample_files.get(sample_type, sample_files["collision"])
    if not os.path.exists(sample_path):
        return jsonify({"error": "Sample file not found"}), 404

    try:
        sample_name = os.path.basename(sample_path)
        analysis_result = process_telemetry_file(sample_path, f"[DEMO] {sample_name}")
        return jsonify(analysis_result)
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
        return jsonify({"error": f"Demo run failed: {str(e)}"}), 500


@app.route("/api/reports/<path:filename>")
def api_download_report(filename):
    """Download or view generated PDF forensic report."""
    try:
        # Check direct existence first
        direct_path = os.path.join(REPORTS_DIR, filename)
        if os.path.isfile(direct_path):
            return send_from_directory(REPORTS_DIR, filename, mimetype="application/pdf", as_attachment=request.args.get("download", "0") == "1")
        
        # Fallback with sanitized name
        safe_name = secure_filename(filename)
        safe_path = os.path.join(REPORTS_DIR, safe_name)
        if os.path.isfile(safe_path):
            return send_from_directory(REPORTS_DIR, safe_name, mimetype="application/pdf", as_attachment=request.args.get("download", "0") == "1")
        
        # Search by partial match if needed
        for existing in os.listdir(REPORTS_DIR):
            if existing == filename or existing == safe_name:
                return send_from_directory(REPORTS_DIR, existing, mimetype="application/pdf", as_attachment=request.args.get("download", "0") == "1")

        return jsonify({"error": "Report file not found"}), 404
    except Exception as e:
        logger.error(f"Error serving report {filename}: {e}")
        return jsonify({"error": f"Failed to retrieve report: {str(e)}"}), 500


@app.route("/api/incidents", methods=["GET"])
def api_incidents():
    """Get list of previously analyzed incidents."""
    return jsonify({"incidents": INCIDENT_HISTORY[:30]})


@app.route("/api/status", methods=["GET"])
def api_status():
    """System health check."""
    return jsonify({
        "status": "online",
        "engine": "DriveOps XGBoost + Forensic Kinematics v2.4",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


# ── Cloud Evidence (Supabase) Routes ───────────────────────────────

@app.route("/api/cloud/config", methods=["GET", "POST"])
def api_cloud_config():
    """Get or update Supabase cloud configuration."""
    if request.method == "POST":
        data = request.get_json() or {}
        new_cfg = {}
        if "supabase_url" in data:
            new_cfg["supabase_url"] = data["supabase_url"].strip()
        if "supabase_key" in data and data["supabase_key"].strip():
            new_cfg["supabase_key"] = data["supabase_key"].strip()
        if "bucket_name" in data:
            new_cfg["bucket_name"] = data["bucket_name"].strip() or "evidentia-evidence"
        if "folder_prefix" in data:
            new_cfg["folder_prefix"] = data["folder_prefix"].strip()
        
        saved = supabase_client.save_config(new_cfg)
        test_res = supabase_client.test_supabase_connection()
        return jsonify({
            "saved": saved,
            "connection_test": test_res
        })
    
    # GET
    cfg = supabase_client.load_config()
    key = cfg.get("supabase_key", "")
    masked_key = (key[:6] + "..." + key[-4:]) if len(key) > 10 else ("***" if key else "")
    return jsonify({
        "supabase_url": cfg.get("supabase_url", ""),
        "supabase_key_masked": masked_key,
        "is_configured": bool(cfg.get("supabase_url") and cfg.get("supabase_key")),
        "bucket_name": cfg.get("bucket_name", "evidentia-evidence"),
        "folder_prefix": cfg.get("folder_prefix", "accidents")
    })


@app.route("/api/cloud/files", methods=["GET"])
def api_cloud_files():
    """List telemetry files (.xlsx, .csv, .json) in Supabase bucket."""
    folder = request.args.get("folder")
    result = supabase_client.list_cloud_evidence_files(folder_override=folder)
    return jsonify(result)


@app.route("/api/cloud/analyze", methods=["POST"])
def api_cloud_analyze():
    """Fetch a telemetry file from Supabase Storage and run AI analysis."""
    data = request.get_json() or {}
    remote_path = data.get("path")
    if not remote_path:
        return jsonify({"error": "Missing 'path' parameter for cloud file"}), 400

    try:
        logger.info(f"Fetching cloud evidence from Supabase: {remote_path}")
        local_path = supabase_client.download_cloud_file(remote_path)
        original_name = f"[CLOUD] {os.path.basename(remote_path)}"
        analysis_result = process_telemetry_file(local_path, original_name)
        analysis_result["source"] = "supabase_cloud"
        analysis_result["remote_path"] = remote_path
        return jsonify(analysis_result)
    except Exception as e:
        logger.error(f"Error fetching/analyzing cloud file '{remote_path}': {e}", exc_info=True)
        return jsonify({"error": f"Cloud evidence analysis failed: {str(e)}"}), 500


@app.route("/api/cloud/test", methods=["GET", "POST"])
def api_cloud_test():
    """Test connection to Supabase bucket."""
    res = supabase_client.test_supabase_connection()
    return jsonify(res)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 DriveOps Web UI Server running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

