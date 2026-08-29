"""
Evidentia AI / DriveOps — Realtime Cloud Auto-Extractor & Watcher
Polls Supabase Storage continuously, detects newly pushed accident telemetry files,
and instantly extracts, analyzes, and generates reports.
"""

import os
import sys
import time
import logging
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import supabase_client
import test_model
import report_generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("EvidentiaWatcher")

PROCESSED_LOG_FILE = os.path.join(BASE_DIR, "data", "processed_cloud_files.txt")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)


def load_processed_set():
    """Load set of already extracted/analyzed file names."""
    if os.path.exists(PROCESSED_LOG_FILE):
        with open(PROCESSED_LOG_FILE, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def mark_processed(remote_path):
    """Mark a file as processed."""
    with open(PROCESSED_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{remote_path}\n")


def watch_and_extract(interval_seconds=10, folder="accidents"):
    """Continuously poll Supabase storage for newly pushed accident files."""
    logger.info("================================================================")
    logger.info("  🚀 Starting Evidentia AI Cloud Auto-Extractor Daemon")
    logger.info(f"  Monitoring Supabase Bucket: evidentia-evidence/{folder}/")
    logger.info(f"  Polling Interval: {interval_seconds} seconds")
    logger.info("================================================================")

    # Pre-test connection
    test_res = supabase_client.test_supabase_connection()
    if not test_res.get("success"):
        logger.warning(f"⚠️ Supabase connection issue: {test_res.get('error')}")
        logger.info("Please configure Supabase URL & Key in supabase_config.json or the web UI.")

    processed_set = load_processed_set()
    logger.info(f"Loaded {len(processed_set)} previously processed records.")

    assets = None

    while True:
        try:
            res = supabase_client.list_cloud_evidence_files(folder_override=folder)
            if res.get("success"):
                files = res.get("files", [])
                new_files = [f for f in files if f["remote_path"] not in processed_set]

                if new_files:
                    logger.info(f"⚡ Detected {len(new_files)} NEW accident file(s) pushed to cloud!")
                    
                    if assets is None:
                        logger.info("Loading AI prediction models into memory...")
                        assets = test_model.load_prediction_assets()

                    for f in new_files:
                        remote_path = f["remote_path"]
                        name = f["name"]
                        logger.info(f"\n📥 Extracting: {remote_path} ({f.get('size_formatted')})...")
                        
                        try:
                            # 1. Download/extract locally
                            local_path = supabase_client.download_cloud_file(remote_path)
                            logger.info(f"   Saved to: {local_path}")

                            # 2. Run AI reconstruction
                            payload = test_model.auto_analyze_telemetry(local_path, assets=assets)
                            forensic = payload.get("forensic_insights", {})

                            # 3. Generate PDF Report
                            base_name = os.path.splitext(name)[0]
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            pdf_filename = f"{base_name}_Forensic_Report_{timestamp_str}.pdf"
                            pdf_path = os.path.join(REPORTS_DIR, pdf_filename)

                            report_generator.generate_report(
                                results_df=payload["results"],
                                summary=payload["summary"],
                                output_path=pdf_path,
                                input_info={"input_path": f"[SUPABASE CLOUD] {remote_path}"},
                                forensic_insights=payload.get("forensic_insights"),
                                raw_telemetry_df=payload.get("processed_raw_df"),
                            )

                            logger.info(f"   🎯 Cause Detected: {forensic.get('primary_cause_display')} [Severity: {forensic.get('severity')}]")
                            logger.info(f"   📄 PDF Report Generated: {pdf_filename}")

                            # Auto-upload ONLY the PDF report to target Supabase Storage (evidentia bucket)
                            # Format: report / {vehicle_id} / {report_filename}.pdf
                            if os.path.exists(pdf_path) and pdf_path.lower().endswith(".pdf"):
                                vehicle_id = supabase_client.extract_vehicle_id_from_telemetry(local_path)
                                upload_res = supabase_client.upload_report_to_supabase(pdf_path, remote_filename=pdf_filename, vehicle_id=vehicle_id)
                                if upload_res.get("success"):
                                    logger.info(f"   ☁️ PDF Report uploaded to Target Storage: {upload_res.get('remote_path')}")

                            # Mark as done
                            mark_processed(remote_path)
                            processed_set.add(remote_path)

                        except Exception as e:
                            logger.error(f"   ❌ Error extracting {remote_path}: {e}", exc_info=True)

            else:
                logger.debug(f"Polling status: {res.get('error')}")

        except Exception as e:
            logger.error(f"Watcher loop error: {e}")

        time.sleep(interval_seconds)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evidentia Supabase Live Cloud Watcher")
    parser.add_argument("--interval", type=int, default=10, help="Check interval in seconds (default: 10)")
    parser.add_argument("--folder", type=str, default="accidents", help="Folder prefix (default: accidents)")
    args = parser.parse_args()

    watch_and_extract(interval_seconds=args.interval, folder=args.folder)
