"""
DriveOps / Evidentia AI — Automated Cloud Evidence Sync Script
Batch-fetches unanalyzed accident telemetry files from Supabase Storage (evidentia-evidence),
runs the AI accident cause classifier, and generates PDF forensic reports.

Usage:
    python sync_supabase_evidence.py
    python sync_supabase_evidence.py --list
    python sync_supabase_evidence.py --file accidents/accident_20260828_233236.xlsx
"""

import os
import sys
import argparse
import logging
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import supabase_client
import test_model
import report_generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvidentiaCloudSync")


def sync_and_analyze_all(folder="accidents"):
    """Fetch and process all accident telemetry files from Supabase."""
    logger.info(f"Connecting to Supabase bucket to list evidence in '{folder}'...")
    
    res = supabase_client.list_cloud_evidence_files(folder_override=folder)
    if not res.get("success"):
        logger.error(f"Failed to list Supabase files: {res.get('error')}")
        return False

    files = res.get("files", [])
    logger.info(f"Found {len(files)} telemetry files in Supabase Storage ({res.get('bucket')}).")

    if not files:
        logger.info("No files to process.")
        return True

    assets = test_model.load_prediction_assets()
    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    success_count = 0
    for f in files:
        remote_path = f["remote_path"]
        name = f["name"]
        logger.info(f"==> Downloading & Analyzing: {remote_path} ({f.get('size_formatted')})")
        
        try:
            local_path = supabase_client.download_cloud_file(remote_path)
            
            # 1. Run AI analysis
            payload = test_model.auto_analyze_telemetry(local_path, assets=assets)
            
            # 2. Generate PDF Report
            base_name = os.path.splitext(name)[0]
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"{base_name}_Forensic_Report_{timestamp_str}.pdf"
            pdf_path = os.path.join(reports_dir, pdf_filename)
            
            report_generator.generate_report(
                results_df=payload["results"],
                summary=payload["summary"],
                output_path=pdf_path,
                input_info={"input_path": f"[SUPABASE] {remote_path}"},
                forensic_insights=payload.get("forensic_insights"),
                raw_telemetry_df=payload.get("processed_raw_df"),
            )
            
            forensic = payload.get("forensic_insights", {})
            logger.info(f"    ✅ Analysis Completed!")
            logger.info(f"    Primary Cause: {forensic.get('primary_cause_display')} (Severity: {forensic.get('severity')})")
            logger.info(f"    Forensic PDF:  {pdf_path}")

            # Auto-upload ONLY the PDF report to target Supabase Storage (evidentia bucket)
            # Format: report / {vehicle_id} / {report_filename}.pdf
            if os.path.exists(pdf_path) and pdf_path.lower().endswith(".pdf"):
                vehicle_id = supabase_client.extract_vehicle_id_from_telemetry(local_path)
                upload_res = supabase_client.upload_report_to_supabase(pdf_path, remote_filename=pdf_filename, vehicle_id=vehicle_id)
                if upload_res.get("success"):
                    logger.info(f"    ☁️ PDF Report uploaded to Target Storage: {upload_res.get('remote_path')}")
            
            success_count += 1
        except Exception as e:
            logger.error(f"    ❌ Error analyzing {remote_path}: {e}", exc_info=True)

    logger.info(f"Sync complete! Processed {success_count}/{len(files)} cloud files.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evidentia Supabase Cloud Evidence Sync")
    parser.add_argument("--list", action="store_true", help="List files in cloud bucket")
    parser.add_argument("--folder", type=str, default="accidents", help="Folder prefix (default: accidents)")
    parser.add_argument("--file", type=str, help="Download & analyze a single remote path")
    args = parser.parse_args()

    if args.list:
        res = supabase_client.list_cloud_evidence_files(folder_override=args.folder)
        if res.get("success"):
            print(f"\nBucket: {res.get('bucket')} | Folder: {res.get('folder')}")
            print(f"{'Filename':<45} | {'Size':<10} | {'Uploaded'}")
            print("-" * 80)
            for item in res.get("files", []):
                print(f"{item['name']:<45} | {item['size_formatted']:<10} | {item.get('created_at')}")
            print(f"\nTotal Files: {len(res.get('files', []))}\n")
        else:
            print(f"Error: {res.get('error')}")
    elif args.file:
        local_f = supabase_client.download_cloud_file(args.file)
        print(f"Downloaded to {local_f}, running analysis...")
        assets = test_model.load_prediction_assets()
        payload = test_model.auto_analyze_telemetry(local_f, assets=assets)
        print("\nPrimary Cause:", payload.get("forensic_insights", {}).get("primary_cause_display"))
        print("Severity:", payload.get("forensic_insights", {}).get("severity"))
        print("Safety Score:", payload.get("forensic_insights", {}).get("telemetry_metrics"))
    else:
        sync_and_analyze_all(folder=args.folder)
