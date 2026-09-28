"""
DriveOps / Evidentia AI — Supabase Cloud Storage Client
Fetches accident telemetry files (.xlsx, .csv, .json) directly from Supabase Storage
Bucket: evidentia-evidence (or configured bucket)
"""

import os
import sys
import json
import logging
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime

logger = logging.getLogger("DriveOpsSupabase")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "supabase_config.json")
CLOUD_DOWNLOADS_DIR = os.path.join(BASE_DIR, "data", "cloud_evidence")
os.makedirs(CLOUD_DOWNLOADS_DIR, exist_ok=True)


def get_default_config():
    """Return default configuration template."""
    return {
        "supabase_url": os.environ.get("SUPABASE_URL", "https://leyoprudlasvwmyhlvhu.supabase.co"),
        "supabase_key": os.environ.get("SUPABASE_KEY", ""),
        "bucket_name": os.environ.get("SUPABASE_BUCKET", "evidentia-evidence"),
        "folder_prefix": os.environ.get("SUPABASE_FOLDER", "accidents"),
        "table_name": os.environ.get("SUPABASE_TABLE", "telemetry"),
        # Target Cloud Storage for Generated Forensic Reports
        "report_supabase_url": os.environ.get("REPORT_SUPABASE_URL", "https://wsppfztyqycefdrcwzsr.supabase.co"),
        "report_supabase_key": os.environ.get("REPORT_SUPABASE_KEY", ""),
        "report_bucket_name": os.environ.get("REPORT_SUPABASE_BUCKET", "evidentia"),
        "report_folder_prefix": os.environ.get("REPORT_SUPABASE_FOLDER", "report"),
        "auto_upload_reports_to_cloud": True,
        "auto_analyze_on_fetch": True
    }


def load_config():
    """Load configuration from supabase_config.json or environment variables."""
    config = get_default_config()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                config.update(saved)
        except Exception as e:
            logger.error(f"Error loading {CONFIG_FILE}: {e}")
    return config


def save_config(new_config):
    """Save configuration to supabase_config.json."""
    config = load_config()
    config.update(new_config)
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Supabase configuration saved to {CONFIG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving {CONFIG_FILE}: {e}")
        return False


def _get_headers(api_key):
    return {
        "apiKey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Evidentia-AI-DriveOps/2.4"
    }


def test_supabase_connection(config=None):
    """Test connection and bucket access to Supabase."""
    if config is None:
        config = load_config()

    url = (config.get("supabase_url") or "").rstrip("/")
    key = config.get("supabase_key") or ""
    bucket = config.get("bucket_name") or "evidentia-evidence"

    if not url or not key:
        return {
            "success": False,
            "error": "Supabase URL and API Key are required. Please provide them in settings."
        }

    # Normalize url if someone enters https://xxxx.supabase.co
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    list_url = f"{url}/storage/v1/object/list/{bucket}"
    payload = json.dumps({
        "prefix": config.get("folder_prefix", "accidents"),
        "limit": 5,
        "sortBy": {"column": "created_at", "order": "desc"}
    }).encode("utf-8")

    req = urllib.request.Request(list_url, data=payload, headers=_get_headers(key), method="POST")

    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return {
                "success": True,
                "message": f"Connected to Supabase bucket '{bucket}' successfully!",
                "item_count_sample": len(res_data) if isinstance(res_data, list) else 0
            }
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        logger.error(f"Supabase connection HTTP error: {e.code} - {body}")
        return {
            "success": False,
            "error": f"HTTP {e.code}: {body or e.reason}"
        }
    except Exception as e:
        logger.error(f"Supabase connection failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def list_cloud_evidence_files(config=None, folder_override=None, limit=100):
    """List accident telemetry files (.xlsx, .csv, .json) in Supabase Storage."""
    if config is None:
        config = load_config()

    url = (config.get("supabase_url") or "").rstrip("/")
    key = config.get("supabase_key") or ""
    bucket = config.get("bucket_name") or "evidentia-evidence"
    folder = folder_override if folder_override is not None else config.get("folder_prefix", "accidents")

    if not url or not key:
        return {
            "success": False,
            "error": "Supabase URL or Key not configured",
            "files": []
        }

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    list_url = f"{url}/storage/v1/object/list/{bucket}"
    
    # Try specified folder first, and if empty/folder not found, check root
    folders_to_try = [folder]
    if folder and folder != "":
        folders_to_try.append("")

    all_files = []
    
    for prefix in folders_to_try:
        payload = json.dumps({
            "prefix": prefix,
            "limit": limit,
            "sortBy": {"column": "created_at", "order": "desc"}
        }).encode("utf-8")

        req = urllib.request.Request(list_url, data=payload, headers=_get_headers(key), method="POST")

        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                items = json.loads(response.read().decode("utf-8"))
                if isinstance(items, list):
                    for item in items:
                        name = item.get("name", "")
                        # Filter out directory placeholders
                        if not name or name == ".emptyFolderPlaceholder" or item.get("id") is None:
                            continue
                        
                        full_remote_path = f"{prefix}/{name}" if prefix else name
                        full_remote_path = full_remote_path.lstrip("/")

                        # Check extension
                        lower_name = name.lower()
                        if any(lower_name.endswith(ext) for ext in [".xlsx", ".xls", ".csv", ".json"]):
                            metadata = item.get("metadata", {})
                            size_bytes = metadata.get("size", item.get("size", 0))
                            
                            all_files.append({
                                "name": name,
                                "remote_path": full_remote_path,
                                "size": size_bytes,
                                "size_formatted": format_bytes(size_bytes),
                                "created_at": item.get("created_at", ""),
                                "updated_at": item.get("updated_at", ""),
                                "mimetype": metadata.get("mimetype", "application/octet-stream")
                            })
                    if all_files:
                        break
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            logger.warning(f"Error listing prefix '{prefix}': HTTP {e.code} - {body}")
            if prefix == folders_to_try[-1] and not all_files:
                return {
                    "success": False,
                    "error": f"HTTP {e.code}: {body or e.reason}",
                    "files": []
                }
        except Exception as e:
            logger.error(f"Error listing cloud files: {e}")
            if prefix == folders_to_try[-1] and not all_files:
                return {
                    "success": False,
                    "error": str(e),
                    "files": []
                }

    return {
        "success": True,
        "bucket": bucket,
        "folder": folder,
        "files": all_files
    }


def fetch_telemetry_from_database_table(table_name=None, limit=2000, target_local_path=None, config=None):
    """
    Fetch raw telemetry records directly from a Supabase PostgreSQL table (e.g. 'telemetry', 'accidents', 'sensor_data'),
    convert the records into a clean tabular DataFrame/CSV, and save locally.
    """
    if config is None:
        config = load_config()

    url = (config.get("supabase_url") or "").rstrip("/")
    key = config.get("supabase_key") or ""
    table = table_name or config.get("table_name") or "telemetry"

    if not url or not key:
        raise ValueError("Supabase URL and API Key are required to fetch database table records.")

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    query_url = f"{url}/rest/v1/{table}?select=*&order=created_at.desc&limit={limit}"
    req = urllib.request.Request(query_url, headers=_get_headers(key), method="GET")

    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            rows = json.loads(response.read().decode("utf-8"))
            if not isinstance(rows, list) or len(rows) == 0:
                raise ValueError(f"No records found in Supabase table '{table}'.")

            import pandas as pd
            df = pd.DataFrame(rows)
            
            # Sort ascending by timestamp if present
            for time_col in ["created_at", "timestamp", "Timestamp", "Time", "time"]:
                if time_col in df.columns:
                    df = df.sort_values(by=time_col).reset_index(drop=True)
                    break

            if not target_local_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_local_path = os.path.join(CLOUD_DOWNLOADS_DIR, f"supabase_table_{table}_{timestamp}.xlsx")

            if target_local_path.endswith(".csv"):
                df.to_csv(target_local_path, index=False)
            else:
                df.to_excel(target_local_path, index=False)

            logger.info(f"Fetched {len(df)} rows from table '{table}', saved to {target_local_path}")
            return target_local_path
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Failed to query Supabase table '{table}': HTTP {e.code} ({body or e.reason})")
    except Exception as e:
        raise RuntimeError(f"Error fetching table '{table}': {str(e)}")


def download_cloud_file(remote_path, target_local_path=None, config=None):
    """
    Download a file from Supabase Storage by its remote path.
    e.g. accidents/accident_20260828_233236.xlsx
    """
    if config is None:
        config = load_config()

    url = (config.get("supabase_url") or "").rstrip("/")
    key = config.get("supabase_key") or ""
    bucket = config.get("bucket_name") or "evidentia-evidence"

    if not url or not key:
        raise ValueError("Supabase URL and API Key are required to download files.")

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    clean_path = remote_path.lstrip("/")
    # Supabase Storage download endpoint: /storage/v1/object/authenticated/{bucket}/{path} or /storage/v1/object/{bucket}/{path}
    download_url = f"{url}/storage/v1/object/authenticated/{bucket}/{clean_path}"

    if not target_local_path:
        filename = os.path.basename(clean_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_local_path = os.path.join(CLOUD_DOWNLOADS_DIR, f"{timestamp}_{filename}")

    req = urllib.request.Request(download_url, headers=_get_headers(key), method="GET")

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()
            with open(target_local_path, "wb") as f:
                f.write(content)
            logger.info(f"Downloaded {remote_path} ({len(content)} bytes) to {target_local_path}")
            return target_local_path
    except urllib.error.HTTPError as e:
        # Fallback to standard object download endpoint
        alt_download_url = f"{url}/storage/v1/object/{bucket}/{clean_path}"
        req2 = urllib.request.Request(alt_download_url, headers=_get_headers(key), method="GET")
        try:
            with urllib.request.urlopen(req2, timeout=30) as response:
                content = response.read()
                with open(target_local_path, "wb") as f:
                    f.write(content)
                logger.info(f"Downloaded via fallback {remote_path} ({len(content)} bytes) to {target_local_path}")
                return target_local_path
        except Exception as e2:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Failed to download cloud file '{remote_path}': HTTP {e.code} ({body or str(e2)})")


VEHICLE_REGISTRY_FILE = os.path.join(BASE_DIR, "data", "vehicle_registry.json")


def load_vehicle_registry():
    """Load persistent vehicle mappings and next ID counter."""
    if os.path.exists(VEHICLE_REGISTRY_FILE):
        try:
            with open(VEHICLE_REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"mappings": {}, "next_id": 1}


def save_vehicle_registry(registry):
    """Save persistent vehicle mappings."""
    try:
        with open(VEHICLE_REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving vehicle registry: {e}")


def format_vehicle_id_folder(raw_id):
    """
    Format vehicle ID into standard directory name:
    e.g. 0 or 1 -> 'VehicleID-01'
    e.g. 2 -> 'VehicleID-02'
    e.g. '01' -> 'VehicleID-01'
    e.g. 'VIN123' -> 'VehicleID-VIN123'
    """
    import re
    if raw_id is None:
        return "VehicleID-01"
    clean = str(raw_id).strip()
    if not clean or clean.lower() in ("nan", "none", "null", "default", "vehicle_default", "vehicle"):
        return "VehicleID-01"
    
    # Strip any existing prefix
    clean_no_prefix = re.sub(r'^(?:vehicle_?id|veh_?id|vehicle|veh)[\-_]?', '', clean, flags=re.IGNORECASE).strip()
    if clean_no_prefix.isdigit():
        num = int(clean_no_prefix)
        formatted_num = f"{num:02d}" if num > 0 else "01"
        return f"VehicleID-{formatted_num}"
    
    clean_str = re.sub(r'[^a-zA-Z0-9_\-]', '_', clean_no_prefix or clean).strip('_')
    if not clean_str.lower().startswith("vehicleid-"):
        return f"VehicleID-{clean_str}"
    return clean_str


def extract_vehicle_id_from_telemetry(filepath_or_df=None):
    """
    Auto-allocates unique vehicle IDs for each distinct telemetry analysis:
    - 1st distinct accident report -> VehicleID-01
    - 2nd distinct accident report -> VehicleID-02
    - 3rd distinct accident report -> VehicleID-03, and so on.
    - Re-analyzing the same accident telemetry re-uses its allocated VehicleID.
    """
    import re

    # 1. Check if file has an explicit non-zero vehicle ID (e.g. VehicleID-02, veh_3, etc.)
    explicit_veh_id = None
    telemetry_sig = None

    if isinstance(filepath_or_df, str) and os.path.exists(filepath_or_df):
        base = os.path.basename(filepath_or_df)
        clean_name = re.sub(r'^[0-9]+_', '', base)
        telemetry_sig = os.path.splitext(clean_name)[0]

        try:
            if filepath_or_df.lower().endswith(('.xlsx', '.xls')):
                import pandas as pd
                xl = pd.ExcelFile(filepath_or_df)
                df0 = xl.parse(xl.sheet_names[0])
                if df0.shape[1] == 2:
                    kv = {str(r.iloc[0]).strip().lower(): str(r.iloc[1]).strip() for _, r in df0.iterrows()}
                    acc_id = kv.get("accident id", "") or kv.get("incident id", "")
                    if acc_id:
                        telemetry_sig = f"accident_{acc_id}"
                    
                    v_raw = kv.get("vehicle id", "") or kv.get("vehicle_id", "")
                    if v_raw and str(v_raw).strip() not in ("0", "00", ""):
                        explicit_veh_id = format_vehicle_id_folder(v_raw)
        except Exception as e:
            logger.debug(f"Vehicle ID extraction from file failed: {e}")

    # If explicit vehicle ID found, use it
    if explicit_veh_id:
        return explicit_veh_id

    # 2. Dynamic Auto-Allocation based on telemetry signature
    if not telemetry_sig:
        telemetry_sig = "session_telemetry_default"

    registry = load_vehicle_registry()
    mappings = registry.get("mappings", {})

    if telemetry_sig in mappings:
        return mappings[telemetry_sig]

    # Allocate next unique VehicleID
    next_num = registry.get("next_id", 1)
    allocated_id = f"VehicleID-{next_num:02d}"
    mappings[telemetry_sig] = allocated_id
    registry["next_id"] = next_num + 1
    registry["mappings"] = mappings
    save_vehicle_registry(registry)

    logger.info(f"Allocated unique Vehicle ID '{allocated_id}' for telemetry signature '{telemetry_sig}'")
    return allocated_id


def upload_report_to_supabase(local_pdf_path, remote_filename=None, vehicle_id=None, config=None):
    """
    Upload a generated PDF forensic report to target Supabase Storage with strict hierarchy:
    Format: report / VehicleID-XX / {report_filename}.pdf
    """
    if config is None:
        config = load_config()

    url = (config.get("report_supabase_url") or config.get("supabase_url") or "").rstrip("/")
    key = config.get("report_supabase_key") or config.get("supabase_key") or ""
    bucket = config.get("report_bucket_name") or "evidentia"
    root_folder = config.get("report_folder_prefix", "report").strip("/")

    if not url or not key:
        logger.warning("Target report storage credentials not configured. Skipping cloud report upload.")
        return {"success": False, "error": "Report storage URL/Key not configured"}

    if not os.path.exists(local_pdf_path):
        return {"success": False, "error": f"Local PDF report not found: {local_pdf_path}"}

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    if not remote_filename:
        remote_filename = os.path.basename(local_pdf_path)

    import re
    safe_remote_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', remote_filename).strip('_')
    if not safe_remote_filename.lower().endswith(".pdf"):
        safe_remote_filename += ".pdf"

    # Format vehicle folder: e.g. VehicleID-01, VehicleID-02
    vehicle_folder = format_vehicle_id_folder(vehicle_id) if vehicle_id else "VehicleID-01"

    # Exact hierarchy: report / VehicleID-XX / report.pdf
    if root_folder:
        remote_path = f"{root_folder}/{vehicle_folder}/{safe_remote_filename}"
    else:
        remote_path = f"report/{vehicle_folder}/{safe_remote_filename}"

    clean_remote_path = urllib.parse.quote(remote_path.lstrip("/"), safe="/")
    upload_url = f"{url}/storage/v1/object/{bucket}/{clean_remote_path}"

    try:
        with open(local_pdf_path, "rb") as f:
            pdf_bytes = f.read()

        headers = {
            "apiKey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/pdf",
            "x-upsert": "true"
        }

        req = urllib.request.Request(upload_url, data=pdf_bytes, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            logger.info(f"🚀 Uploaded forensic PDF report to Supabase: {bucket}/{remote_path} ({len(pdf_bytes)} bytes)")
            
            # Public/Authenticated access path
            cloud_url = f"{url}/storage/v1/object/public/{bucket}/{clean_remote_path}"
            
            return {
                "success": True,
                "bucket": bucket,
                "remote_path": remote_path,
                "cloud_url": cloud_url,
                "supabase_response": res_data
            }
    except Exception as e:
        logger.error(f"Failed to upload report '{local_pdf_path}' to Supabase: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def list_cloud_reports(config=None):
    """List uploaded PDF forensic reports in target Supabase Storage."""
    if config is None:
        config = load_config()

    url = (config.get("report_supabase_url") or config.get("supabase_url") or "").rstrip("/")
    key = config.get("report_supabase_key") or config.get("supabase_key") or ""
    bucket = config.get("report_bucket_name") or "evidentia"
    folder = config.get("report_folder_prefix", "report").strip("/")

    if not url or not key:
        return {"success": False, "error": "Report storage not configured", "reports": []}

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    list_url = f"{url}/storage/v1/object/list/{bucket}"
    payload = json.dumps({
        "prefix": folder,
        "limit": 100,
        "sortBy": {"column": "created_at", "order": "desc"}
    }).encode("utf-8")

    req = urllib.request.Request(list_url, data=payload, headers=_get_headers(key), method="POST")

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            items = json.loads(response.read().decode("utf-8"))
            reports = []
            if isinstance(items, list):
                for item in items:
                    name = item.get("name", "")
                    if name.endswith(".pdf"):
                        reports.append({
                            "name": name,
                            "remote_path": f"{folder}/{name}" if folder else name,
                            "size": item.get("metadata", {}).get("size", 0),
                            "created_at": item.get("created_at")
                        })
            return {"success": True, "bucket": bucket, "reports": reports}
    except Exception as e:
        logger.error(f"Error listing cloud reports: {e}")
        return {"success": False, "error": str(e), "reports": []}


def format_bytes(size):
    """Format bytes into readable string."""
    try:
        size = float(size)
    except (ValueError, TypeError):
        return "0 B"
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TB"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evidentia / DriveOps Supabase Cloud Evidence Fetcher")
    parser.add_argument("--test", action="store_true", help="Test Supabase connection")
    parser.add_argument("--list", action="store_true", help="List cloud evidence telemetry files")
    parser.add_argument("--download", type=str, help="Remote path to download (e.g. accidents/accident_20260828_233236.xlsx)")
    parser.add_argument("--set-url", type=str, help="Configure Supabase URL")
    parser.add_argument("--set-key", type=str, help="Configure Supabase API Key")
    parser.add_argument("--set-bucket", type=str, help="Configure Supabase Bucket Name (default: evidentia-evidence)")
    
    args = parser.parse_args()
    
    updates = {}
    if args.set_url:
        updates["supabase_url"] = args.set_url
    if args.set_key:
        updates["supabase_key"] = args.set_key
    if args.set_bucket:
        updates["bucket_name"] = args.set_bucket
    if updates:
        save_config(updates)
        print("Updated configuration.")

    if args.test:
        res = test_supabase_connection()
        print("Connection Test:", json.dumps(res, indent=2))

    if args.list:
        files = list_cloud_evidence_files()
        print("Cloud Files:", json.dumps(files, indent=2))

    if args.download:
        dest = download_cloud_file(args.download)
        print(f"Downloaded file to: {dest}")
