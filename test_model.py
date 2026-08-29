import argparse
import copy
import json
import os
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

try:
    from scipy.signal import butter, filtfilt
except ImportError:  # pragma: no cover - fallback only used if scipy is missing
    butter = None
    filtfilt = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

DEFAULT_WINDOW_SIZE = 50
DEFAULT_STEP_SIZE = 25
DEFAULT_HEURISTIC_CONFIG_PATH = os.path.join(MODELS_DIR, "heuristic_config.json")

CAUSE_LABEL_ORDER = [
    "normal_driving",
    "rash_driving",
    "hard_braking",
    "possible_brake_failure",
    "sharp_turning_or_skid",
    "possible_collision_impact",
]

RAW_COLUMN_ALIASES = {
    "accx": "AccX",
    "acc_x": "AccX",
    "accelx": "AccX",
    "accelerationx": "AccX",
    "useraccelerationx": "AccX",
    "xaxisg": "AccX",
    "accy": "AccY",
    "acc_y": "AccY",
    "accely": "AccY",
    "accelerationy": "AccY",
    "useraccelerationy": "AccY",
    "yaxisg": "AccY",
    "accz": "AccZ",
    "acc_z": "AccZ",
    "accelz": "AccZ",
    "accelerationz": "AccZ",
    "useraccelerationz": "AccZ",
    "zaxisg": "AccZ",
    "gyrox": "GyroX",
    "gyro_x": "GyroX",
    "rotationratex": "GyroX",
    "xaxisdegs": "GyroX",
    "gyroy": "GyroY",
    "gyro_y": "GyroY",
    "rotationratey": "GyroY",
    "yaxisdegs": "GyroY",
    "gyroz": "GyroZ",
    "gyro_z": "GyroZ",
    "rotationratez": "GyroZ",
    "zaxisdegs": "GyroZ",
    "speed": "Speed",
    "vehiclespeed": "Speed",
    "speedmps": "Speed",
    "speed_mps": "Speed",
    "throttle": "ThrottlePct",
    "throttlepct": "ThrottlePct",
    "acceleratorpedalpositione": "ThrottlePct",
    "brake": "BrakePct",
    "brakepct": "BrakePct",
    "brakepedal": "BrakePct",
    "brakeposition": "BrakePct",
    "brakepressure": "BrakePct",
    "enginetemperature": "EngineTemp",
    "engine_temperature": "EngineTemp",
    "enginetemp": "EngineTemp",
    "oiltemperature": "OilTemp",
    "oil_temperature": "OilTemp",
    "oiltemp": "OilTemp",
    "oilpressure": "OilPressure",
    "oil_pressure": "OilPressure",
    "rpm": "RPM",
    "label": "label",
    "class": "label",
    "target": "label",
    "activity": "label",
    "timestamp": "Timestamp",
    "time": "Timestamp",
    "cause": "cause_label",
    "causelabel": "cause_label",
    "eventlabel": "cause_label",
    "eventcause": "cause_label",
    "incidentlabel": "cause_label",
    "accidentcause": "cause_label",
}

CAUSE_VALUE_ALIASES = {
    "normal": "normal_driving",
    "normaldriving": "normal_driving",
    "safe": "normal_driving",
    "rashdriving": "rash_driving",
    "aggressive": "rash_driving",
    "aggressivedriving": "rash_driving",
    "hardbraking": "hard_braking",
    "suddenbraking": "hard_braking",
    "brakefailure": "possible_brake_failure",
    "possiblebrakefailure": "possible_brake_failure",
    "sharpturn": "sharp_turning_or_skid",
    "sharpturning": "sharp_turning_or_skid",
    "sharpturningorskid": "sharp_turning_or_skid",
    "skid": "sharp_turning_or_skid",
    "skidding": "sharp_turning_or_skid",
    "overspeeding": "rash_driving",
    "overspeed": "rash_driving",
    "speeding": "rash_driving",
    "rash": "rash_driving",
    "collision": "possible_collision_impact",
    "impact": "possible_collision_impact",
    "crash": "possible_collision_impact",
    "possiblecollisionimpact": "possible_collision_impact",
    "overheating": "engine_overheating",
    "engineheating": "engine_overheating",
    "engineoverheating": "engine_overheating",
}

CAUSE_DISPLAY_NAMES = {
    "normal_driving": "Normal / Safe Driving",
    "rash_driving": "Rash Driving / Overspeeding",
    "hard_braking": "Hard / Emergency Braking",
    "possible_brake_failure": "Possible Brake Failure",
    "sharp_turning_or_skid": "Sharp Turning / Vehicle Skid",
    "possible_collision_impact": "Collision Impact",
    "engine_overheating": "Engine Overheating / Thermal Failure",
}

METRIC_LABELS = {
    "acc_mag_peak": "peak acceleration",
    "acc_mag_mean": "average acceleration",
    "jerk_peak": "peak jerk",
    "gyro_mag_peak": "peak rotation",
    "engine_temp_peak": "peak engine temperature",
    "oil_temp_peak": "peak oil temperature",
    "horizontal_acc_peak": "horizontal acceleration",
    "vertical_acc_peak": "vertical acceleration",
    "speed_drop": "speed drop",
    "speed_gain": "speed gain",
    "speed_mean": "speed level",
    "speed_decel_sustained_ratio": "sustained deceleration",
    "brake_peak": "peak brake input",
    "brake_sustained_ratio": "sustained brake input",
    "throttle_peak": "peak throttle",
    "low_throttle_ratio": "low throttle ratio",
    "high_throttle_ratio": "high throttle ratio",
    "yaw_peak": "peak yaw rate",
    "yaw_sustained_ratio": "sustained yaw",
    "lateral_acc_peak": "lateral acceleration",
    "longitudinal_acc_min": "minimum longitudinal acceleration",
    "high_acc_duration_ratio": "time spent at high acceleration",
}


def default_heuristic_config():
    return {
        "sampling_rate_hz": 50.0,
        "preprocessing": {
            "sort_by_timestamp": True,
            "apply_lowpass": True,
            "sensor_lowpass_hz": 8.0,
            "gravity_lowpass_hz": 0.35,
            "filter_order": 4,
            "remove_gravity": True,
            "normalize_orientation": True,
        },
        "duration_thresholds": {
            "high_acc_mag": 3.2,
            "high_brake_pct": 35.0,
            "low_throttle_pct": 12.0,
            "high_throttle_pct": 60.0,
            "high_yaw_rate": 0.35,
            "speed_decel_per_sample": 0.05,
            "speed_accel_per_sample": 0.05,
        },
        "smoothing": {
            "min_consecutive_windows": 2,
            "majority_window": 3,
            "single_window_causes": ["possible_collision_impact"],
            "protected_causes": ["possible_collision_impact"],
            "protect_score_threshold": 0.72,
        },
        "calibration": {
            "positive_low_quantile": 0.25,
            "positive_high_quantile": 0.75,
            "negative_high_quantile": 0.95,
            "negative_low_quantile": 0.05,
        },
        "scoring": {
            "default_cause": "normal_driving",
            "min_score_to_emit": 0.48,
            "causes": {
                "possible_collision_impact": {
                    "base_reason": "impact-like motion spike detected",
                    "metrics": {
                        "acc_mag_peak": {"direction": "high", "low": 3.8, "high": 6.0, "weight": 0.35},
                        "jerk_peak": {"direction": "high", "low": 2.0, "high": 4.0, "weight": 0.30},
                        "horizontal_acc_peak": {"direction": "high", "low": 2.0, "high": 4.2, "weight": 0.20},
                        "speed_drop": {"direction": "high", "low": 1.0, "high": 6.0, "weight": 0.15},
                    },
                },
                "possible_brake_failure": {
                    "base_reason": "strong braking input did not create the expected slowdown",
                    "required_signals": ["has_speed_signal", "has_brake_signal"],
                    "metrics": {
                        "brake_peak": {"direction": "high", "low": 35.0, "high": 80.0, "weight": 0.20},
                        "brake_sustained_ratio": {"direction": "high", "low": 0.20, "high": 0.70, "weight": 0.25},
                        "low_throttle_ratio": {"direction": "high", "low": 0.35, "high": 0.90, "weight": 0.15},
                        "speed_drop": {"direction": "low", "low": 0.20, "high": 3.20, "weight": 0.20},
                        "speed_decel_sustained_ratio": {"direction": "low", "low": 0.08, "high": 0.40, "weight": 0.10},
                        "speed_mean": {"direction": "high", "low": 4.0, "high": 15.0, "weight": 0.10},
                    },
                },
                "hard_braking": {
                    "base_reason": "strong deceleration pattern detected",
                    "metrics": {
                        "speed_drop": {"direction": "high", "low": 2.0, "high": 8.0, "weight": 0.25},
                        "brake_peak": {"direction": "high", "low": 20.0, "high": 70.0, "weight": 0.15},
                        "brake_sustained_ratio": {"direction": "high", "low": 0.12, "high": 0.60, "weight": 0.15},
                        "longitudinal_acc_min": {"direction": "low", "low": -4.0, "high": -0.8, "weight": 0.15},
                        "acc_mag_peak": {"direction": "high", "low": 2.5, "high": 4.5, "weight": 0.15},
                        "jerk_peak": {"direction": "high", "low": 1.3, "high": 2.8, "weight": 0.15},
                    },
                },
                "sharp_turning_or_skid": {
                    "base_reason": "yaw and lateral motion suggest a sharp turn or skid",
                    "metrics": {
                        "yaw_peak": {"direction": "high", "low": 0.18, "high": 0.65, "weight": 0.25},
                        "yaw_sustained_ratio": {"direction": "high", "low": 0.12, "high": 0.45, "weight": 0.20},
                        "horizontal_acc_peak": {"direction": "high", "low": 1.0, "high": 2.8, "weight": 0.25},
                        "lateral_acc_peak": {"direction": "high", "low": 0.8, "high": 2.0, "weight": 0.15},
                        "gyro_mag_peak": {"direction": "high", "low": 0.25, "high": 0.80, "weight": 0.10},
                        "speed_mean": {"direction": "high", "low": 4.0, "high": 15.0, "weight": 0.05},
                    },
                },
                "rash_driving": {
                    "base_reason": "combined motion intensity matches rash driving",
                    "behavior_bonus": {"AGGRESSIVE": 0.12},
                    "metrics": {
                        "acc_mag_peak": {"direction": "high", "low": 2.5, "high": 4.5, "weight": 0.18},
                        "jerk_peak": {"direction": "high", "low": 1.2, "high": 2.5, "weight": 0.16},
                        "gyro_mag_peak": {"direction": "high", "low": 0.20, "high": 0.70, "weight": 0.14},
                        "throttle_peak": {"direction": "high", "low": 35.0, "high": 75.0, "weight": 0.16},
                        "speed_gain": {"direction": "high", "low": 2.0, "high": 8.0, "weight": 0.12},
                        "high_acc_duration_ratio": {"direction": "high", "low": 0.08, "high": 0.35, "weight": 0.12},
                        "yaw_sustained_ratio": {"direction": "high", "low": 0.08, "high": 0.30, "weight": 0.12},
                    },
                },
                "engine_overheating": {
                    "base_reason": "critical engine thermal elevation / coolant temperature exceeded safe threshold (>105°C)",
                    "metrics": {
                        "engine_temp_peak": {"direction": "high", "low": 102.0, "high": 115.0, "weight": 0.50},
                        "high_engine_temp_ratio": {"direction": "high", "low": 0.15, "high": 0.70, "weight": 0.30},
                        "oil_temp_peak": {"direction": "high", "low": 110.0, "high": 130.0, "weight": 0.20},
                    },
                },
            },
        },
    }


def clone_default_config():
    return copy.deepcopy(default_heuristic_config())


def deep_merge_dicts(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def save_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def ensure_default_config_file(path):
    if not os.path.exists(path):
        save_json(path, clone_default_config())


def load_heuristic_config(path):
    ensure_default_config_file(path)
    with open(path, "r", encoding="utf-8") as handle:
        override = json.load(handle)
    return deep_merge_dicts(clone_default_config(), override)


def sanitize_column_name(name):
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def normalize_sensor_columns(df):
    rename_map = {}
    reserved_targets = set(df.columns)

    for col in df.columns:
        target = RAW_COLUMN_ALIASES.get(sanitize_column_name(col))
        if not target or col == target or target in reserved_targets:
            continue
        rename_map[col] = target
        reserved_targets.add(target)

    return df.rename(columns=rename_map)


def to_numeric_series(df, col):
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def canonicalize_cause_label(value):
    if pd.isna(value):
        return None

    text_value = sanitize_column_name(value)
    if not text_value:
        return None
    return CAUSE_VALUE_ALIASES.get(text_value, str(value).strip())


def lowpass_filter_array(values, cutoff_hz, sampling_rate_hz, order):
    arr = np.asarray(values, dtype=float)
    if len(arr) < 5:
        return arr

    if butter is None or filtfilt is None:
        window = max(3, min(len(arr) // 3, 9))
        if window % 2 == 0:
            window += 1
        if window <= 1:
            return arr
        kernel = np.ones(window, dtype=float) / float(window)
        return np.convolve(arr, kernel, mode="same")

    nyquist = 0.5 * float(sampling_rate_hz)
    if nyquist <= 0 or cutoff_hz <= 0:
        return arr
    normalized_cutoff = min(0.99, float(cutoff_hz) / nyquist)
    b, a = butter(int(order), normalized_cutoff, btype="low", analog=False)
    min_required = 3 * max(len(a), len(b))
    if len(arr) < min_required:
        return arr
    return filtfilt(b, a, arr)


def rename_requested_cause_column(df, cause_label_col):
    if not cause_label_col:
        return df
    if cause_label_col not in df.columns:
        raise ValueError(f"Cause label column '{cause_label_col}' was not found in the input CSV.")
    if cause_label_col == "cause_label":
        return df
    return df.rename(columns={cause_label_col: "cause_label"})


def preprocess_raw_sensor_df(df, config):
    raw_df = normalize_sensor_columns(df.copy())
    preprocessing_cfg = config.get("preprocessing", {})
    sampling_rate_hz = float(config.get("sampling_rate_hz", 50.0))

    if preprocessing_cfg.get("sort_by_timestamp", True) and "Timestamp" in raw_df.columns:
        raw_df = raw_df.sort_values("Timestamp", kind="stable").reset_index(drop=True)

    numeric_cols = [
        "AccX",
        "AccY",
        "AccZ",
        "GyroX",
        "GyroY",
        "GyroZ",
        "Speed",
        "ThrottlePct",
        "BrakePct",
        "Timestamp",
    ]
    for col in numeric_cols:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    if "cause_label" in raw_df.columns:
        raw_df["cause_label"] = raw_df["cause_label"].apply(canonicalize_cause_label)

    filter_order = int(preprocessing_cfg.get("filter_order", 4))
    sensor_cutoff = float(preprocessing_cfg.get("sensor_lowpass_hz", 8.0))
    gravity_cutoff = float(preprocessing_cfg.get("gravity_lowpass_hz", 0.35))
    apply_lowpass = bool(preprocessing_cfg.get("apply_lowpass", True))

    sensor_axes = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
    for col in sensor_axes:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce").fillna(0.0)
            if apply_lowpass:
                raw_df[col] = lowpass_filter_array(
                    raw_df[col].to_numpy(dtype=float),
                    cutoff_hz=sensor_cutoff,
                    sampling_rate_hz=sampling_rate_hz,
                    order=filter_order,
                )

    for col in ["Speed", "ThrottlePct", "BrakePct"]:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce").fillna(0.0)

    if all(col in raw_df.columns for col in ["AccX", "AccY", "AccZ"]):
        filtered_acc = raw_df[["AccX", "AccY", "AccZ"]].to_numpy(dtype=float)
        gravity = filtered_acc.copy()

        if preprocessing_cfg.get("remove_gravity", True):
            gravity_components = []
            for axis_index in range(3):
                gravity_components.append(
                    lowpass_filter_array(
                        filtered_acc[:, axis_index],
                        cutoff_hz=gravity_cutoff,
                        sampling_rate_hz=sampling_rate_hz,
                        order=filter_order,
                    )
                )
            gravity = np.column_stack(gravity_components)
            linear_acc = filtered_acc - gravity
        else:
            linear_acc = filtered_acc

        raw_df["GravityX"] = gravity[:, 0]
        raw_df["GravityY"] = gravity[:, 1]
        raw_df["GravityZ"] = gravity[:, 2]
        raw_df["LinAccX"] = linear_acc[:, 0]
        raw_df["LinAccY"] = linear_acc[:, 1]
        raw_df["LinAccZ"] = linear_acc[:, 2]
        raw_df["LinAccMag"] = np.linalg.norm(linear_acc, axis=1)
        raw_df["AccMag"] = np.linalg.norm(filtered_acc, axis=1)

        if preprocessing_cfg.get("normalize_orientation", True):
            gravity_norm = np.linalg.norm(gravity, axis=1)
            gravity_unit = np.divide(
                gravity,
                gravity_norm[:, None],
                out=np.zeros_like(gravity),
                where=gravity_norm[:, None] > 1e-8,
            )
            vertical_acc = np.sum(linear_acc * gravity_unit, axis=1)
            horizontal_acc = linear_acc - vertical_acc[:, None] * gravity_unit
            raw_df["VerticalAccAbs"] = np.abs(vertical_acc)
            raw_df["HorizontalAccMag"] = np.linalg.norm(horizontal_acc, axis=1)
        else:
            raw_df["VerticalAccAbs"] = np.abs(linear_acc[:, 2])
            raw_df["HorizontalAccMag"] = np.sqrt(linear_acc[:, 0] ** 2 + linear_acc[:, 1] ** 2)

    if all(col in raw_df.columns for col in ["GyroX", "GyroY", "GyroZ"]):
        gyro_values = raw_df[["GyroX", "GyroY", "GyroZ"]].to_numpy(dtype=float)
        raw_df["GyroMag"] = np.linalg.norm(gyro_values, axis=1)

    return raw_df


def extract_window_features(window):
    feat = {}
    sensor_cols = [
        col
        for col in [
            "AccX",
            "AccY",
            "AccZ",
            "GyroX",
            "GyroY",
            "GyroZ",
            "AccMag",
            "GyroMag",
            "Speed",
            "ThrottlePct",
            "BrakePct",
        ]
        if col in window.columns
    ]

    for col in sensor_cols:
        vals = pd.to_numeric(window[col], errors="coerce").fillna(0.0).values.astype(float)
        feat[f"{col}_mean"] = float(np.mean(vals))
        feat[f"{col}_std"] = float(np.std(vals))
        feat[f"{col}_max"] = float(np.max(vals))
        feat[f"{col}_min"] = float(np.min(vals))
        feat[f"{col}_rms"] = float(np.sqrt(np.mean(vals ** 2)))

    acc_series_name = "LinAccMag" if "LinAccMag" in window.columns else "AccMag"
    if acc_series_name in window.columns:
        acc = pd.to_numeric(window[acc_series_name], errors="coerce").fillna(0.0).values.astype(float)
        if len(acc) > 1:
            jerk = np.diff(acc)
            feat["jerk_mean"] = float(np.mean(np.abs(jerk)))
            feat["jerk_max"] = float(np.max(np.abs(jerk)))
            feat["jerk_std"] = float(np.std(jerk))
        else:
            feat["jerk_mean"] = 0.0
            feat["jerk_max"] = 0.0
            feat["jerk_std"] = 0.0
        feat["acc_energy"] = float(np.sum(acc ** 2))

    corr_acc_col = "LinAccX" if "LinAccX" in window.columns else "AccX"
    if corr_acc_col in window.columns and "GyroZ" in window.columns:
        acc_x = pd.to_numeric(window[corr_acc_col], errors="coerce").fillna(0.0).values.astype(float)
        gyro_z = pd.to_numeric(window["GyroZ"], errors="coerce").fillna(0.0).values.astype(float)
        try:
            corr = np.corrcoef(acc_x, gyro_z)[0, 1]
            feat["corr_accX_gyroZ"] = 0.0 if np.isnan(corr) else float(corr)
        except Exception:
            feat["corr_accX_gyroZ"] = 0.0

    return feat


def summarize_window(window, config):
    duration_cfg = config.get("duration_thresholds", {})
    summary = {
        "acc_mag_peak": np.nan,
        "acc_mag_mean": np.nan,
        "acc_mag_std": np.nan,
        "gyro_mag_peak": np.nan,
        "gyro_mag_mean": np.nan,
        "jerk_peak": np.nan,
        "jerk_mean": np.nan,
        "high_acc_duration_ratio": np.nan,
        "throttle_mean": np.nan,
        "throttle_peak": np.nan,
        "low_throttle_ratio": np.nan,
        "high_throttle_ratio": np.nan,
        "brake_mean": np.nan,
        "brake_peak": np.nan,
        "brake_sustained_ratio": np.nan,
        "speed_mean": np.nan,
        "speed_std": np.nan,
        "speed_drop": np.nan,
        "speed_gain": np.nan,
        "speed_decel_peak": np.nan,
        "speed_accel_peak": np.nan,
        "speed_decel_sustained_ratio": np.nan,
        "speed_accel_sustained_ratio": np.nan,
        "longitudinal_acc_min": np.nan,
        "lateral_acc_peak": np.nan,
        "horizontal_acc_peak": np.nan,
        "horizontal_acc_mean": np.nan,
        "vertical_acc_peak": np.nan,
        "yaw_peak": np.nan,
        "yaw_mean_abs": np.nan,
        "yaw_sustained_ratio": np.nan,
        "has_speed_signal": 0.0,
        "has_throttle_signal": 0.0,
        "has_brake_signal": 0.0,
        "window_rows": float(len(window)),
    }

    acc_col = "LinAccMag" if "LinAccMag" in window.columns else "AccMag"
    if acc_col in window.columns:
        acc = pd.to_numeric(window[acc_col], errors="coerce").fillna(0.0).values.astype(float)
        summary["acc_mag_peak"] = float(np.max(acc))
        summary["acc_mag_mean"] = float(np.mean(acc))
        summary["acc_mag_std"] = float(np.std(acc))
        if len(acc) > 1:
            jerk = np.abs(np.diff(acc))
            summary["jerk_peak"] = float(np.max(jerk))
            summary["jerk_mean"] = float(np.mean(jerk))
        else:
            summary["jerk_peak"] = 0.0
            summary["jerk_mean"] = 0.0
        summary["high_acc_duration_ratio"] = float(
            np.mean(acc >= float(duration_cfg.get("high_acc_mag", 3.2)))
        )

    if "GyroMag" in window.columns:
        gyro_mag = pd.to_numeric(window["GyroMag"], errors="coerce").fillna(0.0).values.astype(float)
        summary["gyro_mag_peak"] = float(np.max(gyro_mag))
        summary["gyro_mag_mean"] = float(np.mean(gyro_mag))

    if "ThrottlePct" in window.columns:
        throttle = pd.to_numeric(window["ThrottlePct"], errors="coerce").fillna(0.0).values.astype(float)
        summary["throttle_mean"] = float(np.mean(throttle))
        summary["throttle_peak"] = float(np.max(throttle))
        summary["low_throttle_ratio"] = float(
            np.mean(throttle <= float(duration_cfg.get("low_throttle_pct", 12.0)))
        )
        summary["high_throttle_ratio"] = float(
            np.mean(throttle >= float(duration_cfg.get("high_throttle_pct", 60.0)))
        )
        summary["has_throttle_signal"] = 1.0

    if "BrakePct" in window.columns:
        brake = pd.to_numeric(window["BrakePct"], errors="coerce").fillna(0.0).values.astype(float)
        summary["brake_mean"] = float(np.mean(brake))
        summary["brake_peak"] = float(np.max(brake))
        summary["brake_sustained_ratio"] = float(
            np.mean(brake >= float(duration_cfg.get("high_brake_pct", 35.0)))
        )
        summary["has_brake_signal"] = 1.0

    if "Speed" in window.columns:
        speed = pd.to_numeric(window["Speed"], errors="coerce").fillna(0.0).values.astype(float)
        diffs = np.diff(speed)
        neg_diffs = np.clip(-diffs, 0.0, None)
        pos_diffs = np.clip(diffs, 0.0, None)
        summary["speed_mean"] = float(np.mean(speed))
        summary["speed_std"] = float(np.std(speed))
        summary["speed_drop"] = float(max(0.0, speed[0] - speed[-1]))
        summary["speed_gain"] = float(max(0.0, speed[-1] - speed[0]))
        summary["speed_decel_peak"] = float(np.max(neg_diffs)) if len(neg_diffs) else 0.0
        summary["speed_accel_peak"] = float(np.max(pos_diffs)) if len(pos_diffs) else 0.0
        summary["speed_decel_sustained_ratio"] = float(
            np.mean(diffs <= -float(duration_cfg.get("speed_decel_per_sample", 0.05)))
        ) if len(diffs) else 0.0
        summary["speed_accel_sustained_ratio"] = float(
            np.mean(diffs >= float(duration_cfg.get("speed_accel_per_sample", 0.05)))
        ) if len(diffs) else 0.0
        summary["has_speed_signal"] = 1.0

    longitudinal_col = "LinAccX" if "LinAccX" in window.columns else "AccX"
    if longitudinal_col in window.columns:
        longitudinal = pd.to_numeric(window[longitudinal_col], errors="coerce").fillna(0.0).values.astype(float)
        summary["longitudinal_acc_min"] = float(np.min(longitudinal))

    lateral_col = "LinAccY" if "LinAccY" in window.columns else "AccY"
    if lateral_col in window.columns:
        lateral = pd.to_numeric(window[lateral_col], errors="coerce").fillna(0.0).values.astype(float)
        summary["lateral_acc_peak"] = float(np.max(np.abs(lateral)))

    if "HorizontalAccMag" in window.columns:
        horizontal_acc = pd.to_numeric(window["HorizontalAccMag"], errors="coerce").fillna(0.0).values.astype(float)
        summary["horizontal_acc_peak"] = float(np.max(horizontal_acc))
        summary["horizontal_acc_mean"] = float(np.mean(horizontal_acc))

    if "VerticalAccAbs" in window.columns:
        vertical_acc = pd.to_numeric(window["VerticalAccAbs"], errors="coerce").fillna(0.0).values.astype(float)
        summary["vertical_acc_peak"] = float(np.max(vertical_acc))

    if "GyroZ" in window.columns:
        yaw = pd.to_numeric(window["GyroZ"], errors="coerce").fillna(0.0).values.astype(float)
        summary["yaw_peak"] = float(np.max(np.abs(yaw)))
        summary["yaw_mean_abs"] = float(np.mean(np.abs(yaw)))
        summary["yaw_sustained_ratio"] = float(
            np.mean(np.abs(yaw) >= float(duration_cfg.get("high_yaw_rate", 0.35)))
        )

    if "EngineTemp" in window.columns:
        eng_temp = pd.to_numeric(window["EngineTemp"], errors="coerce").dropna().values.astype(float)
        if len(eng_temp):
            summary["engine_temp_peak"] = float(np.max(eng_temp))
            summary["engine_temp_mean"] = float(np.mean(eng_temp))
            summary["high_engine_temp_ratio"] = float(np.mean(eng_temp >= 105.0))
            summary["has_engine_temp_signal"] = 1.0

    if "OilTemp" in window.columns:
        oil_temp = pd.to_numeric(window["OilTemp"], errors="coerce").dropna().values.astype(float)
        if len(oil_temp):
            summary["oil_temp_peak"] = float(np.max(oil_temp))
            summary["oil_temp_mean"] = float(np.mean(oil_temp))
            summary["high_oil_temp_ratio"] = float(np.mean(oil_temp >= 120.0))

    return summary


def aggregate_window_label(window, column_name, normalizer=None):
    if column_name not in window.columns:
        return None
    values = window[column_name].dropna()
    if values.empty:
        return None
    if normalizer:
        normalized = [normalizer(value) for value in values]
        normalized = [value for value in normalized if value is not None]
        if not normalized:
            return None
        return pd.Series(normalized).mode().iloc[0]
    return values.mode().iloc[0]


def build_features_from_raw(df, window_size, step_size, config):
    rows = []
    meta = []

    if len(df) == 0:
        raise ValueError("Raw input CSV has no rows.")

    if len(df) < window_size:
        windows = [(0, len(df))]
    else:
        windows = [
            (start, start + window_size)
            for start in range(0, len(df) - window_size + 1, step_size)
        ]

    for window_index, (start, end) in enumerate(windows):
        window = df.iloc[start:end]
        rows.append(extract_window_features(window))

        meta_row = {
            "window_index": window_index,
            "start_row": start,
            "end_row": end - 1,
        }
        meta_row.update(summarize_window(window, config))

        actual_behavior = aggregate_window_label(window, "label")
        if actual_behavior is not None:
            meta_row["actual_window_label_raw"] = actual_behavior

        actual_cause = aggregate_window_label(window, "cause_label", canonicalize_cause_label)
        if actual_cause is not None:
            meta_row["actual_window_cause_raw"] = actual_cause

        meta.append(meta_row)

    return pd.DataFrame(rows), meta


def align_features(feature_df, feature_names):
    aligned = feature_df.copy()
    for col in feature_names:
        if col not in aligned.columns:
            aligned[col] = 0.0
    return aligned[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def default_output_path(input_path):
    root, ext = os.path.splitext(input_path)
    return f"{root}_predictions{ext or '.csv'}"


STANDARD_SENSOR_COLUMNS = [
    "AccX", "AccY", "AccZ",
    "GyroX", "GyroY", "GyroZ",
    "Speed", "ThrottlePct", "BrakePct",
    "EngineTemp", "OilTemp", "RPM",
]


def read_input_file(path):
    """
    Auto-detect file format and return telemetry DataFrame.
    For multi-sheet Excel files, prioritizes continuous sensor sheets (Combined Telemetry, OutGauge, MotionSim).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        try:
            xl = pd.ExcelFile(path)
            # Check for continuous sensor telemetry sheet
            for preferred in ["Combined Telemetry", "combined_telemetry", "Telemetry", "telemetry", "OutGauge", "MotionSim"]:
                if preferred in xl.sheet_names:
                    df = xl.parse(preferred)
                    if len(df) > 0:
                        return df
            return xl.parse(xl.sheet_names[0])
        except Exception:
            return pd.read_excel(path)
    elif ext == ".json":
        return pd.read_json(path)
    else:
        return pd.read_csv(path)


def save_output_file(df, path, fmt=None):
    """Save DataFrame in the requested format. Auto-detects from extension if fmt is None."""
    if fmt is None:
        fmt = os.path.splitext(path)[1].lower().lstrip(".")
    fmt = fmt.lower()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fmt in ("xlsx", "xls"):
        df.to_excel(path, index=False)
    elif fmt == "json":
        df.to_json(path, orient="records", indent=2, force_ascii=False)
    else:
        df.to_csv(path, index=False)


def standardize_dataframe(df, column_mapping=None):
    """Normalize column names and ensure all standard sensor columns exist (fill missing with 0)."""
    result = normalize_sensor_columns(df.copy())
    if column_mapping:
        result = result.rename(columns=column_mapping)
    for col in STANDARD_SENSOR_COLUMNS:
        if col not in result.columns:
            result[col] = 0.0
    return result


def predict_single_entry(sensor_values, config_path=DEFAULT_HEURISTIC_CONFIG_PATH, assets=None):
    """Predict from a single dict of sensor values. Missing fields default to 0."""
    assets = assets or load_prediction_assets(config_path)
    config = assets["config"]
    model = assets["model"]
    scaler = assets["scaler"]
    feature_names = assets["feature_names"]
    label_map = assets["label_map"]

    row = {col: 0.0 for col in STANDARD_SENSOR_COLUMNS}
    for key, value in sensor_values.items():
        canonical = RAW_COLUMN_ALIASES.get(sanitize_column_name(key), key)
        try:
            row[canonical] = float(value)
        except (ValueError, TypeError):
            row[canonical] = 0.0

    source_df = pd.DataFrame([row])
    source_df = preprocess_raw_sensor_df(source_df, config)

    window_size = min(len(source_df), DEFAULT_WINDOW_SIZE)
    feature_df, meta = build_features_from_raw(source_df, window_size, window_size, config)
    meta_df = pd.DataFrame(meta)

    X = align_features(feature_df, feature_names)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    results = meta_df.copy()
    results["prediction_id"] = preds.astype(int)
    results["prediction_label"] = [label_map.get(int(p), p) for p in preds]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)
        add_probability_columns(results, probs, model.classes_, label_map)

    results = predict_causes(results, config)
    results = smooth_cause_labels(results, config)

    summary = summarize_prediction_results(results, "raw")
    return {
        "results": results,
        "summary": summary,
        "label_map": label_map,
        "config_path": assets["config_path"],
    }


def default_calibration_output_path():
    return os.path.join(MODELS_DIR, "heuristic_config.calibrated.json")


def default_calibration_report_path():
    return os.path.join(DATA_DIR, "heuristic_calibration_report.csv")


def add_probability_columns(results, probs, classes_, label_map):
    for idx, class_id in enumerate(classes_):
        label = str(label_map.get(int(class_id), class_id))
        results[f"prob_{label}"] = probs[:, idx]
    results["confidence"] = probs.max(axis=1)


def metric_score(value, metric_cfg):
    if pd.isna(value):
        return None

    low = float(metric_cfg["low"])
    high = float(metric_cfg["high"])
    direction = metric_cfg["direction"]

    if np.isclose(high, low):
        if direction == "high":
            return 1.0 if value >= high else 0.0
        return 1.0 if value <= low else 0.0

    if direction == "high":
        return float(np.clip((value - low) / (high - low), 0.0, 1.0))
    if direction == "low":
        return float(np.clip((high - value) / (high - low), 0.0, 1.0))

    raise ValueError(f"Unsupported metric direction: {direction}")


def build_cause_reason(base_reason, contributions):
    strong_metrics = [
        METRIC_LABELS.get(metric_name, metric_name)
        for metric_name, metric_score_value, _weight, _raw_value in contributions
        if metric_score_value >= 0.45
    ]
    if strong_metrics:
        metric_text = ", ".join(strong_metrics[:2])
        return f"{base_reason}; strongest signals: {metric_text}"
    return base_reason


def score_single_cause(summary, behavior_label, cause_name, cause_cfg):
    required_signals = cause_cfg.get("required_signals", [])
    missing_signals = [
        signal_name
        for signal_name in required_signals
        if float(summary.get(signal_name, 0.0) or 0.0) <= 0.0
    ]
    if missing_signals:
        return {
            "score": 0.0,
            "reason": f"{cause_name} needs signals: {', '.join(missing_signals)}",
            "contributions": [],
        }

    metrics_cfg = cause_cfg.get("metrics", {})
    total_metric_weight = sum(float(metric_cfg["weight"]) for metric_cfg in metrics_cfg.values())
    available_weight = 0.0
    weighted_sum = 0.0
    contributions = []

    for metric_name, metric_cfg in metrics_cfg.items():
        metric_value = summary.get(metric_name, np.nan)
        metric_score_value = metric_score(metric_value, metric_cfg)
        if metric_score_value is None:
            continue
        weight = float(metric_cfg["weight"])
        available_weight += weight
        weighted_sum += metric_score_value * weight
        contributions.append((metric_name, metric_score_value, weight, metric_value))

    if available_weight == 0.0 or total_metric_weight == 0.0:
        return {
            "score": 0.0,
            "reason": cause_cfg.get("base_reason", cause_name),
            "contributions": [],
        }

    coverage = available_weight / total_metric_weight
    core_score = weighted_sum / available_weight
    behavior_bonus = float(cause_cfg.get("behavior_bonus", {}).get(str(behavior_label), 0.0))
    final_score = min(1.0, core_score * coverage + behavior_bonus)
    ordered_contributions = sorted(contributions, key=lambda item: item[1] * item[2], reverse=True)

    return {
        "score": float(final_score),
        "reason": build_cause_reason(cause_cfg.get("base_reason", cause_name), ordered_contributions),
        "contributions": ordered_contributions,
    }


def predict_causes(results, config):
    scoring_cfg = config.get("scoring", {})
    default_cause = scoring_cfg.get("default_cause", "normal_driving")
    min_score_to_emit = float(scoring_cfg.get("min_score_to_emit", 0.48))
    cause_cfgs = scoring_cfg.get("causes", {})
    cause_names = list(cause_cfgs.keys())

    score_columns = {f"score_{cause_name}": [] for cause_name in cause_names}
    raw_labels = []
    raw_scores = []
    raw_reasons = []

    for _, row in results.iterrows():
        row_dict = row.to_dict()
        behavior_label = row_dict.get("prediction_label")
        scored_causes = {}

        for cause_name, cause_cfg in cause_cfgs.items():
            cause_result = score_single_cause(row_dict, behavior_label, cause_name, cause_cfg)
            scored_causes[cause_name] = cause_result
            score_columns[f"score_{cause_name}"].append(cause_result["score"])

        best_cause, best_result = max(
            scored_causes.items(),
            key=lambda item: (item[1]["score"], -CAUSE_LABEL_ORDER.index(item[0]) if item[0] in CAUSE_LABEL_ORDER else 0),
        )
        best_score = float(best_result["score"])

        if best_score >= min_score_to_emit:
            raw_labels.append(best_cause)
            raw_scores.append(best_score)
            raw_reasons.append(best_result["reason"])
        else:
            raw_labels.append(default_cause)
            raw_scores.append(best_score)
            raw_reasons.append("no non-normal cause score crossed the configured threshold")

    scored_results = results.copy()
    for column_name, values in score_columns.items():
        scored_results[column_name] = values
    scored_results["raw_predicted_cause"] = raw_labels
    scored_results["raw_cause_score"] = raw_scores
    scored_results["raw_cause_reason"] = raw_reasons
    return scored_results


def smooth_cause_labels(results, config):
    smoothing_cfg = config.get("smoothing", {})
    scoring_cfg = config.get("scoring", {})
    default_cause = scoring_cfg.get("default_cause", "normal_driving")
    min_run = int(smoothing_cfg.get("min_consecutive_windows", 2))
    majority_window = int(smoothing_cfg.get("majority_window", 3))
    single_window_causes = set(smoothing_cfg.get("single_window_causes", []))
    protected_causes = set(smoothing_cfg.get("protected_causes", []))
    protect_threshold = float(smoothing_cfg.get("protect_score_threshold", 0.72))

    labels = results["raw_predicted_cause"].tolist()
    scores = results["raw_cause_score"].astype(float).tolist()
    notes = ["" for _ in labels]

    if min_run > 1 and labels:
        run_filtered = labels[:]
        start = 0
        while start < len(run_filtered):
            end = start + 1
            while end < len(run_filtered) and run_filtered[end] == run_filtered[start]:
                end += 1
            run_label = run_filtered[start]
            run_length = end - start
            if (
                run_label != default_cause
                and run_label not in single_window_causes
                and run_length < min_run
            ):
                for idx in range(start, end):
                    run_filtered[idx] = default_cause
                    notes[idx] = f"suppressed short {run_label} run"
            start = end
        labels = run_filtered

    if majority_window > 1 and labels:
        half_window = majority_window // 2
        majority_filtered = labels[:]

        for idx in range(len(labels)):
            raw_label = results.iloc[idx]["raw_predicted_cause"]
            raw_score = float(results.iloc[idx]["raw_cause_score"])

            if raw_label in protected_causes and raw_score >= protect_threshold:
                continue

            start = max(0, idx - half_window)
            end = min(len(labels), idx + half_window + 1)
            label_weights = defaultdict(float)

            for neighbor_idx in range(start, end):
                neighbor_label = labels[neighbor_idx]
                neighbor_score = float(results.iloc[neighbor_idx]["raw_cause_score"])
                label_weights[neighbor_label] += max(neighbor_score, 0.01)

            best_label, _best_weight = max(label_weights.items(), key=lambda item: item[1])
            if best_label != labels[idx]:
                majority_filtered[idx] = best_label
                notes[idx] = notes[idx] or f"smoothed toward neighboring {best_label} windows"

        labels = majority_filtered

    final_labels = labels[:]
    final_reasons = []
    final_scores = []
    smoothing_notes = []

    for idx, final_label in enumerate(final_labels):
        raw_label = results.iloc[idx]["raw_predicted_cause"]
        raw_reason = results.iloc[idx]["raw_cause_reason"]
        note = notes[idx]

        if final_label == raw_label:
            final_reason = raw_reason
            smoothing_note = note or ""
        elif final_label == default_cause:
            final_reason = f"{raw_reason}; smoothed to {default_cause} because the signal was not sustained"
            smoothing_note = note or f"smoothed from {raw_label} to {default_cause}"
        else:
            final_reason = f"{raw_reason}; neighboring windows reinforced {final_label}"
            smoothing_note = note or f"smoothed from {raw_label} to {final_label}"

        if final_label == default_cause:
            max_non_normal = max(
                float(results.iloc[idx].get(f"score_{cause_name}", 0.0))
                for cause_name in config.get("scoring", {}).get("causes", {})
            )
            final_score = max(0.0, 1.0 - max_non_normal)
        else:
            final_score = float(results.iloc[idx].get(f"score_{final_label}", results.iloc[idx]["raw_cause_score"]))

        final_reasons.append(final_reason)
        final_scores.append(final_score)
        smoothing_notes.append(smoothing_note)

    smoothed_results = results.copy()
    smoothed_results["predicted_cause"] = final_labels
    smoothed_results["cause_reason"] = final_reasons
    smoothed_results["cause_confidence"] = final_scores
    smoothed_results["smoothing_note"] = smoothing_notes
    return smoothed_results


def coerce_behavior_label_value(value, label_map):
    if pd.isna(value):
        return pd.NA, None

    reverse_label_map = {
        str(label).strip().upper(): int(class_id)
        for class_id, label in label_map.items()
    }

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        class_id = int(numeric)
        return class_id, label_map.get(class_id, class_id)

    text_value = str(value).strip().upper()
    if text_value in reverse_label_map:
        class_id = reverse_label_map[text_value]
        return class_id, label_map.get(class_id, class_id)

    return pd.NA, str(value)


def evaluate_behavior_predictions(y_true, y_pred, label_map, title):
    y_true = pd.to_numeric(pd.Series(y_true), errors="coerce")
    valid_mask = ~y_true.isna()
    if not valid_mask.any():
        return

    y_true = y_true[valid_mask].astype(int)
    y_pred = np.asarray(y_pred)[valid_mask.to_numpy()]

    labels = sorted(np.unique(np.concatenate([y_true.to_numpy(), y_pred])))
    class_names = [str(label_map.get(int(label), label)) for label in labels]

    print(f"\n{title}")
    print("-" * len(title))
    print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        )
    )


def evaluate_cause_predictions(y_true, y_pred, title):
    true_series = pd.Series([canonicalize_cause_label(value) for value in y_true])
    valid_mask = true_series.notna()
    if not valid_mask.any():
        return

    y_true_clean = true_series[valid_mask].astype(str).tolist()
    y_pred_clean = pd.Series(y_pred)[valid_mask].astype(str).tolist()

    observed = list(dict.fromkeys(CAUSE_LABEL_ORDER + sorted(set(y_true_clean + y_pred_clean))))
    labels = [label for label in observed if label in set(y_true_clean + y_pred_clean)]

    accuracy = float(np.mean(np.asarray(y_true_clean) == np.asarray(y_pred_clean)))
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(
        classification_report(
            y_true_clean,
            y_pred_clean,
            labels=labels,
            zero_division=0,
        )
    )


def calibrate_config_from_windows(window_df, config):
    calibration_cfg = config.get("calibration", {})
    positive_low_q = float(calibration_cfg.get("positive_low_quantile", 0.25))
    positive_high_q = float(calibration_cfg.get("positive_high_quantile", 0.75))
    negative_high_q = float(calibration_cfg.get("negative_high_quantile", 0.95))
    negative_low_q = float(calibration_cfg.get("negative_low_quantile", 0.05))

    if "actual_window_cause_raw" not in window_df.columns:
        raise ValueError(
            "Calibration needs a row-level cause label column so each window can inherit a cause label."
        )

    labeled_windows = window_df.copy()
    labeled_windows["actual_window_cause_raw"] = labeled_windows["actual_window_cause_raw"].apply(
        canonicalize_cause_label
    )
    labeled_windows = labeled_windows[labeled_windows["actual_window_cause_raw"].notna()].copy()

    if labeled_windows.empty:
        raise ValueError("No usable cause labels were found after normalization.")

    calibrated_config = copy.deepcopy(config)
    report_rows = []

    for cause_name, cause_cfg in calibrated_config.get("scoring", {}).get("causes", {}).items():
        positive = labeled_windows[labeled_windows["actual_window_cause_raw"] == cause_name]
        negative = labeled_windows[labeled_windows["actual_window_cause_raw"] != cause_name]
        if positive.empty:
            continue

        for metric_name, metric_cfg in cause_cfg.get("metrics", {}).items():
            if metric_name not in positive.columns:
                continue

            positive_values = pd.to_numeric(positive[metric_name], errors="coerce").dropna()
            negative_values = pd.to_numeric(negative[metric_name], errors="coerce").dropna()
            if positive_values.empty:
                continue

            direction = metric_cfg["direction"]
            pos_low = float(positive_values.quantile(positive_low_q))
            pos_high = float(positive_values.quantile(positive_high_q))

            if direction == "high":
                neg_reference = (
                    float(negative_values.quantile(negative_high_q))
                    if not negative_values.empty
                    else pos_low
                )
                suggested_low = (neg_reference + pos_low) / 2.0
                suggested_high = max(pos_high, suggested_low + 1e-6)
            else:
                neg_reference = (
                    float(negative_values.quantile(negative_low_q))
                    if not negative_values.empty
                    else pos_high
                )
                suggested_low = pos_low
                suggested_high = max((neg_reference + pos_high) / 2.0, suggested_low + 1e-6)

            metric_cfg["low"] = float(suggested_low)
            metric_cfg["high"] = float(suggested_high)

            report_rows.append(
                {
                    "cause": cause_name,
                    "metric": metric_name,
                    "direction": direction,
                    "positive_samples": int(len(positive_values)),
                    "negative_samples": int(len(negative_values)),
                    "positive_q_low": pos_low,
                    "positive_q_high": pos_high,
                    "negative_reference": neg_reference,
                    "suggested_low": suggested_low,
                    "suggested_high": suggested_high,
                }
            )

    return calibrated_config, pd.DataFrame(report_rows)


def print_calibration_summary(window_df, report_df):
    cause_counts = (
        window_df["actual_window_cause_raw"]
        .dropna()
        .apply(canonicalize_cause_label)
        .value_counts()
        .sort_index()
    )

    print("\nCause windows available for calibration")
    print("--------------------------------------")
    print(cause_counts.to_string())

    if not report_df.empty:
        print("\nSuggested threshold updates")
        print("---------------------------")
        preview_cols = [
            "cause",
            "metric",
            "positive_q_low",
            "positive_q_high",
            "negative_reference",
            "suggested_low",
            "suggested_high",
        ]
        print(report_df[preview_cols].head(15).to_string(index=False))


def preview_results(results, preview_rows):
    preview_cols = [
        col
        for col in [
            "row_index",
            "window_index",
            "start_row",
            "end_row",
            "prediction_id",
            "prediction_label",
            "raw_predicted_cause",
            "predicted_cause",
            "raw_cause_score",
            "cause_confidence",
            "cause_reason",
            "smoothing_note",
            "confidence",
            "actual_label",
            "actual_cause_label",
            "is_correct",
            "cause_is_correct",
            "acc_mag_peak",
            "gyro_mag_peak",
            "jerk_peak",
            "throttle_mean",
            "speed_drop",
            "brake_peak",
            "brake_sustained_ratio",
            "yaw_peak",
            "yaw_sustained_ratio",
        ]
        if col in results.columns
    ]
    print("\nPreview")
    print("-" * 7)
    print(results[preview_cols].head(preview_rows).to_string(index=False))


def load_prediction_assets(config_path=DEFAULT_HEURISTIC_CONFIG_PATH):
    config = load_heuristic_config(config_path)
    model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    label_map = joblib.load(os.path.join(MODELS_DIR, "label_map.pkl"))
    return {
        "config": config,
        "config_path": os.path.abspath(config_path),
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "label_map": label_map,
    }


def summarize_prediction_results(results, input_mode):
    summary = {
        "input_mode": input_mode,
        "rows": int(len(results)),
    }

    if "prediction_label" in results.columns:
        summary["behavior_counts"] = {
            str(label): int(count)
            for label, count in results["prediction_label"].value_counts().sort_index().items()
        }
    else:
        summary["behavior_counts"] = {}

    if input_mode == "raw" and "predicted_cause" in results.columns:
        summary["cause_counts"] = {
            str(label): int(count)
            for label, count in results["predicted_cause"].value_counts().sort_index().items()
        }
    else:
        summary["cause_counts"] = {}

    if "confidence" in results.columns:
        summary["behavior_confidence_mean"] = float(pd.to_numeric(results["confidence"], errors="coerce").mean())

    if "cause_confidence" in results.columns:
        summary["cause_confidence_mean"] = float(
            pd.to_numeric(results["cause_confidence"], errors="coerce").mean()
        )

    return summary


def predict_from_input_path(
    input_path,
    input_mode,
    config_path=DEFAULT_HEURISTIC_CONFIG_PATH,
    window_size=DEFAULT_WINDOW_SIZE,
    step_size=DEFAULT_STEP_SIZE,
    cause_label_col=None,
    output_path=None,
    assets=None,
):
    if input_mode not in {"raw", "feature"}:
        raise ValueError("input_mode must be 'raw' or 'feature'.")

    assets = assets or load_prediction_assets(config_path)
    config = assets["config"]
    model = assets["model"]
    scaler = assets["scaler"]
    feature_names = assets["feature_names"]
    label_map = assets["label_map"]

    input_path = os.path.abspath(input_path)
    source_df = read_input_file(input_path)

    if input_mode == "feature":
        feature_df = source_df.copy()
        meta_df = pd.DataFrame({"row_index": source_df.index})
        input_kind = "feature CSV"
    else:
        source_df = rename_requested_cause_column(source_df, cause_label_col)
        source_df = preprocess_raw_sensor_df(source_df, config)
        feature_df, meta = build_features_from_raw(source_df, int(window_size), int(step_size), config)
        meta_df = pd.DataFrame(meta)
        input_kind = "raw CSV"

    X = align_features(feature_df, feature_names)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    results = meta_df.copy()
    results["prediction_id"] = preds.astype(int)
    results["prediction_label"] = [label_map.get(int(pred), pred) for pred in preds]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)
        add_probability_columns(results, probs, model.classes_, label_map)

    if input_mode == "raw":
        results = predict_causes(results, config)
        results = smooth_cause_labels(results, config)

        if "actual_window_label_raw" in results.columns:
            mapped = [coerce_behavior_label_value(value, label_map) for value in results["actual_window_label_raw"]]
            results["actual_label_id"] = pd.Series([item[0] for item in mapped], dtype="Int64")
            results["actual_label"] = [item[1] for item in mapped]
            results["is_correct"] = np.where(
                results["actual_label_id"].notna(),
                results["actual_label_id"].astype("Int64") == results["prediction_id"],
                pd.NA,
            )

        if "actual_window_cause_raw" in results.columns:
            results["actual_cause_label"] = results["actual_window_cause_raw"].apply(canonicalize_cause_label)
            results["cause_is_correct"] = np.where(
                results["actual_cause_label"].notna(),
                results["actual_cause_label"] == results["predicted_cause"],
                pd.NA,
            )

    elif "label" in source_df.columns:
        label_series = pd.to_numeric(source_df["label"], errors="coerce")
        results["actual_label_id"] = label_series
        results["actual_label"] = label_series.apply(
            lambda value: label_map.get(int(value), value) if pd.notna(value) else None
        )
        results["is_correct"] = np.where(
            label_series.notna(),
            label_series.astype("Int64") == results["prediction_id"],
            pd.NA,
        )

    output_path = os.path.abspath(output_path or default_output_path(input_path))
    save_output_file(results, output_path)

    return {
        "input_path": input_path,
        "input_mode": input_mode,
        "input_kind": input_kind,
        "output_path": output_path,
        "config_path": assets["config_path"],
        "source_df": source_df,
        "results": results,
        "preds": preds,
        "label_map": label_map,
        "summary": summarize_prediction_results(results, input_mode),
    }


def auto_analyze_telemetry(
    input_path,
    config_path=DEFAULT_HEURISTIC_CONFIG_PATH,
    output_path=None,
    output_format="csv",
    assets=None,
):
    """
    Automated one-click forensic analysis pipeline:
    1. Reads any supported format (.csv, .xlsx, .xls, .json)
    2. Auto-detects if raw sensor or feature-engineered
    3. Normalizes & standardizes missing sensor columns (fills with 0.0)
    4. Dynamically handles small/short telemetry files
    5. Executes XGBoost behavior prediction + Heuristic Accident Cause Engine
    6. Extracts comprehensive forensic metrics (Peak G-Force, Speed drop, Yaw, etc.)
    """
    assets = assets or load_prediction_assets(config_path)
    config = assets["config"]
    model = assets["model"]
    scaler = assets["scaler"]
    feature_names = assets["feature_names"]
    label_map = assets["label_map"]

    input_path = os.path.abspath(input_path)
    source_df = read_input_file(input_path)

    if source_df is None or len(source_df) == 0:
        raise ValueError("The provided file is empty or could not be parsed.")

    # Check if raw, feature CSV, or Key-Value Summary Sheet (e.g. Evidentia accident export)
    is_kv_sheet = False
    if source_df.shape[1] in (2, 3):
        col0_vals = [sanitize_column_name(str(x)) for x in source_df.iloc[:, 0].dropna().head(12)]
        kv_indicators = ["acceleration", "speed", "jerk", "brake", "crash", "evidence", "status", "field", "project", "detection", "classification", "vehicle", "angular"]
        match_count = sum(1 for v in col0_vals if any(k in v for k in kv_indicators))
        if match_count >= 2 or sanitize_column_name(str(source_df.columns[0])) in ("field", "key", "property", "metric", "attribute"):
            is_kv_sheet = True

    matched_features = sum(1 for col in feature_names if col in source_df.columns)
    is_feature_csv = (matched_features >= len(feature_names) * 0.6)

    if is_kv_sheet:
        input_mode = "raw"
        input_kind = "Accident Reconstruction Evidence Sheet"
        
        # Parse Key-Value dictionary
        kv_dict = {}
        for _, row in source_df.iterrows():
            k = sanitize_column_name(str(row.iloc[0]).strip())
            v = row.iloc[1]
            kv_dict[k] = v

        # Extract kinematics with unit normalization
        def _get_num(keys, default=0.0):
            for k in keys:
                for dict_k, dict_v in kv_dict.items():
                    if k in dict_k:
                        try:
                            return float(dict_v)
                        except (ValueError, TypeError):
                            pass
            return default

        raw_acc = _get_num(["accelerationmagnitude", "acceleration", "acc"], 0.0)
        # Convert m/s^2 to g if value > 15.0 or key says ms2
        acc_g = raw_acc / 9.81 if raw_acc > 15.0 else raw_acc

        raw_speed = _get_num(["detectedspeed", "speed", "velocity"], 0.0)
        # Convert m/s to km/h if speed is in m/s (< 45.0)
        speed_kmh = raw_speed * 3.6 if raw_speed < 45.0 and raw_speed > 0 else raw_speed

        raw_speed_drop = _get_num(["speeddrop", "drop"], 0.0)
        speed_drop_kmh = raw_speed_drop * 3.6 if raw_speed_drop < 30.0 and raw_speed_drop > 0 else raw_speed_drop

        raw_jerk = _get_num(["jerk"], 0.0)
        jerk_g = raw_jerk / 9.81 if raw_jerk > 15.0 else raw_jerk

        raw_gyro = _get_num(["angularacceleration", "angular", "gyro", "yaw"], 0.0)
        raw_brake = _get_num(["brake", "brakepct"], 0.0)
        brake_pct = raw_brake * 100.0 if 0 < raw_brake <= 1.0 else raw_brake

        raw_throttle = _get_num(["throttle", "throttlepct"], 0.0)
        throttle_pct = raw_throttle * 100.0 if 0 < raw_throttle <= 1.0 else raw_throttle

        # Synthesize time-series wave matching the exact evidence metrics
        n_samples = 100
        impact_idx = 50
        time_arr = np.linspace(0, 2.0, n_samples)
        
        sim_speed = np.ones(n_samples) * max(speed_kmh, 45.0)
        decay = np.exp(-np.linspace(0, 4, n_samples - impact_idx))
        sim_speed[impact_idx:] = np.maximum(0.0, sim_speed[impact_idx] - speed_drop_kmh * (1 - decay))

        sim_acc_x = np.random.normal(0, 0.1, n_samples)
        sim_acc_y = np.random.normal(0, 0.1, n_samples)
        sim_acc_z = np.random.normal(0, 0.1, n_samples)

        pulse_w = 6
        pulse = np.sin(np.linspace(0, np.pi, pulse_w)) * acc_g
        for k, p in enumerate(pulse):
            if impact_idx + k < n_samples:
                sim_acc_x[impact_idx + k] -= p

        sim_gyro_z = np.random.normal(0, 0.05, n_samples)
        sim_gyro_z[impact_idx:impact_idx + pulse_w] += raw_gyro * 0.1

        processed_raw_df = pd.DataFrame({
            "AccX": sim_acc_x, "AccY": sim_acc_y, "AccZ": sim_acc_z,
            "GyroX": np.zeros(n_samples), "GyroY": np.zeros(n_samples), "GyroZ": sim_gyro_z,
            "Speed": sim_speed,
            "ThrottlePct": np.ones(n_samples) * throttle_pct,
            "BrakePct": np.ones(n_samples) * brake_pct,
            "AccMag": np.sqrt(sim_acc_x**2 + sim_acc_y**2 + sim_acc_z**2),
            "LinAccMag": np.sqrt(sim_acc_x**2 + sim_acc_y**2 + sim_acc_z**2),
            "GyroMag": np.abs(sim_gyro_z),
        })

        feature_df, meta = build_features_from_raw(processed_raw_df, DEFAULT_WINDOW_SIZE, DEFAULT_STEP_SIZE, config)
        meta_df = pd.DataFrame(meta)

    elif is_feature_csv:
        input_mode = "feature"
        input_kind = "Feature Matrix"
        feature_df = source_df.copy()
        meta_df = pd.DataFrame({"row_index": source_df.index})
        processed_raw_df = pd.DataFrame()
    else:
        input_mode = "raw"
        input_kind = "Raw Sensor Telemetry"
        # Standardize columns: maps aliases and fills missing standard columns with 0.0
        standardized_df = standardize_dataframe(source_df)
        processed_raw_df = preprocess_raw_sensor_df(standardized_df, config)

        # Dynamic windowing so short recordings (even 5-20 rows) produce windows
        total_rows = len(processed_raw_df)
        window_size = min(total_rows, DEFAULT_WINDOW_SIZE)
        if total_rows < DEFAULT_WINDOW_SIZE:
            step_size = max(1, window_size // 2)
        else:
            step_size = DEFAULT_STEP_SIZE

        feature_df, meta = build_features_from_raw(processed_raw_df, window_size, step_size, config)
        meta_df = pd.DataFrame(meta)

    X = align_features(feature_df, feature_names)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    results = meta_df.copy()
    results["prediction_id"] = preds.astype(int)
    results["ml_cause_label"] = [label_map.get(int(pred), pred) for pred in preds]
    results["prediction_label"] = results["ml_cause_label"]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)
        add_probability_columns(results, probs, model.classes_, label_map)

    # Use ML model predictions as the core predicted cause
    results["predicted_cause"] = results["ml_cause_label"]
    results["cause_confidence"] = results.get("confidence", 0.95)

    if input_mode == "raw":
        scored_heuristics = predict_causes(results, config)
        results["cause_reason"] = scored_heuristics["raw_cause_reason"]
    else:
        results["cause_reason"] = "Direct ML Kinematic Analysis"

    summary = summarize_prediction_results(results, input_mode)

    # Determine primary forensic cause
    cause_counts = summary.get("cause_counts", {})
    non_normal_causes = {k: v for k, v in cause_counts.items() if k != "normal_driving"}
    if non_normal_causes:
        primary_cause_key = max(non_normal_causes, key=non_normal_causes.get)
    elif cause_counts:
        primary_cause_key = max(cause_counts, key=cause_counts.get)
    else:
        primary_cause_key = "normal_driving"

    primary_cause_display = CAUSE_DISPLAY_NAMES.get(primary_cause_key, primary_cause_key.replace("_", " ").title())

    # Behavior
    behavior_counts = summary.get("behavior_counts", {})
    primary_behavior = max(behavior_counts, key=behavior_counts.get) if behavior_counts else "NORMAL"

    # Find the most descriptive reason for the primary cause
    primary_reason = "Normal vehicle operation without critical anomaly spikes."
    if "predicted_cause" in results.columns and "cause_reason" in results.columns:
        matching_rows = results[results["predicted_cause"] == primary_cause_key]
        if not matching_rows.empty and "cause_reason" in matching_rows.columns:
            reasons = matching_rows["cause_reason"].dropna().tolist()
            if reasons:
                primary_reason = max(reasons, key=len)

    # Compute key telemetry metrics
    telemetry_metrics = {
        "max_speed": 0.0,
        "avg_speed": 0.0,
        "speed_drop": 0.0,
        "peak_g_force": 0.0,
        "avg_g_force": 0.0,
        "peak_jerk": 0.0,
        "peak_yaw": 0.0,
        "max_brake": 0.0,
        "max_throttle": 0.0,
        "total_duration_samples": len(source_df),
    }

    if not processed_raw_df.empty:
        if "Speed" in processed_raw_df.columns:
            speeds = pd.to_numeric(processed_raw_df["Speed"], errors="coerce").dropna()
            if not speeds.empty:
                telemetry_metrics["max_speed"] = float(speeds.max())
                telemetry_metrics["avg_speed"] = float(speeds.mean())
                if len(speeds) > 1:
                    telemetry_metrics["speed_drop"] = float(max(0.0, speeds.iloc[0] - speeds.iloc[-1]))
        
        acc_col = "LinAccMag" if "LinAccMag" in processed_raw_df.columns else "AccMag"
        if acc_col in processed_raw_df.columns:
            accs = pd.to_numeric(processed_raw_df[acc_col], errors="coerce").dropna()
            if not accs.empty:
                telemetry_metrics["peak_g_force"] = float(accs.max())
                telemetry_metrics["avg_g_force"] = float(accs.mean())
                if len(accs) > 1:
                    telemetry_metrics["peak_jerk"] = float(np.max(np.abs(np.diff(accs))))

        if "GyroZ" in processed_raw_df.columns:
            yaws = pd.to_numeric(processed_raw_df["GyroZ"], errors="coerce").dropna()
            if not yaws.empty:
                telemetry_metrics["peak_yaw"] = float(np.max(np.abs(yaws)))

        if "BrakePct" in processed_raw_df.columns:
            brakes = pd.to_numeric(processed_raw_df["BrakePct"], errors="coerce").dropna()
            if not brakes.empty:
                telemetry_metrics["max_brake"] = float(brakes.max())

        if "ThrottlePct" in processed_raw_df.columns:
            throttles = pd.to_numeric(processed_raw_df["ThrottlePct"], errors="coerce").dropna()
            if not throttles.empty:
                telemetry_metrics["max_throttle"] = float(throttles.max())

        if "EngineTemp" in processed_raw_df.columns:
            eng_temps = pd.to_numeric(processed_raw_df["EngineTemp"], errors="coerce").dropna()
            if not eng_temps.empty:
                telemetry_metrics["peak_engine_temp"] = float(eng_temps.max())
                telemetry_metrics["avg_engine_temp"] = float(eng_temps.mean())

        if "OilTemp" in processed_raw_df.columns:
            oil_temps = pd.to_numeric(processed_raw_df["OilTemp"], errors="coerce").dropna()
            if not oil_temps.empty:
                telemetry_metrics["peak_oil_temp"] = float(oil_temps.max())

    # Assess overall severity
    if primary_cause_key in ("possible_collision_impact", "possible_brake_failure", "engine_overheating"):
        severity = "CRITICAL"
        severity_color = "#EA4335"
    elif primary_cause_key in ("rash_driving", "sharp_turning_or_skid", "hard_braking"):
        severity = "HIGH"
        severity_color = "#FBBC04"
    else:
        severity = "LOW"
        severity_color = "#34A853"

    forensic_insights = {
        "primary_cause_key": primary_cause_key,
        "primary_cause_display": primary_cause_display,
        "primary_behavior": primary_behavior,
        "primary_reason": primary_reason,
        "severity": severity,
        "severity_color": severity_color,
        "telemetry_metrics": telemetry_metrics,
        "behavior_confidence": summary.get("behavior_confidence_mean", 0.0),
        "cause_confidence": summary.get("cause_confidence_mean", 0.0),
    }

    # Save output if requested or default
    if output_path is None:
        root, _ = os.path.splitext(input_path)
        output_path = f"{root}_forensic_predictions.{output_format.lower()}"
    output_path = os.path.abspath(output_path)
    save_output_file(results, output_path, output_format)

    return {
        "input_path": input_path,
        "input_kind": input_kind,
        "output_path": output_path,
        "config_path": assets["config_path"],
        "source_df": source_df,
        "processed_raw_df": processed_raw_df,
        "results": results,
        "summary": summary,
        "forensic_insights": forensic_insights,
    }


def run_prediction_mode(args):
    input_mode = "feature" if args.feature_csv else "raw"
    input_path = args.feature_csv or args.raw_csv
    prediction = predict_from_input_path(
        input_path=input_path,
        input_mode=input_mode,
        config_path=args.config,
        window_size=args.window_size,
        step_size=args.step_size,
        cause_label_col=args.cause_label_col,
        output_path=args.output,
    )
    results = prediction["results"]
    source_df = prediction["source_df"]
    preds = prediction["preds"]
    label_map = prediction["label_map"]

    print(f"\nLoaded {prediction['input_kind']}: {prediction['input_path']}")
    print(f"Generated {len(results):,} prediction row(s)")
    print(f"Saved predictions to: {prediction['output_path']}")

    if input_mode == "raw":
        print(f"Heuristic config loaded from: {prediction['config_path']}")
        print(
            "Raw causes are score-based heuristics on top of the saved behavior model. "
            "Include Speed, BrakePct, and ThrottlePct for stronger cause detection."
        )

    preview_results(results, args.preview)

    if input_mode == "feature" and "label" in source_df.columns:
        evaluate_behavior_predictions(source_df["label"], preds, label_map, "Evaluation against feature labels")
    elif input_mode == "raw" and "actual_label_id" in results.columns:
        evaluate_behavior_predictions(
            results["actual_label_id"],
            results["prediction_id"],
            label_map,
            "Evaluation against windowed raw behavior labels",
        )

    if input_mode == "raw" and "actual_cause_label" in results.columns:
        evaluate_cause_predictions(
            results["actual_cause_label"],
            results["predicted_cause"],
            "Evaluation against windowed raw cause labels",
        )


def run_calibration_mode(args):
    config = load_heuristic_config(args.config)
    input_path = os.path.abspath(args.calibrate_raw_csv)
    source_df = pd.read_csv(input_path)
    source_df = rename_requested_cause_column(source_df, args.cause_label_col)
    source_df = preprocess_raw_sensor_df(source_df, config)

    _feature_df, meta = build_features_from_raw(source_df, args.window_size, args.step_size, config)
    window_df = pd.DataFrame(meta)

    calibrated_config, report_df = calibrate_config_from_windows(window_df, config)

    config_output_path = os.path.abspath(args.save_calibrated_config or default_calibration_output_path())
    report_output_path = os.path.abspath(args.calibration_report or default_calibration_report_path())

    save_json(config_output_path, calibrated_config)
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    report_df.to_csv(report_output_path, index=False)

    print(f"\nLoaded labeled raw CSV: {input_path}")
    print(f"Saved calibrated config to: {config_output_path}")
    print(f"Saved calibration report to: {report_output_path}")
    print_calibration_summary(window_df, report_df)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the saved vehicle behavior model on feature or raw sensor CSVs, attach "
            "score-based heuristic accident causes, or calibrate those heuristics from labeled recordings."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--feature-csv",
        help="CSV that already contains the engineered feature columns expected by the model.",
    )
    group.add_argument(
        "--raw-csv",
        help=(
            "CSV with raw sensor columns like AccX, AccY, AccZ, GyroX, GyroY, GyroZ. "
            "Optional Speed, ThrottlePct, and BrakePct columns improve cause detection."
        ),
    )
    group.add_argument(
        "--calibrate-raw-csv",
        help=(
            "Labeled raw sensor CSV used to calibrate the heuristic threshold config. "
            "It should contain a row-level cause column such as cause_label."
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_HEURISTIC_CONFIG_PATH,
        help=f"Path to the heuristic config JSON. Default: {DEFAULT_HEURISTIC_CONFIG_PATH}",
    )
    parser.add_argument(
        "--output",
        help="Where to save prediction output. Defaults to <input>_predictions.csv.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Sliding window size for raw CSV input. Default: {DEFAULT_WINDOW_SIZE}",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=DEFAULT_STEP_SIZE,
        help=f"Sliding window step size for raw CSV input. Default: {DEFAULT_STEP_SIZE}",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="How many prediction rows to print in the terminal. Default: 10",
    )
    parser.add_argument(
        "--cause-label-col",
        help=(
            "Optional row-level cause label column for evaluation or calibration. "
            "Examples: cause_label, Cause, EventLabel."
        ),
    )
    parser.add_argument(
        "--save-calibrated-config",
        help="Where calibration mode should save the tuned heuristic config JSON.",
    )
    parser.add_argument(
        "--calibration-report",
        help="Where calibration mode should save the metric report CSV.",
    )
    args = parser.parse_args()

    if args.calibrate_raw_csv:
        run_calibration_mode(args)
    else:
        run_prediction_mode(args)


if __name__ == "__main__":
    main()
