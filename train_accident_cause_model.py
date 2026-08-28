"""
DriveOps Accident Cause & Behavior Model Training Script
Trains a Multi-Class XGBoost & Random Forest Classifier directly on Vehicle Accident & Telemetry Datasets.
Target Classes:
  0: normal_driving
  1: rash_driving
  2: hard_braking
  3: sharp_turning_or_skid
  4: possible_collision_impact
  5: possible_brake_failure
"""

import os
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb


# ── Directory Paths ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
DATA_DIR = os.path.join(BASE_DIR, "data")

for folder in [MODELS_DIR, PLOTS_DIR, DATA_DIR]:
    os.makedirs(folder, exist_ok=True)

WINDOW_SIZE = 50
STEP_SIZE = 25

LABEL_MAP = {
    0: "normal_driving",
    1: "rash_driving",
    2: "hard_braking",
    3: "sharp_turning_or_skid",
    4: "possible_collision_impact",
    5: "possible_brake_failure",
}


# ── Signal Processing & Feature Extraction ──────────────────────────────

def lowpass_filter(signal, cutoff=8.0, fs=50.0, order=4):
    """Apply low-pass filter to smooth sensor noise."""
    arr = np.asarray(signal, dtype=float)
    if len(arr) < 15:
        return arr
    nyq = 0.5 * fs
    normal_cutoff = min(0.99, cutoff / nyq)
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    if len(arr) < 3 * max(len(a), len(b)):
        return arr
    return filtfilt(b, a, arr)


def extract_window_features(window):
    """Extract standard statistical, kinematic and cross-axis features for a window."""
    feat = {}
    sensor_cols = [
        col for col in [
            "AccX", "AccY", "AccZ",
            "GyroX", "GyroY", "GyroZ",
            "AccMag", "GyroMag",
            "Speed", "ThrottlePct", "BrakePct",
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

    acc_col = "AccMag" if "AccMag" in window.columns else "AccX"
    if acc_col in window.columns:
        acc = pd.to_numeric(window[acc_col], errors="coerce").fillna(0.0).values.astype(float)
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

    if "AccX" in window.columns and "GyroZ" in window.columns:
        acc_x = pd.to_numeric(window["AccX"], errors="coerce").fillna(0.0).values.astype(float)
        gyro_z = pd.to_numeric(window["GyroZ"], errors="coerce").fillna(0.0).values.astype(float)
        try:
            c = np.corrcoef(acc_x, gyro_z)[0, 1]
            feat["corr_accX_gyroZ"] = 0.0 if np.isnan(c) else float(c)
        except Exception:
            feat["corr_accX_gyroZ"] = 0.0

    return feat


def assign_window_kinematic_label(window, trip_type_hint=None):
    """
    Assign ground-truth event labels based on physical sensor kinematics:
    - Collision Impact: Extreme peak G-force (>3.5g) & high jerk
    - Brake Failure: High brake (>60%) with near-zero speed reduction
    - Hard Braking: Sudden strong negative deceleration or severe speed drop
    - Sharp Turning / Skid: High lateral gyro (yaw rate > 0.30 rad/s) or high horizontal acc
    - Rash Driving: High acceleration surges, aggressive throttle, high jerk energy
    - Normal Driving: Baseline safe driving within normal kinematic bounds
    """
    # 1. Collision check
    acc_mag = window.get("AccMag")
    if acc_mag is not None:
        peak_acc = float(np.max(acc_mag))
        if peak_acc >= 3.8:
            return 4  # possible_collision_impact

    # 2. Brake Failure check
    if "BrakePct" in window.columns and "Speed" in window.columns:
        brake_mean = float(np.mean(window["BrakePct"]))
        speed_vals = window["Speed"].values
        if len(speed_vals) > 1:
            speed_drop = float(speed_vals[0] - speed_vals[-1])
            if brake_mean >= 55.0 and speed_drop <= 2.0:
                return 5  # possible_brake_failure

    # 3. Hard Braking check
    if "AccX" in window.columns:
        min_acc_x = float(np.min(window["AccX"]))
        if min_acc_x <= -1.8:
            return 2  # hard_braking
    if "Speed" in window.columns:
        speed_vals = window["Speed"].values
        if len(speed_vals) > 1:
            speed_drop = float(speed_vals[0] - speed_vals[-1])
            if speed_drop >= 12.0:
                return 2  # hard_braking

    # 4. Sharp Turning / Skid check
    if "GyroZ" in window.columns:
        peak_yaw = float(np.max(np.abs(window["GyroZ"])))
        if peak_yaw >= 0.35:
            return 3  # sharp_turning_or_skid
    if "GyroMag" in window.columns:
        peak_gyro = float(np.max(window["GyroMag"]))
        if peak_gyro >= 0.60:
            return 3  # sharp_turning_or_skid

    # 5. Rash Driving / Overspeeding check
    if acc_mag is not None:
        peak_acc = float(np.max(acc_mag))
        if peak_acc >= 2.0 or (trip_type_hint == "AGGRESSIVE" and peak_acc >= 1.5):
            return 1  # rash_driving
    if "ThrottlePct" in window.columns:
        throttle_max = float(np.max(window["ThrottlePct"]))
        if throttle_max >= 75.0:
            return 1  # rash_driving

    return 0  # normal_driving


def process_and_window_df(df, label_override=None, trip_type_hint=None, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Slide a window across a dataframe and extract labeled feature rows."""
    if len(df) == 0:
        return []

    # Standardize columns
    for c in ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "Speed", "ThrottlePct", "BrakePct"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Compute AccMag and GyroMag
    df["AccMag"] = np.sqrt(df["AccX"]**2 + df["AccY"]**2 + df["AccZ"]**2)
    df["GyroMag"] = np.sqrt(df["GyroX"]**2 + df["GyroY"]**2 + df["GyroZ"]**2)

    rows = []
    if len(df) < window_size:
        windows = [(0, len(df))]
    else:
        windows = [(start, start + window_size) for start in range(0, len(df) - window_size + 1, step_size)]

    for start, end in windows:
        window = df.iloc[start:end]
        if label_override is not None:
            lbl = int(label_override)
        else:
            lbl = assign_window_kinematic_label(window, trip_type_hint)

        feat = extract_window_features(window)
        feat["label"] = int(lbl)
        rows.append(feat)

    return rows


def generate_hard_braking_telemetry(num_samples=250, window_len=100):
    """
    Generate realistic emergency hard braking telemetry.
    Characteristics: Extreme negative longitudinal deceleration (-2.0g to -4.5g), steep speed drop, high brake pedal pressure.
    """
    dfs = []
    for _ in range(num_samples):
        n = window_len
        init_speed = np.random.uniform(50.0, 110.0)
        speed = np.ones(n) * init_speed
        
        brake_idx = np.random.randint(n // 4, n // 2)
        decel_rate = np.random.uniform(2.2, 4.5)  # g
        
        acc_x = np.random.normal(0.0, 0.1, n)
        acc_y = np.random.normal(0.0, 0.1, n)
        acc_z = np.random.normal(0.0, 0.1, n)
        
        brake_pct = np.zeros(n)
        brake_duration = np.random.randint(15, 30)
        brake_end = min(n, brake_idx + brake_duration)
        
        acc_x[brake_idx:brake_end] -= decel_rate + np.random.normal(0, 0.15, brake_end - brake_idx)
        brake_pct[brake_idx:brake_end] = np.random.uniform(70.0, 100.0, brake_end - brake_idx)
        
        # Speed drops rapidly
        for i in range(brake_idx, n):
            speed[i] = max(0.0, speed[i-1] - (decel_rate * 9.81 * 0.1 * 3.6))
        
        gyro_x = np.random.normal(0, 0.05, n)
        gyro_y = np.random.normal(0, 0.05, n)
        gyro_z = np.random.normal(0, 0.05, n)

        df = pd.DataFrame({
            "AccX": acc_x, "AccY": acc_y, "AccZ": acc_z,
            "GyroX": gyro_x, "GyroY": gyro_y, "GyroZ": gyro_z,
            "Speed": speed,
            "ThrottlePct": np.zeros(n),
            "BrakePct": brake_pct,
        })
        dfs.append(df)
    return dfs


def generate_skid_turning_telemetry(num_samples=250, window_len=100):
    """
    Generate realistic vehicle skidding and sharp turn maneuver telemetry.
    Characteristics: High lateral yaw rate (GyroZ > 0.40 rad/s), high lateral acceleration (AccY > 1.5g), speed reduction.
    """
    dfs = []
    for _ in range(num_samples):
        n = window_len
        init_speed = np.random.uniform(40.0, 90.0)
        speed = np.ones(n) * init_speed
        
        skid_idx = np.random.randint(n // 4, n // 2)
        skid_duration = np.random.randint(12, 28)
        skid_end = min(n, skid_idx + skid_duration)
        
        acc_x = np.random.normal(0.0, 0.1, n)
        acc_y = np.random.normal(0.0, 0.1, n)
        acc_z = np.random.normal(0.0, 0.1, n)
        
        yaw_rate = np.random.uniform(0.45, 1.8) * np.random.choice([-1, 1])
        lateral_g = np.random.uniform(1.6, 3.2) * np.random.choice([-1, 1])
        
        acc_y[skid_idx:skid_end] += lateral_g + np.random.normal(0, 0.15, skid_end - skid_idx)
        
        gyro_x = np.random.normal(0, 0.05, n)
        gyro_y = np.random.normal(0, 0.05, n)
        gyro_z = np.random.normal(0, 0.05, n)
        gyro_z[skid_idx:skid_end] += yaw_rate + np.random.normal(0, 0.1, skid_end - skid_idx)
        
        for i in range(skid_idx, n):
            speed[i] = max(10.0, speed[i-1] - 0.8)

        df = pd.DataFrame({
            "AccX": acc_x, "AccY": acc_y, "AccZ": acc_z,
            "GyroX": gyro_x, "GyroY": gyro_y, "GyroZ": gyro_z,
            "Speed": speed,
            "ThrottlePct": np.clip(np.random.uniform(10, 30, n), 0, 100),
            "BrakePct": np.clip(np.random.uniform(0, 40, n), 0, 100),
        })
        dfs.append(df)
    return dfs


def generate_collision_impact_telemetry(num_samples=260, window_len=100):
    """
    Generate realistic high-G multi-axis impact shockwave telemetry.
    Characteristics: High peak deceleration (3.5g - 8.0g), extreme jerk (>2.5 g/s), rapid velocity collapse.
    """
    dfs = []
    for _ in range(num_samples):
        n = window_len
        time_arr = np.linspace(0, 2.0, n)
        
        # Initial cruise speed
        init_speed = np.random.uniform(40.0, 90.0)
        speed = np.ones(n) * init_speed
        
        # Impact point
        impact_idx = np.random.randint(n // 3, 2 * n // 3)
        
        # Shockwave G-force pulse
        acc_x = np.random.normal(0.0, 0.15, n)
        acc_y = np.random.normal(0.0, 0.15, n)
        acc_z = np.random.normal(0.0, 0.15, n)
        
        impact_g = np.random.uniform(3.8, 8.5)
        pulse_width = np.random.randint(3, 8)
        pulse = np.sin(np.linspace(0, np.pi, pulse_width)) * impact_g
        
        for k, p in enumerate(pulse):
            if impact_idx + k < n:
                acc_x[impact_idx + k] -= p * np.random.uniform(0.7, 1.0)
                acc_y[impact_idx + k] += p * np.random.uniform(-0.5, 0.5)
                acc_z[impact_idx + k] += p * np.random.uniform(-0.3, 0.4)
        
        # Speed drops violently to 0 or near 0
        decay = np.exp(-np.linspace(0, 5, n - impact_idx))
        speed[impact_idx:] = init_speed * decay * np.random.uniform(0.0, 0.2)
        
        gyro_x = np.random.normal(0, 0.1, n)
        gyro_y = np.random.normal(0, 0.1, n)
        gyro_z = np.random.normal(0, 0.1, n)
        gyro_z[impact_idx:impact_idx + pulse_width] += np.random.uniform(-2.5, 2.5)

        df = pd.DataFrame({
            "AccX": acc_x, "AccY": acc_y, "AccZ": acc_z,
            "GyroX": gyro_x, "GyroY": gyro_y, "GyroZ": gyro_z,
            "Speed": speed,
            "ThrottlePct": np.zeros(n),
            "BrakePct": np.zeros(n),
        })
        dfs.append(df)
    return dfs


def generate_brake_failure_telemetry(num_samples=180, window_len=100):
    """
    Generate realistic brake failure telemetry.
    Characteristics: High persistent brake pedal engagement (60% - 100%) with minimal or no speed reduction.
    """
    dfs = []
    for _ in range(num_samples):
        n = window_len
        init_speed = np.random.uniform(50.0, 100.0)
        
        # Speed remains high or even accelerates slightly despite 80-100% brake
        speed_drift = np.linspace(0, np.random.uniform(-2.0, 5.0), n)
        speed = init_speed + speed_drift + np.random.normal(0, 0.5, n)
        
        brake_pct = np.clip(np.random.uniform(65.0, 98.0, n) + np.random.normal(0, 2, n), 0, 100)
        throttle_pct = np.clip(np.random.uniform(0.0, 10.0, n), 0, 100)
        
        # Acceleration does not show significant negative deceleration
        acc_x = np.random.normal(0.05, 0.2, n)
        acc_y = np.random.normal(0.0, 0.15, n)
        acc_z = np.random.normal(0.0, 0.15, n)
        
        gyro_x = np.random.normal(0, 0.05, n)
        gyro_y = np.random.normal(0, 0.05, n)
        gyro_z = np.random.normal(0, 0.05, n)

        df = pd.DataFrame({
            "AccX": acc_x, "AccY": acc_y, "AccZ": acc_z,
            "GyroX": gyro_x, "GyroY": gyro_y, "GyroZ": gyro_z,
            "Speed": speed,
            "ThrottlePct": throttle_pct,
            "BrakePct": brake_pct,
        })
        dfs.append(df)
    return dfs


# ── Ingest Datasets ─────────────────────────────────────────────────────

def load_and_build_dataset():
    print("=" * 65)
    print("  DRIVEOPS — ACCIDENT CAUSE & BEHAVIOR MODEL TRAINING")
    print("=" * 65)

    all_window_rows = []

    # ── 1. UAH-DriveSet (Real naturalistic driving with annotated behaviors)
    uah_base = os.path.join(DATASET_DIR, "UAH-DRIVESET-v1", "UAH-DRIVESET-v1")
    if os.path.exists(uah_base):
        print("\n[1/5] Loading UAH-DriveSet...")
        trip_dirs = glob.glob(os.path.join(uah_base, "*", "*"))
        uah_count = 0
        for trip_dir in trip_dirs:
            folder_name = os.path.basename(trip_dir).upper()
            acc_file = os.path.join(trip_dir, "RAW_ACCELEROMETERS.txt")
            gps_file = os.path.join(trip_dir, "RAW_GPS.txt")
            
            if not os.path.exists(acc_file):
                continue

            try:
                # Load accelerometer data (timestamp, raw x,y,z, filtered x,y,z)
                acc_data = np.loadtxt(acc_file)
                if acc_data.ndim < 2 or acc_data.shape[1] < 4:
                    continue
                
                acc_df = pd.DataFrame({
                    "AccX": acc_data[:, 1],
                    "AccY": acc_data[:, 2],
                    "AccZ": acc_data[:, 3],
                    "GyroX": 0.0, "GyroY": 0.0, "GyroZ": 0.0,
                    "Speed": 0.0, "ThrottlePct": 0.0, "BrakePct": 0.0
                })

                if os.path.exists(gps_file):
                    gps_data = np.loadtxt(gps_file)
                    if gps_data.ndim >= 2 and gps_data.shape[1] >= 4:
                        # GPS speed is in km/h (col 3)
                        speed_vals = gps_data[:, 3]
                        # Interpolate GPS speed to accelerometer length
                        if len(speed_vals) > 1:
                            acc_df["Speed"] = np.interp(
                                np.linspace(0, 1, len(acc_df)),
                                np.linspace(0, 1, len(speed_vals)),
                                speed_vals
                            )

                # Trip hint
                trip_hint = "AGGRESSIVE" if "AGGRESSIVE" in folder_name else "DROWSY" if "DROWSY" in folder_name else "NORMAL"
                all_window_rows.extend(process_and_window_df(acc_df, trip_type_hint=trip_hint))
                uah_count += 1
            except Exception:
                pass
        print(f"  [OK] Processed {uah_count} UAH-DriveSet trips")

    # ── 2. OBD-II Real Vehicle Dataset
    obd_base = os.path.join(DATASET_DIR, "OBD-II-Dataset")
    if os.path.exists(obd_base):
        print("\n[2/5] Loading OBD-II Emergency & Normal Driving Dataset...")
        obd_files = glob.glob(os.path.join(obd_base, "*.csv"))
        obd_count = 0
        for fpath in obd_files:
            fname = os.path.basename(fpath).upper()
            hint = "AGGRESSIVE" if "BESCHLEUNIGUNG" in fname else "DROWSY" if "GLATTEIS" in fname else "NORMAL"

            try:
                df = pd.read_csv(fpath)
                df.rename(columns={
                    "Vehicle Speed": "Speed",
                    "Accelerator Pedal Position E [%]": "ThrottlePct",
                    "Brake Pedal Position [%]": "BrakePct",
                }, inplace=True)

                if "Speed" in df.columns:
                    speed_arr = pd.to_numeric(df["Speed"], errors="coerce").fillna(0.0).values
                    # Derive longitudinal acceleration (g) from speed (km/h) diff at 10Hz
                    diff_speed = np.diff(speed_arr, prepend=speed_arr[0]) / 3.6  # m/s
                    acc_x = diff_speed / 0.1 / 9.81  # g
                    df["AccX"] = acc_x
                
                all_window_rows.extend(process_and_window_df(df, trip_type_hint=hint))
                obd_count += 1
            except Exception:
                pass
        print(f"  [OK] Processed {obd_count} OBD-II driving records")

    # ── 3. Motion Data
    train_motion_path = os.path.join(DATASET_DIR, "train_motion_data.csv")
    if os.path.exists(train_motion_path):
        print("\n[3/5] Loading Motion Telemetry Dataset...")
        try:
            tm_df = pd.read_csv(train_motion_path)
            for class_name, group_df in tm_df.groupby("Class"):
                hint = "AGGRESSIVE" if class_name == "AGGRESSIVE" else "NORMAL"
                all_window_rows.extend(process_and_window_df(group_df.copy(), trip_type_hint=hint))
            print(f"  [OK] Loaded motion telematics data")
        except Exception as e:
            print(f"  [WARN] {e}")

    # ── 4. Emergency Hard Braking Telemetry
    print("\n[4/7] Generating Emergency Hard Braking Data...")
    braking_dfs = generate_hard_braking_telemetry(num_samples=250)
    for b_df in braking_dfs:
        all_window_rows.extend(process_and_window_df(b_df, label_override=2))  # 2: hard_braking
    print(f"  [OK] Generated {len(braking_dfs)} hard braking sequences")

    # ── 5. Sharp Turning / Skidding Telemetry
    print("\n[5/7] Generating Sharp Turning & Vehicle Skidding Data...")
    skid_dfs = generate_skid_turning_telemetry(num_samples=250)
    for s_df in skid_dfs:
        all_window_rows.extend(process_and_window_df(s_df, label_override=3))  # 3: sharp_turning_or_skid
    print(f"  [OK] Generated {len(skid_dfs)} skidding/turning sequences")

    # ── 6. High-G Collision Impact Telemetry (Crash Simulation)
    print("\n[6/7] Generating Collision & Severe Impact Shockwave Data...")
    collision_dfs = generate_collision_impact_telemetry(num_samples=260)
    for c_df in collision_dfs:
        all_window_rows.extend(process_and_window_df(c_df, label_override=4))  # 4: possible_collision_impact
    print(f"  [OK] Generated {len(collision_dfs)} collision impact test sequences")

    # ── 7. Brake Failure Telemetry
    print("\n[7/7] Generating Brake Failure Dynamics Data...")
    brake_fail_dfs = generate_brake_failure_telemetry(num_samples=260)
    for bf_df in brake_fail_dfs:
        all_window_rows.extend(process_and_window_df(bf_df, label_override=5))  # 5: possible_brake_failure
    print(f"  [OK] Generated {len(brake_fail_dfs)} brake failure sequences")

    # Assemble Master DataFrame
    dataset_df = pd.DataFrame(all_window_rows).dropna()
    print(f"\n[SUMMARY] Master dataset compiled with {len(dataset_df):,} total feature windows.")
    return dataset_df


# ── Model Training & Evaluation ──────────────────────────────────────────

def train_and_evaluate(dataset_df):
    feature_cols = [c for c in dataset_df.columns if c != "label"]

    # Balance majority normal class
    normal_df = dataset_df[dataset_df["label"] == 0]
    non_normal_df = dataset_df[dataset_df["label"] != 0]

    max_minority = non_normal_df["label"].value_counts().max()
    target_normal = min(len(normal_df), max(max_minority * 2, 4000))
    balanced_normal_df = normal_df.sample(n=target_normal, random_state=42)

    curated_df = pd.concat([balanced_normal_df, non_normal_df], ignore_index=True)

    X = curated_df[feature_cols].values
    y = curated_df["label"].values.astype(int)

    print("\nCurated Training Class Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for u, cnt in zip(unique, counts):
        print(f"  Class {u} ({LABEL_MAP.get(u, 'Unknown')}): {cnt:,} samples ({cnt/len(y)*100:.1f}%)")

    # Train / Test Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Feature Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compute balanced sample weights directly for tree training
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight("balanced", y_train)

    print("\nTraining Tuned Multi-Class Accident Cause XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=350,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.90,
        colsample_bytree=0.90,
        objective="multi:softprob",
        num_class=6,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 65)
    print(f"  MODEL EVALUATION RESULTS — ACCURACY: {acc * 100:.2f}%")
    print("=" * 65)

    target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7), dpi=120)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[LABEL_MAP[i].replace("_", " ").title() for i in sorted(LABEL_MAP.keys())],
        yticklabels=[LABEL_MAP[i].replace("_", " ").title() for i in sorted(LABEL_MAP.keys())],
    )
    plt.title(f"Accident Cause Confusion Matrix (Accuracy: {acc*100:.1f}%)", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Cause", fontsize=10, fontweight="bold")
    plt.ylabel("Actual Cause", fontsize=10, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    cm_path = os.path.join(PLOTS_DIR, "accident_cause_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[OK] Saved Confusion Matrix Plot to: {cm_path}")

    # Save Models and Assets
    joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    joblib.dump(model, os.path.join(MODELS_DIR, "accident_cause_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_names.pkl"))
    joblib.dump(LABEL_MAP, os.path.join(MODELS_DIR, "label_map.pkl"))

    print(f"\n[OK] Model & Assets successfully saved to: {MODELS_DIR}")
    return acc


def main():
    dataset_df = load_and_build_dataset()
    train_and_evaluate(dataset_df)


if __name__ == "__main__":
    main()
