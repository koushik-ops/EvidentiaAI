# ==============================================================
#   VEHICLE FORENSIC AI — COMPLETE TRAINING SCRIPT (FIXED)
#   Fixes: KeyError 'label', DeviceMotion path, train_motion labels
# ==============================================================
#   HOW TO RUN:
#   python D:\DriveOps\train_model.py
# ==============================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ──────────────────────────────────────────────────────────────
#  SECTION 0 — YOUR PATHS  (update if needed)
# ──────────────────────────────────────────────────────────────

# From your screenshot your script is in D:\DriveOps\
# Put your dataset folders here:
DEVICE_MOTION_DIR = r"D:\DriveOps\dataset\A_DeviceMotion_data"
OBD_DIR           = r"D:\DriveOps\dataset\OBD-II-Dataset"
TRAIN_MOTION      = r"D:\DriveOps\dataset\train_motion_data.csv"
TEST_MOTION       = r"D:\DriveOps\dataset\test_motion_data.csv"

# Output
OUTPUT_DIR  = r"D:\DriveOps"
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
DATA_DIR    = os.path.join(OUTPUT_DIR, "data")

for folder in [MODELS_DIR, PLOTS_DIR, DATA_DIR]:
    os.makedirs(folder, exist_ok=True)

WINDOW_SIZE = 50
STEP_SIZE   = 25
label_map   = {0: 'SLOW', 1: 'NORMAL', 2: 'AGGRESSIVE'}

print("=" * 65)
print("  VEHICLE FORENSIC AI — MODEL TRAINING (FIXED)")
print("=" * 65)


# ──────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

def lowpass_filter(signal, cutoff=10, fs=50, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if len(signal) < 3 * max(len(a), len(b)):
        return signal
    return filtfilt(b, a, signal)


def assign_label(accmag):
    if accmag < 0.3:  return 0   # SLOW
    elif accmag < 2.0: return 1  # NORMAL
    else:              return 2  # AGGRESSIVE


def extract_window_features(window):
    feat = {}
    sensor_cols = [c for c in
                   ['AccX','AccY','AccZ','GyroX','GyroY','GyroZ',
                    'AccMag','GyroMag','Speed','ThrottlePct']
                   if c in window.columns]

    for col in sensor_cols:
        vals = window[col].values.astype(float)
        feat[f'{col}_mean'] = np.mean(vals)
        feat[f'{col}_std']  = np.std(vals)
        feat[f'{col}_max']  = np.max(vals)
        feat[f'{col}_min']  = np.min(vals)
        feat[f'{col}_rms']  = np.sqrt(np.mean(vals ** 2))

    if 'AccMag' in window.columns:
        acc = window['AccMag'].values.astype(float)
        jerk = np.diff(acc)
        feat['jerk_mean']  = np.mean(np.abs(jerk))
        feat['jerk_max']   = np.max(np.abs(jerk))
        feat['jerk_std']   = np.std(jerk)
        feat['acc_energy'] = float(np.sum(acc ** 2))

    if 'AccX' in window.columns and 'GyroZ' in window.columns:
        try:
            c = np.corrcoef(window['AccX'].values,
                            window['GyroZ'].values)[0, 1]
            feat['corr_accX_gyroZ'] = 0.0 if np.isnan(c) else float(c)
        except Exception:
            feat['corr_accX_gyroZ'] = 0.0

    return feat


def apply_sliding_window(df):
    rows = []
    for start in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
        window = df.iloc[start: start + WINDOW_SIZE]
        feat = extract_window_features(window)
        feat['label'] = int(window['label'].mode()[0])
        rows.append(feat)
    return pd.DataFrame(rows)


def add_magnitudes_and_label(df):
    """Compute AccMag, GyroMag, then assign label if missing."""
    if all(c in df.columns for c in ['AccX','AccY','AccZ']):
        df['AccMag'] = np.sqrt(df['AccX']**2 +
                               df['AccY']**2 +
                               df['AccZ']**2)
    if all(c in df.columns for c in ['GyroX','GyroY','GyroZ']):
        df['GyroMag'] = np.sqrt(df['GyroX']**2 +
                                df['GyroY']**2 +
                                df['GyroZ']**2)

    # Assign label if not already present
    if 'label' not in df.columns:
        if 'AccMag' in df.columns:
            df['label'] = df['AccMag'].apply(assign_label)
        elif 'Speed' in df.columns:
            df['label'] = df['Speed'].diff().abs().fillna(0).apply(assign_label)
        else:
            df['label'] = 1  # default NORMAL
    return df


# ──────────────────────────────────────────────────────────────
#  LOAD DATASETS
# ──────────────────────────────────────────────────────────────

all_dfs = []

# ── 1. A_DeviceMotion_data ─────────────────────────────────────
print("\n[1/4] Loading A_DeviceMotion_data...")

# Try multiple possible locations automatically
dm_candidates = [
    DEVICE_MOTION_DIR,
    r"D:\DriveOps\A_DeviceMotion_data",
    r"D:\dataset\A_DeviceMotion_data",
    r"D:\A_DeviceMotion_data",
]
dm_dir_found = None
for candidate in dm_candidates:
    if os.path.exists(candidate):
        dm_dir_found = candidate
        break

if dm_dir_found:
    print(f"  Found at: {dm_dir_found}")
    dm_loaded = 0
    for i in range(1, 25):
        fpath = os.path.join(dm_dir_found, f"sub_{i}.csv")
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath)
            df.rename(columns={
                'userAcceleration.x': 'AccX',
                'userAcceleration.y': 'AccY',
                'userAcceleration.z': 'AccZ',
                'rotationRate.x':     'GyroX',
                'rotationRate.y':     'GyroY',
                'rotationRate.z':     'GyroZ',
            }, inplace=True)
            df.dropna(subset=[c for c in ['AccX','AccY','AccZ']
                              if c in df.columns], inplace=True)
            for col in ['AccX','AccY','AccZ','GyroX','GyroY','GyroZ']:
                if col in df.columns:
                    df[col] = lowpass_filter(df[col].values)
            df = add_magnitudes_and_label(df)
            df['source'] = 'device_motion'
            all_dfs.append(df)
            dm_loaded += 1
        except Exception as e:
            print(f"  [WARN] sub_{i}: {e}")
    print(f"  [OK] Loaded {dm_loaded} subject files")
else:
    print("  [WARN] A_DeviceMotion_data folder not found - skipping")
    print("  (This is OK - OBD + train_motion will still train the model)")


# ── 2. OBD-II Dataset ─────────────────────────────────────────
print("\n[2/4] Loading OBD-II Dataset...")

obd_candidates = [
    OBD_DIR,
    r"D:\DriveOps\OBD-II-Dataset",
    r"D:\dataset\OBD-II-Dataset",
    r"D:\OBD-II-Dataset",
]
obd_dir_found = None
for candidate in obd_candidates:
    if os.path.exists(candidate):
        obd_dir_found = candidate
        break

if obd_dir_found:
    print(f"  Found at: {obd_dir_found}")
    obd_loaded = 0
    for fname in os.listdir(obd_dir_found):
        if not fname.endswith('.csv'):
            continue
        fu = fname.upper()
        if   'STAU'   in fu: lbl = 0   # SLOW
        elif 'NORMAL' in fu: lbl = 1   # NORMAL
        elif 'FREI'   in fu: lbl = 1   # NORMAL
        else:                lbl = 1
        try:
            df = pd.read_csv(os.path.join(obd_dir_found, fname))
            df.rename(columns={
                'Vehicle Speed':                    'Speed',
                'Engine RPM':                       'RPM',
                'Accelerator Pedal Position E [%]': 'ThrottlePct',
            }, inplace=True)
            keep = [c for c in ['Speed','RPM','ThrottlePct']
                    if c in df.columns]
            if not keep:
                continue
            df = df[keep].copy()
            df.dropna(inplace=True)
            if 'Speed' in df.columns:
                df['AccMag'] = df['Speed'].diff().abs().fillna(0)
            df['label']  = lbl
            df['source'] = 'obd'
            all_dfs.append(df)
            obd_loaded += 1
        except Exception as e:
            pass   # skip bad files silently
    print(f"  [OK] Loaded {obd_loaded} OBD files")
else:
    print(f"  [WARN] OBD folder not found - skipping")


# ── 3. train_motion_data.csv ──────────────────────────────────
print("\n[3/4] Loading train_motion_data.csv...")

# Try multiple locations
tm_candidates = [
    TRAIN_MOTION,
    r"D:\DriveOps\train_motion_data.csv",
    r"D:\dataset\train_motion_data.csv",
    r"D:\train_motion_data.csv",
]
tm_found = None
for candidate in tm_candidates:
    if os.path.exists(candidate):
        tm_found = candidate
        break

if tm_found:
    print(f"  Found at: {tm_found}")
    try:
        df = pd.read_csv(tm_found)
        df.columns = df.columns.str.strip()
        print(f"  Columns: {list(df.columns)}")

        # ── FIX: intelligently find or create label column ───
        # Check if any column is a label
        label_col = None
        for c in df.columns:
            if c.lower() in ['label','class','activity','target',
                              'behaviour','behavior']:
                label_col = c
                break

        if label_col:
            df.rename(columns={label_col: 'label'}, inplace=True)
            # Convert text labels to numbers
            if df['label'].dtype == object:
                le_tmp = LabelEncoder()
                df['label'] = le_tmp.fit_transform(df['label'])
                print(f"  Label encoding: {dict(zip(le_tmp.classes_, le_tmp.transform(le_tmp.classes_)))}")
            print(f"  [OK] Found label column: '{label_col}'")
        else:
            # No label column — create from sensor data
            print("  [WARN] No label column found - assigning from motion intensity")

            # Rename sensor columns if needed
            df.rename(columns={
                'x-axis (g)':    'AccX',
                'y-axis (g)':    'AccY',
                'z-axis (g)':    'AccZ',
                'x-axis (deg/s)':'GyroX',
                'y-axis (deg/s)':'GyroY',
                'z-axis (deg/s)':'GyroZ',
            }, inplace=True)

            df = add_magnitudes_and_label(df)

        df['source'] = 'train_motion'
        all_dfs.append(df)
        print(f"  [OK] Loaded {len(df):,} rows")

    except Exception as e:
        print(f"  [WARN] Error loading train_motion: {e}")
else:
    print(f"  [WARN] train_motion_data.csv not found - skipping")


# ── 4. Combine all ────────────────────────────────────────────
print("\n[4/4] Combining all datasets...")

if not all_dfs:
    print("\nERROR: No data loaded at all!")
    print("Check that your dataset folders exist and paths are correct.")
    sys.exit(1)

# Combine — NaN for missing columns is fine
combined = pd.concat(all_dfs, ignore_index=True, sort=False)

# ── CRITICAL FIX: make sure 'label' column exists ─────────────
if 'label' not in combined.columns:
    print("  [WARN] 'label' column missing after combine - assigning from AccMag")
    if 'AccMag' not in combined.columns:
        acc_cols = [c for c in combined.columns
                    if 'acc' in c.lower()]
        if len(acc_cols) >= 3:
            combined['AccMag'] = np.sqrt(
                combined[acc_cols[0]]**2 +
                combined[acc_cols[1]]**2 +
                combined[acc_cols[2]]**2
            )
    if 'AccMag' in combined.columns:
        combined['label'] = combined['AccMag'].apply(assign_label)
    else:
        combined['label'] = 1   # all NORMAL as last resort

# Convert label to int, drop NaN labels
combined['label'] = pd.to_numeric(combined['label'], errors='coerce')
combined.dropna(subset=['label'], inplace=True)
combined['label'] = combined['label'].astype(int)

# Keep only numeric columns
combined = combined.select_dtypes(include=[np.number])
combined.fillna(0, inplace=True)

# Make sure label is still there after numeric filter
if 'label' not in combined.columns:
    print("ERROR: label column lost after numeric filter")
    sys.exit(1)

print(f"  [OK] Total rows: {len(combined):,}")
print(f"  [OK] Columns: {list(combined.columns)}")
print(f"\n  Label distribution (0=SLOW, 1=NORMAL, 2=AGGRESSIVE):")
dist = combined['label'].value_counts().sort_index()
for lbl, count in dist.items():
    pct = count / len(combined) * 100
    print(f"    {label_map.get(int(lbl), lbl):<12}: {count:>8,}  ({pct:.1f}%)")

# Save combined
raw_path = os.path.join(DATA_DIR, "combined_raw.csv")
combined.to_csv(raw_path, index=False)
print(f"\n  [OK] Saved: {raw_path}")


# ──────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FEATURE ENGINEERING - SLIDING WINDOW")
print("=" * 65)
print(f"\n  Window size : {WINDOW_SIZE} rows")
print(f"  Step size   : {STEP_SIZE} rows (50% overlap)")
print(f"  Processing  : {len(combined):,} rows -> feature windows...")

features_df = apply_sliding_window(combined)

print(f"\n  [OK] Windows created : {len(features_df):,}")
print(f"  [OK] Features/window : {len(features_df.columns) - 1}")
print(f"\n  Label distribution after windowing:")
for lbl, count in features_df['label'].value_counts().sort_index().items():
    pct = count / len(features_df) * 100
    print(f"    {label_map.get(int(lbl), lbl):<12}: {count:>7,}  ({pct:.1f}%)")

feat_path = os.path.join(DATA_DIR, "features.csv")
features_df.to_csv(feat_path, index=False)
print(f"\n  [OK] Saved: {feat_path}")


# ──────────────────────────────────────────────────────────────
#  PREPARE FOR TRAINING
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PREPARING TRAINING DATA")
print("=" * 65)

X = features_df.drop(columns=['label'], errors='ignore')
y = features_df['label'].astype(int)
X = X.select_dtypes(include=[np.number]).fillna(0)
feature_names = X.columns.tolist()

print(f"\n  Features : {len(feature_names)}")
print(f"  Samples  : {len(X):,}")

# Normalise
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train : {len(X_train):,}")
print(f"  Test  : {len(X_test):,}")

# SMOTE — balance classes
print(f"\n  Applying SMOTE...")
print(f"  Before: {dict(y_train.value_counts().sort_index())}")
try:
    sm = SMOTE(random_state=42, k_neighbors=min(5, y_train.value_counts().min()-1))
    X_train_b, y_train_b = sm.fit_resample(X_train, y_train)
    print(f"  After : {dict(pd.Series(y_train_b).value_counts().sort_index())}")
except Exception as e:
    print(f"  [WARN] SMOTE skipped: {e} - using original")
    X_train_b, y_train_b = X_train, y_train


# ──────────────────────────────────────────────────────────────
#  TRAIN MODELS
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  TRAINING MODELS")
print("=" * 65)

# Random Forest
print("\n  [1/2] Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                             random_state=42, n_jobs=1)
rf.fit(X_train_b, y_train_b)
rf_preds = rf.predict(X_test)
rf_acc   = accuracy_score(y_test, rf_preds)
print(f"  [OK] Accuracy: {rf_acc*100:.2f}%")

# XGBoost
print("\n  [2/2] XGBoost (main model)...")
n_classes = len(np.unique(y))
xgb_params = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': 1,
    'use_label_encoder': False,
}
if n_classes > 2:
    xgb_params.update({
        'objective': 'multi:softmax',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
    })
else:
    xgb_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
    })

xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train_b, y_train_b,
              eval_set=[(X_test, y_test)], verbose=100)
xgb_preds = xgb_model.predict(X_test)
xgb_acc   = accuracy_score(y_test, xgb_preds)
print(f"\n  [OK] XGBoost Accuracy: {xgb_acc*100:.2f}%")


# ──────────────────────────────────────────────────────────────
#  EVALUATION + PLOTS
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  EVALUATION")
print("=" * 65)

class_names = [label_map[i] for i in sorted(np.unique(y_test))]
print("\n  Random Forest:\n",
      classification_report(y_test, rf_preds, target_names=class_names))
print("  XGBoost:\n",
      classification_report(y_test, xgb_preds, target_names=class_names))

# Plot 1 — Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Confusion Matrices", fontsize=14)
for ax, preds, title in zip(axes,
                             [rf_preds, xgb_preds],
                             ['Random Forest', 'XGBoost']):
    cm = confusion_matrix(y_test, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_title(f"{title}  ({accuracy_score(y_test,preds)*100:.1f}%)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"), dpi=150)
plt.close(fig)

# Plot 2 — Feature importance
importances = xgb_model.feature_importances_
top_feat = pd.Series(importances, index=feature_names).nlargest(20).sort_values()
fig, ax = plt.subplots(figsize=(9, 7))
top_feat.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
ax.set_title("Top 20 Features — XGBoost", fontsize=13)
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
plt.close(fig)

# Plot 3 — Accuracy comparison
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(['Random Forest', 'XGBoost'],
               [rf_acc*100, xgb_acc*100],
               color=['steelblue', 'seagreen'], edgecolor='white', width=0.4)
for bar, acc in zip(bars, [rf_acc*100, xgb_acc*100]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5, f'{acc:.1f}%',
            ha='center', fontweight='bold')
ax.set_ylim(0, 110)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Comparison")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150)
plt.close(fig)


# ──────────────────────────────────────────────────────────────
#  SAVE MODEL
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SAVING MODEL")
print("=" * 65)

joblib.dump(xgb_model,     os.path.join(MODELS_DIR, "xgboost_model.pkl"))
joblib.dump(scaler,        os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
joblib.dump(label_map,     os.path.join(MODELS_DIR, "label_map.pkl"))

print(f"\n  [OK] xgboost_model.pkl -> {MODELS_DIR}")
print(f"  [OK] scaler.pkl")
print(f"  [OK] feature_names.pkl")
print(f"  [OK] label_map.pkl")


# ──────────────────────────────────────────────────────────────
#  FINAL SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  DONE - FINAL SUMMARY")
print("=" * 65)
print(f"""
  Total rows loaded      : {len(combined):,}
  Feature windows        : {len(features_df):,}
  Features per window    : {len(feature_names)}

  Random Forest accuracy : {rf_acc*100:.2f}%
  XGBoost accuracy       : {xgb_acc*100:.2f}%
  Improvement            : +{(xgb_acc - rf_acc)*100:.2f}%

  All files saved to     : {OUTPUT_DIR}
""")

if xgb_acc >= 0.85:
    print("  [OK] EXCELLENT - above 85%")
elif xgb_acc >= 0.75:
    print("  [OK] GOOD - above 75%, acceptable for major project")
else:
    print("  [WARN] Below 75% - check label distribution above")
    print("     Most likely cause: one class has 90%+ of data")

print("=" * 65)
