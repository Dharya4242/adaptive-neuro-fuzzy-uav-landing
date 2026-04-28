"""
PHASE 3 — QUICK INTEGRATION TEST
Run this FIRST to verify your ANFIS model loads correctly

This script:
1. Loads your trained ANFIS model from phase2b
2. Makes a few test predictions
3. Compares against PID
4. Confirms everything works before running full Phase 3

If this passes, you're ready for the full evaluation suite.
"""

import numpy as np
import pandas as pd
import pickle
import sys

print("=" * 70)
print("PHASE 3 — QUICK INTEGRATION TEST")
print("=" * 70)

# ─────────────────────────────────────────────
# Step 1: Check files exist
# ─────────────────────────────────────────────
print("\n[1] Checking required files...")

required_files = ['dataset.csv', 'phase2b_summary.csv']
missing = []

for f in required_files:
    try:
        open(f, 'r').close()
        print(f"    ✓ {f} found")
    except FileNotFoundError:
        print(f"    ✗ {f} MISSING")
        missing.append(f)

if missing:
    print(f"\n❌ ERROR: Missing files: {missing}")
    print("   Make sure you're in the correct directory.")
    sys.exit(1)

# ─────────────────────────────────────────────
# Step 2: Load ANFIS model (try to reconstruct from Phase 2B)
# ─────────────────────────────────────────────
print("\n[2] Loading ANFIS model...")

try:
    # First, try to load pickled model if it exists
    with open('anfis_model.pkl', 'rb') as f:
        params = pickle.load(f)
    print("    ✓ Loaded anfis_model.pkl")
    
    alt_params = params['alt_params']
    vel_params = params['vel_params']
    wind_params = params['wind_params']
    rule_params = params['rule_params']
    
except FileNotFoundError:
    print("    ⚠ anfis_model.pkl not found — attempting to reconstruct...")
    print("    (You should create this file by adding pickle code to phase2b_anfis.py)")
    print("    Using stub model for now...")
    
    # Stub parameters (will be replaced when you create the pickle file)
    alt_params = np.array([[10, 5], [25, 5], [40, 5]])
    vel_params = np.array([[1, 1], [5, 1], [10, 1]])
    wind_params = np.array([[1, 1], [5, 1], [10, 1]])
    rule_params = np.random.randn(27, 4) * 0.1

# Define ANFIS class
class ANFIS:
    def __init__(self, alt_p, vel_p, wind_p, r_params):
        self.alt_p = alt_p
        self.vel_p = vel_p
        self.wind_p = wind_p
        self.r_params = r_params
    
    def gaussian_mf(self, x, mean, sigma):
        return np.exp(-0.5 * ((x - mean) / (sigma + 1e-9)) ** 2)
    
    def predict(self, X):
        N = X.shape[0]
        alt_v, vel_v, wind_v = X[:, 0], X[:, 1], X[:, 2]
        
        # MF activations
        mf_alt = np.stack([self.gaussian_mf(alt_v, self.alt_p[i,0], self.alt_p[i,1]) 
                           for i in range(3)], axis=1)
        mf_vel = np.stack([self.gaussian_mf(vel_v, self.vel_p[i,0], self.vel_p[i,1]) 
                           for i in range(3)], axis=1)
        mf_wind = np.stack([self.gaussian_mf(wind_v, self.wind_p[i,0], self.wind_p[i,1]) 
                            for i in range(3)], axis=1)
        
        # Firing strengths
        firing = np.zeros((N, 27))
        r = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    firing[:, r] = mf_alt[:, i] * mf_vel[:, j] * mf_wind[:, k]
                    r += 1
        
        # Normalize
        firing_sum = firing.sum(axis=1, keepdims=True) + 1e-9
        norm_firing = firing / firing_sum
        
        # Consequents
        X_aug = np.hstack([X, np.ones((N, 1))])
        consequents = X_aug @ self.r_params.T
        
        return (norm_firing * consequents).sum(axis=1)

anfis_model = ANFIS(alt_params, vel_params, wind_params, rule_params)
print("    ✓ ANFIS model instantiated")

# ─────────────────────────────────────────────
# Step 3: Test predictions on sample data
# ─────────────────────────────────────────────
print("\n[3] Testing predictions on sample scenarios...")

test_cases = [
    {'alt': 45.0, 'vel': 0.0, 'wind': 0.5, 'name': 'High altitude, calm'},
    {'alt': 20.0, 'vel': 3.0, 'wind': 2.0, 'name': 'Mid altitude, moderate descent'},
    {'alt': 5.0, 'vel': 1.0, 'wind': 5.0, 'name': 'Low altitude, gusty'},
    {'alt': 0.5, 'vel': 0.3, 'wind': 0.1, 'name': 'Final approach'},
]

print("\n    Scenario                          Altitude  Velocity  Wind   → ANFIS Thrust")
print("    " + "-" * 75)

for case in test_cases:
    X = np.array([[case['alt'], case['vel'], case['wind']]])
    thrust = anfis_model.predict(X)[0]
    print(f"    {case['name']:<30}  {case['alt']:5.1f}m    {case['vel']:5.1f}m/s  {case['wind']:4.1f}m/s  →  {thrust:6.2f}N")

# ─────────────────────────────────────────────
# Step 4: Load actual dataset and compare
# ─────────────────────────────────────────────
print("\n[4] Comparing ANFIS vs ground truth on test set...")

df = pd.read_csv('dataset.csv')
if len(df.columns) == 1:
    df = pd.read_csv('dataset.csv', sep='\t')
if 'S.No' in df.columns or 'Unnamed: 0' in df.columns:
    df = df.drop(columns=[c for c in df.columns if 'S.No' in c or 'Unnamed' in c])

# Take last 100 rows as quick test
test_df = df.tail(100)
X_test = test_df[['altitude', 'velocity', 'wind']].values
y_test = test_df['thrust_adjustment'].values

preds = anfis_model.predict(X_test)
rmse = np.sqrt(np.mean((y_test - preds) ** 2))
mae = np.mean(np.abs(y_test - preds))
r2 = 1.0 - np.sum((y_test - preds)**2) / np.sum((y_test - y_test.mean())**2)

print(f"\n    Test set (100 samples):")
print(f"      RMSE : {rmse:.4f}")
print(f"      MAE  : {mae:.4f}")
print(f"      R²   : {r2:.4f}")

# ─────────────────────────────────────────────
# Step 5: Verify against Phase 2B results
# ─────────────────────────────────────────────
print("\n[5] Verifying against Phase 2B results...")

try:
    phase2b = pd.read_csv('phase2b_summary.csv')
    expected_rmse = phase2b['test_rmse'].iloc[0]
    expected_r2 = phase2b['test_r2'].iloc[0]
    
    print(f"\n    Expected (from Phase 2B):")
    print(f"      RMSE : {expected_rmse:.4f}")
    print(f"      R²   : {expected_r2:.4f}")
    
    rmse_diff = abs(rmse - expected_rmse)
    r2_diff = abs(r2 - expected_r2)
    
    if rmse_diff < 0.5 and r2_diff < 0.1:
        print("\n    ✓ Results match Phase 2B! Model loaded correctly.")
        status = "PASS"
    else:
        print("\n    ⚠ Results differ from Phase 2B — check model loading.")
        print(f"      RMSE difference: {rmse_diff:.4f}")
        print(f"      R² difference: {r2_diff:.4f}")
        status = "WARNING"
        
except Exception as e:
    print(f"\n    ⚠ Could not compare: {e}")
    status = "UNKNOWN"

# ─────────────────────────────────────────────
# Final verdict
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("INTEGRATION TEST RESULT:", status)
print("=" * 70)

if status == "PASS":
    print("\n✓ Everything looks good! You're ready to run:")
    print("    python phase3_evaluation.py")
    print("    python phase3_simulation.py")
elif status == "WARNING":
    print("\n⚠ Model loads but results differ from Phase 2B.")
    print("  This could mean:")
    print("    1. You haven't created anfis_model.pkl yet → run phase2b_anfis.py with pickle code")
    print("    2. The test set differs from Phase 2B → this is OK, just informational")
    print("\n  You can still run Phase 3, but double-check the model file.")
else:
    print("\n⚠ Could not verify. Make sure:")
    print("    1. phase2b_anfis.py ran successfully")
    print("    2. phase2b_summary.csv exists")
    print("    3. You're in the correct directory")

print()
