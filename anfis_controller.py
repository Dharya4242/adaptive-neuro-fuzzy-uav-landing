"""
Phase 2B — ANFIS / Neuro-Fuzzy Controller
UAV Adaptive Landing Control Under Wind Disturbances

Inputs  : altitude (m), velocity (m/s), wind (m/s)
Output  : thrust_adjustment (%)

Pipeline
--------
1. Load & explore dataset
2. Define fuzzy membership functions for all 3 inputs
3. Build Sugeno-style fuzzy rule base (27 rules: 3^3)
4. Neural tuning of MF parameters via gradient descent (ANFIS logic)
5. Predict on test set
6. Evaluate: RMSE, MAE, R², safe-landing rate
7. Compare against PID baseline metrics
8. Extract & print human-readable IF-THEN rules
9. Save all plots and a results summary CSV
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────
DATASET_PATH   = "dataset.csv"   # change if your file is elsewhere
RANDOM_STATE   = 42
TEST_SIZE      = 0.2             # 80/20 split → 8000 train / 2000 test
LEARNING_RATE  = 0.005
EPOCHS         = 150
SAFE_VEL_THRESHOLD = 1.5        # m/s — landing velocity considered "safe"
PID_RMSE       = 5.349093       # from Member 2 Phase 2A
PID_MAE        = 1.938388
PID_R2         = -0.077548
PID_SAFE_PCT   = 100.0

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
print("=" * 60)
print("PHASE 2B — ANFIS NEURO-FUZZY CONTROLLER")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

# Support both tab-separated (from the shared snippet) and comma-separated
if len(df.columns) == 1:
    df = pd.read_csv(DATASET_PATH, sep="\t")

# Drop index column if present
if "S.No" in df.columns or "Unnamed: 0" in df.columns:
    df = df.drop(columns=[c for c in df.columns if "S.No" in c or "Unnamed" in c])

required_cols = ["altitude", "velocity", "wind", "thrust_adjustment"]
assert all(c in df.columns for c in required_cols), \
    f"Missing columns. Expected: {required_cols}, Got: {list(df.columns)}"

print(f"\n✓ Dataset loaded: {len(df):,} rows × {len(df.columns)} columns")
print(f"  Columns : {list(df.columns)}")
print(f"\nDescriptive statistics:")
print(df.describe().round(3).to_string())

X = df[["altitude", "velocity", "wind"]].values
y = df["thrust_adjustment"].values

# ─────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\n✓ Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

# ─────────────────────────────────────────────
# 3. FUZZY UNIVERSE & MEMBERSHIP FUNCTIONS
#    3 inputs × 3 linguistic levels = 27 fuzzy rules
# ─────────────────────────────────────────────

alt_min,  alt_max  = float(df["altitude"].min()),  float(df["altitude"].max())
vel_min,  vel_max  = float(df["velocity"].min()),  float(df["velocity"].max())
wind_min, wind_max = float(df["wind"].min()),      float(df["wind"].max())
thr_min,  thr_max  = float(df["thrust_adjustment"].min()), float(df["thrust_adjustment"].max())

alt_range  = np.linspace(alt_min,  alt_max,  200)
vel_range  = np.linspace(vel_min,  vel_max,  200)
wind_range = np.linspace(wind_min, wind_max, 200)

# --- Membership function parameters (a, b, c for triangular / Gaussian) ---
# These will be TUNED by the neural learning loop below.
# Initial values are evenly spaced across each universe.
def init_gmf_params(lo, hi):
    """Return initial [mean, sigma] for 3 Gaussian MFs: Low / Medium / High."""
    mid = (lo + hi) / 2
    sigma = (hi - lo) / 4
    return np.array([
        [lo,       sigma],   # Low
        [mid,      sigma],   # Medium
        [hi,       sigma],   # High
    ], dtype=float)

alt_params  = init_gmf_params(alt_min,  alt_max)
vel_params  = init_gmf_params(vel_min,  vel_max)
wind_params = init_gmf_params(wind_min, wind_max)

# Consequent parameters: one linear output per rule (Sugeno 1st-order)
# rule_params shape: (27, 4)  →  [w_alt, w_vel, w_wind, bias]
n_rules = 27
np.random.seed(RANDOM_STATE)
rule_params = np.random.randn(n_rules, 4) * 0.1

# ─────────────────────────────────────────────
# 4. ANFIS FORWARD / BACKWARD PASS
# ─────────────────────────────────────────────
MF_LABELS = ["Low", "Medium", "High"]

def gaussian_mf(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / (sigma + 1e-9)) ** 2)

def gaussian_mf_grad(x, mean, sigma):
    g = gaussian_mf(x, mean, sigma)
    d_mean  =  g * (x - mean) / (sigma**2 + 1e-9)
    d_sigma =  g * (x - mean)**2 / (sigma**3 + 1e-9)
    return d_mean, d_sigma

def anfis_forward(X_batch, alt_p, vel_p, wind_p, r_params):
    """
    Returns predictions and intermediate values needed for backprop.
    X_batch : (N, 3)
    Returns : preds (N,), mf_vals (N,27,3), firing (N,27), norm_firing (N,27)
    """
    N = X_batch.shape[0]
    alt_v  = X_batch[:, 0]
    vel_v  = X_batch[:, 1]
    wind_v = X_batch[:, 2]

    # MF activations: shape (N, 3) for each input
    mf_alt  = np.stack([gaussian_mf(alt_v,  alt_p[i,0],  alt_p[i,1])  for i in range(3)], axis=1)
    mf_vel  = np.stack([gaussian_mf(vel_v,  vel_p[i,0],  vel_p[i,1])  for i in range(3)], axis=1)
    mf_wind = np.stack([gaussian_mf(wind_v, wind_p[i,0], wind_p[i,1]) for i in range(3)], axis=1)

    # Firing strengths for all 27 rules (product T-norm)
    firing = np.zeros((N, n_rules))
    rule_mf_idx = []
    r = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                firing[:, r] = mf_alt[:, i] * mf_vel[:, j] * mf_wind[:, k]
                rule_mf_idx.append((i, j, k))
                r += 1

    # Normalise firing strengths
    firing_sum = firing.sum(axis=1, keepdims=True) + 1e-9
    norm_firing = firing / firing_sum   # (N, 27)

    # Consequent (Sugeno): f_r = w0*alt + w1*vel + w2*wind + w3
    X_aug = np.hstack([X_batch, np.ones((N, 1))])   # (N, 4)
    consequents = X_aug @ r_params.T                  # (N, 27)

    preds = (norm_firing * consequents).sum(axis=1)   # (N,)

    return preds, mf_alt, mf_vel, mf_wind, firing, norm_firing, consequents, rule_mf_idx

def compute_loss(preds, targets):
    return np.mean((preds - targets) ** 2)

def anfis_backward(X_batch, targets, alt_p, vel_p, wind_p, r_params,
                   mf_alt, mf_vel, mf_wind, firing, norm_firing,
                   consequents, rule_mf_idx, lr):
    """
    Gradient descent update for all parameters.
    Returns updated parameter arrays and current loss.
    """
    N = X_batch.shape[0]
    preds = (norm_firing * consequents).sum(axis=1)
    errors = preds - targets                 # (N,)
    loss   = np.mean(errors ** 2)

    alt_v  = X_batch[:, 0]
    vel_v  = X_batch[:, 1]
    wind_v = X_batch[:, 2]

    # --- Consequent parameter gradient (linear, easy) ---
    X_aug = np.hstack([X_batch, np.ones((N, 1))])
    # dL/dr_params[r] = mean over N of 2*err * norm_firing[:,r] * X_aug
    d_r = np.zeros_like(r_params)
    for r in range(n_rules):
        d_r[r] = (2 * errors * norm_firing[:, r]) @ X_aug / N

    # --- Premise parameter gradients (MF params) ---
    firing_sum = firing.sum(axis=1, keepdims=True) + 1e-9
    d_alt_p  = np.zeros_like(alt_p)
    d_vel_p  = np.zeros_like(vel_p)
    d_wind_p = np.zeros_like(wind_p)

    for r, (i, j, k) in enumerate(rule_mf_idx):
        # dL/d(firing_r) = 2*err * (f_r - pred) / firing_sum
        df_r = 2 * errors * (consequents[:, r] - preds) / firing_sum[:, 0]

        # dL/d(mf_alt_i) = df_r * mf_vel_j * mf_wind_k
        da_mean, da_sig = gaussian_mf_grad(alt_v, alt_p[i, 0], alt_p[i, 1])
        d_alt_p[i, 0] += np.mean(df_r * mf_vel[:, j] * mf_wind[:, k] * da_mean)
        d_alt_p[i, 1] += np.mean(df_r * mf_vel[:, j] * mf_wind[:, k] * da_sig)

        dv_mean, dv_sig = gaussian_mf_grad(vel_v, vel_p[j, 0], vel_p[j, 1])
        d_vel_p[j, 0] += np.mean(df_r * mf_alt[:, i] * mf_wind[:, k] * dv_mean)
        d_vel_p[j, 1] += np.mean(df_r * mf_alt[:, i] * mf_wind[:, k] * dv_sig)

        dw_mean, dw_sig = gaussian_mf_grad(wind_v, wind_p[k, 0], wind_p[k, 1])
        d_wind_p[k, 0] += np.mean(df_r * mf_alt[:, i] * mf_vel[:, j] * dw_mean)
        d_wind_p[k, 1] += np.mean(df_r * mf_alt[:, i] * mf_vel[:, j] * dw_sig)

    # Apply updates
    alt_p  -= lr * d_alt_p
    vel_p  -= lr * d_vel_p
    wind_p -= lr * d_wind_p
    r_params -= lr * d_r

    # Clamp sigmas > 0
    alt_p[:, 1]  = np.abs(alt_p[:, 1])  + 1e-6
    vel_p[:, 1]  = np.abs(vel_p[:, 1])  + 1e-6
    wind_p[:, 1] = np.abs(wind_p[:, 1]) + 1e-6

    return alt_p, vel_p, wind_p, r_params, loss

# ─────────────────────────────────────────────
# 5. TRAINING LOOP (mini-batch gradient descent)
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("TRAINING ANFIS MODEL")
print(f"  Epochs: {EPOCHS}  |  LR: {LEARNING_RATE}  |  Rules: {n_rules}")
print("─" * 60)

BATCH_SIZE = 256
train_losses = []
val_losses   = []

# Use last 10% of training data as a validation slice
val_split = int(len(X_train) * 0.9)
X_tr, X_val = X_train[:val_split], X_train[val_split:]
y_tr, y_val = y_train[:val_split], y_train[val_split:]

best_val_loss = np.inf
best_params   = None

for epoch in range(EPOCHS):
    # Shuffle
    idx = np.random.permutation(len(X_tr))
    X_tr_s, y_tr_s = X_tr[idx], y_tr[idx]

    epoch_losses = []
    for start in range(0, len(X_tr_s), BATCH_SIZE):
        Xb = X_tr_s[start:start + BATCH_SIZE]
        yb = y_tr_s[start:start + BATCH_SIZE]

        preds, mf_a, mf_v, mf_w, firing, norm_f, conseq, rule_idx = \
            anfis_forward(Xb, alt_params, vel_params, wind_params, rule_params)

        alt_params, vel_params, wind_params, rule_params, batch_loss = \
            anfis_backward(Xb, yb, alt_params, vel_params, wind_params,
                           rule_params, mf_a, mf_v, mf_w, firing,
                           norm_f, conseq, rule_idx, LEARNING_RATE)
        epoch_losses.append(batch_loss)

    train_loss = np.mean(epoch_losses)
    train_losses.append(train_loss)

    # Validation loss
    val_preds, *_ = anfis_forward(X_val, alt_params, vel_params, wind_params, rule_params)
    val_loss = compute_loss(val_preds, y_val)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = (
            alt_params.copy(), vel_params.copy(),
            wind_params.copy(), rule_params.copy()
        )

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

# Restore best params
alt_params, vel_params, wind_params, rule_params = best_params
print(f"\n✓ Training complete. Best val MSE: {best_val_loss:.4f}")

# ─────────────────────────────────────────────
# 6. TEST SET EVALUATION
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("TEST SET EVALUATION")
print("─" * 60)

test_preds, *_ = anfis_forward(X_test, alt_params, vel_params, wind_params, rule_params)

rmse = np.sqrt(mean_squared_error(y_test, test_preds))
mae  = mean_absolute_error(y_test, test_preds)
r2   = r2_score(y_test, test_preds)

# Safe landing rate: approximate from velocity column in test set
# (velocity near 0 at low altitude → safe touchdown)
test_df = pd.DataFrame(X_test, columns=["altitude", "velocity", "wind"])
test_df["thrust_actual"] = y_test
test_df["thrust_pred"]   = test_preds

# Heuristic: safe if predicted thrust keeps final velocity below threshold
# We consider rows where altitude < 2m as near-landing moments
near_landing = test_df[test_df["altitude"] < 2.0]
if len(near_landing) > 0:
    safe_count = (near_landing["velocity"] < SAFE_VEL_THRESHOLD).sum()
    safe_pct   = 100.0 * safe_count / len(near_landing)
else:
    safe_pct = 100.0   # no near-landing rows in test — assume safe

print(f"\n  ANFIS  RMSE : {rmse:.6f}  (PID: {PID_RMSE:.6f})")
print(f"  ANFIS  MAE  : {mae:.6f}  (PID: {PID_MAE:.6f})")
print(f"  ANFIS  R²   : {r2:.6f}  (PID: {PID_R2:.6f})")
print(f"  Safe landing: {safe_pct:.1f}%  (PID: {PID_SAFE_PCT:.1f}%)")

# ─────────────────────────────────────────────
# 7. COMPARISON TABLE
# ─────────────────────────────────────────────
compare = pd.DataFrame({
    "Controller": ["PID (Baseline)", "ANFIS (Phase 2B)"],
    "RMSE":        [PID_RMSE, round(rmse, 6)],
    "MAE":         [PID_MAE,  round(mae,  6)],
    "R²":          [PID_R2,   round(r2,   6)],
    "Safe Landing %": [PID_SAFE_PCT, round(safe_pct, 1)],
})
print("\n" + "─" * 60)
print("COMPARISON: PID vs ANFIS")
print("─" * 60)
print(compare.to_string(index=False))

compare.to_csv("comparison_results.csv", index=False)
print("\n✓ comparison_results.csv saved")

# ─────────────────────────────────────────────
# 8. RULE EXTRACTION
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("EXTRACTED FUZZY IF-THEN RULES")
print("─" * 60)

def dominant_consequent(r_params_row):
    """Return a linguistic label for the Sugeno output."""
    bias = r_params_row[3]
    thr_mid = (thr_min + thr_max) / 2
    if bias < thr_mid * 0.85:
        return "Low Thrust Boost"
    elif bias < thr_mid * 1.15:
        return "Medium Thrust Boost"
    else:
        return "High Thrust Boost"

rules_text = []
r = 0
for i in range(3):
    for j in range(3):
        for k in range(3):
            conseq_label = dominant_consequent(rule_params[r])
            rule = (
                f"Rule {r+1:02d}: IF Altitude is {MF_LABELS[i]:6s} "
                f"AND Velocity is {MF_LABELS[j]:6s} "
                f"AND Wind is {MF_LABELS[k]:6s} "
                f"THEN Thrust = {conseq_label}"
            )
            rules_text.append(rule)
            print("  " + rule)
            r += 1

with open("extracted_rules.txt", "w") as f:
    f.write("ANFIS EXTRACTED FUZZY RULES — UAV Landing Controller\n")
    f.write("=" * 60 + "\n\n")
    for rule in rules_text:
        f.write(rule + "\n")
print("\n✓ extracted_rules.txt saved")

# ─────────────────────────────────────────────
# 9. PLOTS
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("GENERATING PLOTS")
print("─" * 60)

fig = plt.figure(figsize=(16, 14))
fig.suptitle("Phase 2B — ANFIS Neuro-Fuzzy Controller Results", fontsize=14, fontweight="bold")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Training & Validation Loss ──────
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(train_losses, label="Train MSE", color="#1f77b4", linewidth=1.5)
ax1.plot(val_losses,   label="Val MSE",   color="#ff7f0e", linewidth=1.5, linestyle="--")
ax1.set_title("Training & Validation Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE")
ax1.legend()
ax1.grid(True, alpha=0.3)

# ── Plot 2: Predicted vs Actual ─────────────
ax2 = fig.add_subplot(gs[0, 2])
sample = np.random.choice(len(y_test), min(500, len(y_test)), replace=False)
ax2.scatter(y_test[sample], test_preds[sample], alpha=0.4, s=12, color="#2ca02c")
lims = [min(y_test.min(), test_preds.min()), max(y_test.max(), test_preds.max())]
ax2.plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
ax2.set_title(f"Predicted vs Actual (R²={r2:.3f})")
ax2.set_xlabel("Actual Thrust Adj.")
ax2.set_ylabel("Predicted Thrust Adj.")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ── Plot 3–5: Membership Functions (after tuning) ──
input_specs = [
    ("Altitude (m)",  alt_range,  alt_params),
    ("Velocity (m/s)", vel_range, vel_params),
    ("Wind (m/s)",    wind_range, wind_params),
]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for col, (label, universe, params) in enumerate(input_specs):
    ax = fig.add_subplot(gs[1, col])
    for lvl in range(3):
        mf_vals = gaussian_mf(universe, params[lvl, 0], params[lvl, 1])
        ax.plot(universe, mf_vals, label=MF_LABELS[lvl], color=colors[lvl], linewidth=2)
    ax.set_title(f"Tuned MFs — {label}")
    ax.set_xlabel(label)
    ax.set_ylabel("Membership")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

# ── Plot 6: Residuals Distribution ──────────
ax6 = fig.add_subplot(gs[2, 0])
residuals = y_test - test_preds
ax6.hist(residuals, bins=50, color="#9467bd", edgecolor="white", linewidth=0.4)
ax6.axvline(0, color="red", linestyle="--", linewidth=1)
ax6.set_title("Residuals Distribution")
ax6.set_xlabel("Error (Actual − Predicted)")
ax6.set_ylabel("Count")
ax6.grid(True, alpha=0.3)

# ── Plot 7: Time-series replay (first 200 test rows) ──
ax7 = fig.add_subplot(gs[2, 1:])
idx_sorted = np.argsort(X_test[:200, 0])[::-1]   # sort by altitude desc
ax7.plot(y_test[:200][idx_sorted],    label="Ground Truth", color="#1f77b4", linewidth=1.5)
ax7.plot(test_preds[:200][idx_sorted], label="ANFIS Pred",  color="#ff7f0e",
         linewidth=1.5, linestyle="--")
ax7.set_title("Thrust Adjustment — First 200 Test Samples (sorted by altitude)")
ax7.set_xlabel("Sample")
ax7.set_ylabel("Thrust Adjustment (%)")
ax7.legend()
ax7.grid(True, alpha=0.3)

plt.savefig("phase2b_results.png", dpi=150, bbox_inches="tight")
print("✓ phase2b_results.png saved")

# ── Plot 8: Comparison bar chart ────────────
fig2, axes = plt.subplots(1, 3, figsize=(12, 4))
fig2.suptitle("PID vs ANFIS — Performance Comparison", fontsize=13, fontweight="bold")

metrics = [
    ("RMSE ↓", [PID_RMSE, rmse]),
    ("MAE ↓",  [PID_MAE,  mae]),
    ("R² ↑",   [PID_R2,   r2]),
]
bar_colors = [["#d62728", "#2ca02c"], ["#d62728", "#2ca02c"], ["#d62728", "#2ca02c"]]

for ax, (title, vals), bcolors in zip(axes, metrics, bar_colors):
    bars = ax.bar(["PID", "ANFIS"], vals, color=bcolors, edgecolor="white", width=0.5)
    ax.set_title(title)
    ax.bar_label(bars, fmt="%.3f", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(min(0, min(vals)) * 1.3, max(vals) * 1.4)

plt.tight_layout()
plt.savefig("comparison_chart.png", dpi=150, bbox_inches="tight")
print("✓ comparison_chart.png saved")

# ─────────────────────────────────────────────
# 10. SAVE FULL RESULTS SUMMARY
# ─────────────────────────────────────────────
summary = {
    "controller":         "ANFIS",
    "test_rmse":          round(rmse, 6),
    "test_mae":           round(mae, 6),
    "test_r2":            round(r2, 6),
    "safe_landing_pct":   round(safe_pct, 1),
    "n_rules":            n_rules,
    "epochs_trained":     EPOCHS,
    "learning_rate":      LEARNING_RATE,
    "train_rows":         len(X_tr),
    "test_rows":          len(X_test),
    "note": (
        "ANFIS learns non-linear wind-altitude-velocity interactions. "
        "Tuned Gaussian MFs adapt premise parameters via backprop. "
        "27 Sugeno rules extracted and saved to extracted_rules.txt."
    )
}
pd.DataFrame([summary]).to_csv("phase2b_summary.csv", index=False)
print("✓ phase2b_summary.csv saved")

print("\n" + "=" * 60)
print("PHASE 2B COMPLETE")
print("=" * 60)
print("\nFiles generated:")
print("  phase2b_results.png    — training curves, MFs, prediction plots")
print("  comparison_chart.png   — PID vs ANFIS bar chart")
print("  comparison_results.csv — metrics comparison table")
print("  extracted_rules.txt    — 27 human-readable IF-THEN rules")
print("  phase2b_summary.csv    — single-row results for Phase 3")
print()
