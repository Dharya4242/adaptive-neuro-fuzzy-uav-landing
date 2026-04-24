# Phase 2B — ANFIS / Neuro-Fuzzy UAV Landing Model

## Overview

Build a **pure-Python ANFIS** (Adaptive Neuro-Fuzzy Inference System) that:
1. Trains on `dataset.csv` to predict `thrust_adjustment` from `(altitude, velocity, wind)`
2. Matches or beats the PID baseline on **both** RMSE and safe-landing rate simultaneously
3. Extracts human-readable IF-THEN fuzzy rules for academic explainability
4. Produces a training loss curve and benchmark metrics for Phase 3 handoff

The implementation will use **`scikit-fuzzy`** for fuzzy membership functions and **NumPy** gradient descent for the "neuro" (parameter tuning) layer — no heavy ML frameworks needed.

---

## Architecture: ANFIS (Takagi-Sugeno Type)

```
Inputs:  altitude  [0–50 m]
         velocity  [0–20 m/s]   (accounts for fast descents seen in dataset)
         wind      [0–15 m/s]

MFs:     3 Gaussian MFs per input (Low / Medium / High)
         → 3³ = 27 rules

Layers:
  L1 — Fuzzification    : μ_ij(x)  (Gaussian MFs)
  L2 — Rule strength    : w_r = Π μ_ij   (product T-norm)
  L3 — Normalisation    : w̄_r = w_r / Σ w_r
  L4 — Consequent       : f_r = p_r·alt + q_r·vel + s_r·wind + c_r  (linear)
  L5 — Aggregation      : output = Σ w̄_r · f_r

Trainable params:
  Premise (MF): centre μ_ij, width σ_ij   → 3 inputs × 3 MFs × 2 = 18 params
  Consequent:   [p, q, s, c] per rule      → 27 rules × 4 = 108 params
  Total:        126 parameters
```

The consequent layer uses **least-squares regression** (closed-form, fast) at each epoch, while the premise layer uses **backpropagation / gradient descent** (chain rule through the Sugeno output). This is the classic hybrid ANFIS learning algorithm.

---

## Proposed Changes

### Single new file

#### [NEW] `anfis_mem3.py`

**Sections:**

| Section | Description |
|---|---|
| Constants | Match `DT`, `GRAVITY`, `MASS`, `SAFE_VEL`, `MAX_STEPS` from Phase 1 & 2A |
| `GaussianMF` class | Vectorised Gaussian with gradient for premise backprop |
| `ANFIS` class | Full 5-layer forward pass + hybrid learning (LS consequent + SGD premise) |
| `train()` | Epoch loop, train/val RMSE tracking, early stopping |
| `extract_rules()` | Print IF-THEN rules with linguistic labels |
| `run_episode()` | Physics sim (identical to PID version for fair comparison) |
| `eval_safe_landing()` | 300-episode safe landing rate |
| `eval_gust()` | Gust response metric |
| `plot_training_curve()` | Loss curve (train vs validation RMSE) |
| `plot_anfis_evaluation()` | 4-panel dashboard (predicted vs true, residuals, gust, summary) |
| `plot_sample_landing()` | Sample landing trace — ANFIS thrust vs ground truth |
| `save_metrics()` | Append ANFIS row to `anfis_metrics.csv` |
| `main()` | Orchestrates all of the above |

**Key design choices:**
- **Hybrid learning**: Consequents solved by LS (fast convergence), premises tuned by gradient (adaptive)
- **Batch mini-batch SGD** with batch size 256, learning rate 0.005, cosine LR decay
- **Early stopping** (patience = 15 epochs) to prevent overfitting
- **Normalisation**: StandardScaler on inputs so MF centres init in [−2, +2] range
- **Rule extraction**: After training, un-scale MF centres back to physical units and print linguistic bins

---

## Outputs (Deliverables)

| File | Description |
|---|---|
| `anfis_mem3.py` | Full ANFIS implementation |
| `anfis_training_curve.png` | Train vs validation RMSE across epochs |
| `anfis_evaluation.png` | 4-panel evaluation dashboard |
| `anfis_sample_landing.png` | Sample landing: ANFIS output vs ground truth |
| `extracted_rules.txt` | 27 IF-THEN rules extracted from trained model |
| `anfis_metrics.csv` | ANFIS benchmark metrics for Phase 3 handoff |

---

## Verification Plan

### Quantitative targets (to beat PID baseline)

| Metric | PID (baseline) | ANFIS target |
|---|---|---|
| Test RMSE | 5.35 N | < 2.0 N |
| Test R² | −0.077 | > 0.90 |
| Safe landing rate | 100% | ≥ 100% |
| Gust response time | 0.0 s | ≤ 0.1 s |

### Automated checks in `main()`
1. Assert `test_rmse < pid_rmse` (ANFIS beats PID on accuracy)
2. Assert `safe_landing_pct >= 95.0` (ANFIS maintains safety)
3. Print pass/fail against each target

### Visual checks
- Training curve converges (no divergence or plateau at epoch 1)
- Predicted vs true scatter tight around identity line (R² > 0.90)
- 27 rules extracted and saved to `extracted_rules.txt`

---

## Open Questions

> [!IMPORTANT]
> **Epoch count**: Default plan is 150 epochs with early stopping (≈ 60 s runtime on a laptop CPU). If you want faster iteration set it lower (e.g., 50 epochs); if you want higher accuracy set it higher (e.g., 300). Let me know if you have a preference.

> [!NOTE]
> **Number of MFs per input**: The workflow specifies 3 (Low/Medium/High) = 27 rules. This is what will be implemented. Increasing to 5 MFs gives 125 rules and potentially better accuracy but slower training.

> [!NOTE]
> **Safe-landing simulation**: The ANFIS output will be fed into the same physics simulator used by PID (Member 2's `run_episode` logic), re-implemented internally so `anfis_mem3.py` is fully self-contained.
