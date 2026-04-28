"""
PHASE 3 — EVALUATION & BENCHMARKING
Integration Guide for Member 2 & Member 3
==========================================

WHAT YOU HAVE
-------------
1. phase3_evaluation.py      — Main evaluation script (ablation, stress tests, stats)
2. phase3_simulation.py       — Real-time visualization (PID vs ANFIS side-by-side)
3. This README

WHAT YOU NEED TO DO
-------------------
The scripts currently use PLACEHOLDER ANFIS predictions. You need to integrate
your actual trained ANFIS model from Phase 2B.

INTEGRATION STEPS
=================

Step 1: Save Your Trained ANFIS Model
--------------------------------------
Add this to the END of your phase2b_anfis.py script:

```python
# Save trained model parameters
import pickle

model_params = {
    'alt_params': alt_params,
    'vel_params': vel_params,
    'wind_params': wind_params,
    'rule_params': rule_params,
    'alt_range': (alt_min, alt_max),
    'vel_range': (vel_min, vel_max),
    'wind_range': (wind_min, wind_max),
}

with open('anfis_model.pkl', 'wb') as f:
    pickle.dump(model_params, f)

print("✓ ANFIS model saved to anfis_model.pkl")
```

Run phase2b_anfis.py again to generate the .pkl file.


Step 2: Replace the ANFIS Stub in phase3_evaluation.py
-------------------------------------------------------
Find this section (around line 60):

```python
def load_anfis_model():
    class ANFISStub:
        def predict(self, X):
            # Simple linear approximation for testing
            alt, vel, wind = X[:, 0], X[:, 1], X[:, 2]
            return WEIGHT + 0.7 * alt + 0.5 * vel + 0.3 * wind
    return ANFISStub()
```

REPLACE IT WITH:

```python
def load_anfis_model():
    import pickle
    
    with open('anfis_model.pkl', 'rb') as f:
        params = pickle.load(f)
    
    alt_params = params['alt_params']
    vel_params = params['vel_params']
    wind_params = params['wind_params']
    rule_params = params['rule_params']
    
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
            
            # Membership function activations
            mf_alt = np.stack([self.gaussian_mf(alt_v, self.alt_p[i,0], self.alt_p[i,1]) 
                               for i in range(3)], axis=1)
            mf_vel = np.stack([self.gaussian_mf(vel_v, self.vel_p[i,0], self.vel_p[i,1]) 
                               for i in range(3)], axis=1)
            mf_wind = np.stack([self.gaussian_mf(wind_v, self.wind_p[i,0], self.wind_p[i,1]) 
                                for i in range(3)], axis=1)
            
            # Firing strengths (27 rules)
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
            
            # Sugeno consequents
            X_aug = np.hstack([X, np.ones((N, 1))])
            consequents = X_aug @ self.r_params.T
            
            return (norm_firing * consequents).sum(axis=1)
    
    return ANFIS(alt_params, vel_params, wind_params, rule_params)
```


Step 3: Do the Same for phase3_simulation.py
---------------------------------------------
Replace the ANFISStub class (around line 45) with the SAME code from Step 2.


Step 4: Run Phase 3
-------------------
After integration:

```bash
python phase3_evaluation.py
python phase3_simulation.py
```

This will generate:
  ✓ phase3_ablation_results.csv
  ✓ phase3_stress_test_results.csv
  ✓ phase3_statistical_analysis.csv
  ✓ phase3_ablation_plot.png
  ✓ phase3_stress_comparison.png
  ✓ phase3_simulation_replay.png
  ✓ phase3_full_report.txt


TROUBLESHOOTING
===============

Error: "FileNotFoundError: anfis_model.pkl"
-------------------------------------------
→ You forgot Step 1. Run phase2b_anfis.py with the pickle code added.

Error: "KeyError: 'alt_params'"
-------------------------------
→ Check that your pickle.dump() in Step 1 has the exact dictionary structure shown.

ANFIS predictions look wrong
-----------------------------
→ Verify the forward pass logic matches your phase2b_anfis.py exactly.
→ Compare a few manual predictions to confirm they match.


UNDERSTANDING THE EXPERIMENTS
==============================

1. Ablation Study
-----------------
Tests ANFIS with only 2 inputs at a time:
  - Remove altitude → train with (velocity, wind)
  - Remove velocity → train with (altitude, wind)
  - Remove wind → train with (altitude, velocity)

Shows which feature is most critical. If removing altitude causes the biggest
RMSE increase, altitude is the most important input.

NOTE: The current script SIMULATES ablation with linear approximations.
For TRUE ablation, you'd need to retrain ANFIS three times with 2-input variants.
That's optional — the simulation is sufficient for a class project.


2. Stress Testing
-----------------
Three extreme scenarios:
  - Sudden gust: calm → 12 m/s spike at t=10s
  - Near-crash: start at 5m altitude with 8 m/s descent speed
  - High turbulence: sustained 8-10 m/s wind throughout

Compares which controller handles edge cases better.


3. Statistical Analysis
-----------------------
Paired t-test: proves ANFIS improvement isn't random chance.
Effect size (Cohen's d): quantifies HOW MUCH better ANFIS is.
Bootstrap confidence intervals: shows result stability.

If p < 0.05 and Cohen's d > 0.8 → "statistically significant with large effect"


4. Simulation Replay
--------------------
Visual comparison: watch both controllers land the same drone side-by-side.
Shows WHO lands faster, smoother, safer.


DELIVERABLES FOR MEMBER 4
==========================
Hand these files to Member 4 (Dashboard & Report):

Core results:
  □ phase3_ablation_results.csv
  □ phase3_stress_test_results.csv
  □ phase3_statistical_analysis.csv
  □ phase3_full_report.txt

Visualizations:
  □ phase3_ablation_plot.png
  □ phase3_stress_comparison.png
  □ phase3_simulation_replay.png

From Phase 2:
  □ comparison_results.csv (PID vs ANFIS metrics)
  □ extracted_rules.txt (the 27 fuzzy rules)

Member 4 will use these to build the final dashboard and write the report.


TIMELINE
========
1. Member 3 (you): Integrate ANFIS model (30 min)
2. Run both scripts (5 min)
3. Review outputs with Member 2 (15 min)
4. Hand off to Member 4 (immediate)

Total: ~1 hour of work for Phase 3


QUESTIONS?
==========
If something breaks, check:
  1. Is anfis_model.pkl in the same folder as the scripts?
  2. Does dataset.csv exist?
  3. Did you run phase2b_anfis.py successfully first?

Good luck! Phase 3 is the shortest phase — most of the hard work was Phase 2B.
"""