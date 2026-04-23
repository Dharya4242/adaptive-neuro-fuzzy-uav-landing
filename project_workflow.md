# Neuro-Project

![uav_project_architecture.svg](uav_project_architecture.svg)

Here's a complete implementation blueprint for your project, designed to justify 4 members with clear deliverables for each.

---

## What we're adding to make it grade-worthy

Before the plan, here are three additions that will clearly differentiate your project:

1. **Comparison baseline** — implement a plain PID controller alongside ANFIS, then benchmark both. This gives you a "before vs after" story that evaluators love.
2. **Visualization dashboard** — a real-time plot showing altitude, wind, and thrust adjustments over time. Makes the project feel like a finished system, not just a script.
3. **Rule extraction** — after training ANFIS, explicitly extract and display the fuzzy IF-THEN rules it learned. This is the academic punchline of neuro-fuzzy: explainability.

---

## Full Implementation Plan

### Phase 1 — Data Simulation (Member 1)

The goal is a realistic synthetic dataset. Write a Python script that simulates a drone descending over ~10 seconds in discrete time steps (e.g., 0.05s intervals). Each row in the dataset captures one moment:

**Physics logic:**

- `altitude(t+1) = altitude(t) - velocity(t) * dt`
- `velocity(t+1) = velocity(t) + (gravity - thrust) * dt`
- `wind(t)` = generated using a correlated noise model (not purely random — use a low-pass filtered Gaussian so gusts have duration, like real wind)

**Output column (the label):** `thrust_adjustment` — computed as the PD-law correction needed to keep the descent velocity safe given the current wind. This gives you a ground-truth label to train against.

Generate ~10,000 rows covering varied scenarios: high-altitude stable landings, low-altitude gusty conditions, and near-crash recovery situations. Save as `dataset.csv`.

**Deliverable:** `data_generator.py` + `dataset.csv` + a short plot showing sample trajectories.

---

### Phase 2A — PID Baseline Controller (Member 2)

Implement a standard PID controller in Python that takes altitude, velocity, and wind as inputs and outputs a thrust adjustment. Tune the P, I, D gains manually (or use a grid search). Run it on a test split of the dataset and record performance metrics: RMSE on thrust prediction, percentage of "safe landings" (defined as: altitude reaches 0 with velocity below a threshold), and response time to a sudden gust.

This baseline exists purely to be beaten by ANFIS. The comparison is the academic contribution.

**Deliverable:** `pid_controller.py` + performance metrics CSV.

---

### Phase 2B — ANFIS / Neuro-Fuzzy Model (Member 3)

This is the technical core. Two routes depending on your stack:

**Option A — Python (`scikit-fuzzy` + neural tuning):**

- Define 3 input universes: altitude [0–50m], velocity [0–5 m/s], wind [0–15 m/s]
- Define membership functions for each: 3 fuzzy sets each (Low/Medium/High) = 27 possible rules
- Use the dataset to tune membership function parameters via gradient descent (this is the "neuro" in neuro-fuzzy)
- Extract the learned rules and print them as IF-THEN statements

**Option B — MATLAB ANFIS Designer:**

- Load the CSV, use `anfis()` function with `genfis()` for automatic rule generation
- Train for ~100 epochs, monitor training/validation error
- Export the FIS (Fuzzy Inference System) and inspect rules in the Fuzzy Logic Designer GUI

**Deliverable:** trained model file + `extracted_rules.txt` (the IF-THEN rules) + training loss curve.

---

### Phase 3 — Evaluation & Benchmarking (Members 2 + 3 jointly)

Run both controllers on the same held-out test scenarios (at least 500 landing simulations). Create a comparison table:

| Metric | PID | ANFIS |
| --- | --- | --- |
| RMSE (thrust prediction) | — | — |
| Safe landing rate (%) | — | — |
| Avg response to gust (ms) | — | — |
| Explainability | None | IF-THEN rules |

Also run an ablation: test ANFIS with only 2 inputs (remove wind) to show that the wind feature actually matters. This single experiment adds a lot of academic weight.

**Deliverable:** `evaluation.py` + comparison table + ablation result.

---

### Phase 4 — Dashboard & Report (Member 4)

Build a simulation replay dashboard using Matplotlib or Plotly that animates a drone landing in real time. Show three synchronized subplots: altitude over time, wind intensity over time, and thrust adjustment over time — with the ANFIS output plotted against the ground truth. Add a slider to scrub through scenarios.

For the report, structure it as: Problem → Related Work (briefly mention PID limitations) → Methodology → Results → Extracted Rules → Conclusion. The fuzzy rule table is your Discussion section — write two paragraphs interpreting what the model learned (e.g., "The model assigns maximum thrust only when both altitude is critically low AND wind exceeds 8 m/s, suggesting a conservative safety-first policy").

**Deliverable:** `dashboard.py` + final report PDF + slide deck (10–12 slides).

---

## Work Division Summary

| Member | Role | Core Deliverable |
| --- | --- | --- |
| Member 1 | Data Engineer | `data_generator.py`, 10k-row dataset, trajectory plots |
| Member 2 | Control Systems | PID baseline, metrics, evaluation script |
| Member 3 | ML / Fuzzy Systems | ANFIS model, rule extraction, training curves |
| Member 4 | Visualization & Docs | Dashboard, report, slide deck |

Members 2 and 3 share Phase 3 (benchmarking) — this is intentional and gives them a natural overlap to collaborate and cross-check each other's numbers.

---

## Tools you'll need

For Python: `numpy`, `pandas`, `matplotlib`, `scikit-fuzzy`, `scikit-learn` (for train/test split and RMSE), and optionally `plotly` for the dashboard. If any member prefers MATLAB, the ANFIS portion translates directly — the Fuzzy Logic Toolbox handles everything in Phase 2B with a GUI.