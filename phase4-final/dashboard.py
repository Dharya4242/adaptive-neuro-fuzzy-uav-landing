"""
Phase 4 — Interactive Dashboard: ANFIS vs PID UAV Landing Simulation
Run from the phase4-final/ directory: python dashboard.py
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
import os

# ─── Physics (matches Phase 3) ────────────────────────────────────────────────
MASS   = 2.0   # kg  (Phase 3 value)
WEIGHT = MASS * 9.81
DT     = 0.05  # s
SAFE_VEL = 0.5 # m/s landing threshold

# ─── PID controller (matches Phase 3 gains) ───────────────────────────────────
def pid_thrust(alt, vel, wind, integral, prev_err, dt=DT):
    Ka, Kp, Ki, Kd, Kw = 0.18, 7.0, 0.5, 0.3, 0.15
    v_min, v_max = 0.3, 5.5
    target_vel = np.clip(Ka * alt, v_min, v_max)
    err = vel - target_vel
    integral = np.clip(integral + err * dt, -40.0, 40.0)
    deriv = (err - prev_err) / dt
    thrust = WEIGHT - Kp * err - Ki * integral - Kd * deriv + Kw * wind
    thrust = np.clip(thrust, 0.0, 3.0 * WEIGHT)
    return thrust, integral, err

# ─── ANFIS loader ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_anfis():
    path = os.path.join(SCRIPT_DIR, "anfis_model.pkl")
    with open(path, "rb") as f:
        m = pickle.load(f)

    alt_params  = m["alt_params"]   # shape (3, 2): [[c, s], ...]
    vel_params  = m["vel_params"]
    wind_params = m["wind_params"]
    alt_c,  alt_s  = alt_params[:, 0],  alt_params[:, 1]
    vel_c,  vel_s  = vel_params[:, 0],  vel_params[:, 1]
    wind_c, wind_s = wind_params[:, 0], wind_params[:, 1]
    rule_params     = m["rule_params"]
    alt_range  = m["alt_range"]
    vel_range  = m["vel_range"]
    wind_range = m["wind_range"]

    def gauss(x, c, s):
        return np.exp(-0.5 * ((x - c) / (s + 1e-9)) ** 2)

    def predict(X):
        X = np.atleast_2d(X)
        alts  = np.clip(X[:, 0], alt_range[0],  alt_range[1])
        vels  = np.clip(X[:, 1], vel_range[0],  vel_range[1])
        winds = np.clip(X[:, 2], wind_range[0], wind_range[1])

        out = np.zeros(len(X))
        for i, (a, v, w) in enumerate(zip(alts, vels, winds)):
            mu_a = gauss(a, alt_c,  alt_s)   # shape (3,)
            mu_v = gauss(v, vel_c,  vel_s)
            mu_w = gauss(w, wind_c, wind_s)
            firing = np.array([
                mu_a[ia] * mu_v[iv] * mu_w[iw]
                for ia in range(3)
                for iv in range(3)
                for iw in range(3)
            ])
            total = firing.sum() + 1e-12
            w_norm = firing / total
            feats  = np.array([a, v, w, 1.0])
            conseq = rule_params @ feats
            out[i] = (w_norm * conseq).sum()
        return out

    return predict

# ─── Simulation ───────────────────────────────────────────────────────────────
def simulate(h0, wind_seq, controller="anfis", v0=0.0, anfis_fn=None):
    N = len(wind_seq)
    alt = np.zeros(N + 1);  alt[0] = h0
    vel = np.zeros(N + 1);  vel[0] = v0
    thr = np.zeros(N)
    integral = 0.0;  prev_err = 0.0

    for t in range(N):
        if alt[t] <= 0.0:
            alt[t:] = 0.0;  vel[t:] = vel[t]
            thr[t:] = WEIGHT
            break

        w = wind_seq[t]
        if controller == "pid":
            thrust, integral, prev_err = pid_thrust(alt[t], vel[t], w, integral, prev_err)
        else:
            thrust = float(anfis_fn(np.array([[alt[t], vel[t], w]]))[0])
            thrust = np.clip(thrust, 0.0, 3.0 * WEIGHT)

        thr[t] = thrust
        acc = (thrust - WEIGHT) / MASS
        vel[t + 1] = vel[t] + acc * DT
        alt[t + 1] = alt[t] - vel[t] * DT

        if alt[t + 1] <= 0.0:
            alt[t + 1:] = 0.0
            vel[t + 1:] = vel[t + 1]
            thr[t + 1:] = WEIGHT
            break

    return alt[:N], vel[:N], thr, wind_seq

# ─── Scenario definitions ─────────────────────────────────────────────────────
def make_wind(N, style):
    t = np.arange(N) * DT
    if style == "calm":
        return np.random.uniform(0.0, 2.0, N)
    if style == "moderate":
        base = 4.0 + np.random.randn(N) * 1.5
        return np.clip(base, 0, 10)
    if style == "gust":
        w = np.random.uniform(0.5, 2.0, N)
        w[200:210] = 10.0
        return w
    if style == "turbulent":
        return np.clip(8.0 + np.random.randn(N) * 2.0, 0, 14)
    if style == "sine":
        return 3.0 + 3.0 * np.sin(2 * np.pi * t / 8)
    return np.zeros(N)

SCENARIOS = {
    "Normal Landing":    dict(h0=30.0, wind_style="calm",     v0=0.0),
    "Moderate Wind":     dict(h0=30.0, wind_style="moderate", v0=0.0),
    "Sudden Gust":       dict(h0=30.0, wind_style="gust",     v0=0.0),
    "High Turbulence":   dict(h0=30.0, wind_style="turbulent",v0=0.0),
    "Near-Crash Recovery": dict(h0=8.0,  wind_style="calm",   v0=2.0),
}

# ─── Dashboard ────────────────────────────────────────────────────────────────
class Dashboard:
    def __init__(self):
        self.anfis_fn = load_anfis()
        self.scenario_names = list(SCENARIOS.keys())
        self.current = 0
        self._build_figure()
        self._run_and_plot(self.current)

    def _build_figure(self):
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.patch.set_facecolor("#1a1a2e")
        plt.suptitle("UAV Adaptive Landing — ANFIS vs PID Dashboard",
                     color="white", fontsize=14, fontweight="bold", y=0.98)

        gs = gridspec.GridSpec(3, 1, hspace=0.45, top=0.90, bottom=0.18,
                               left=0.08, right=0.97)
        ax_kwargs = dict(facecolor="#16213e")
        self.ax_alt  = self.fig.add_subplot(gs[0], **ax_kwargs)
        self.ax_wind = self.fig.add_subplot(gs[1], **ax_kwargs)
        self.ax_thr  = self.fig.add_subplot(gs[2], **ax_kwargs)

        for ax in (self.ax_alt, self.ax_wind, self.ax_thr):
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")

        # Scenario buttons
        btn_y = 0.03;  btn_h = 0.07;  btn_w = 0.15;  gap = 0.02
        self.buttons = []
        for i, name in enumerate(self.scenario_names):
            ax_btn = self.fig.add_axes(
                [0.03 + i * (btn_w + gap), btn_y, btn_w, btn_h])
            btn = Button(ax_btn, name, color="#0f3460", hovercolor="#533483")
            btn.label.set_color("white")
            btn.label.set_fontsize(8)
            btn.on_clicked(self._make_callback(i))
            self.buttons.append(btn)

    def _make_callback(self, idx):
        def cb(event):
            self.current = idx
            self._run_and_plot(idx)
            self.fig.canvas.draw_idle()
        return cb

    def _run_and_plot(self, idx):
        name = self.scenario_names[idx]
        cfg  = SCENARIOS[name]
        N    = 600
        np.random.seed(42)
        wind = make_wind(N, cfg["wind_style"])
        h0, v0 = cfg["h0"], cfg["v0"]

        pid_alt,   pid_vel,   pid_thr,  _ = simulate(h0, wind, "pid",   v0)
        anfis_alt, anfis_vel, anfis_thr, _ = simulate(h0, wind, "anfis", v0, self.anfis_fn)

        time = np.arange(N) * DT
        landed_pid   = np.argmax(pid_alt   <= 0.0) if pid_alt.min()   <= 0.0 else N
        landed_anfis = np.argmax(anfis_alt <= 0.0) if anfis_alt.min() <= 0.0 else N

        safe_pid   = pid_vel[max(0, landed_pid - 1)]   <= SAFE_VEL
        safe_anfis = anfis_vel[max(0, landed_anfis - 1)] <= SAFE_VEL

        for ax in (self.ax_alt, self.ax_wind, self.ax_thr):
            ax.cla()
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")

        # Altitude
        self.ax_alt.plot(time, pid_alt,   color="#e94560", lw=1.8, label="PID")
        self.ax_alt.plot(time, anfis_alt, color="#00b4d8", lw=1.8, label="ANFIS")
        self.ax_alt.axhline(0, color="gray", lw=0.7, linestyle="--")
        self.ax_alt.set_ylabel("Altitude (m)", color="white")
        self.ax_alt.set_title(f"Scenario: {name}", color="white", fontsize=11)
        self.ax_alt.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
        self.ax_alt.xaxis.label.set_color("white"); self.ax_alt.yaxis.label.set_color("white")

        # Wind
        self.ax_wind.fill_between(time, 0, wind, color="#f4a261", alpha=0.6, label="Wind (m/s)")
        self.ax_wind.set_ylabel("Wind (m/s)", color="white")
        self.ax_wind.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
        self.ax_wind.xaxis.label.set_color("white"); self.ax_wind.yaxis.label.set_color("white")

        # Thrust
        self.ax_thr.plot(time, pid_thr,   color="#e94560", lw=1.5, label="PID Thrust")
        self.ax_thr.plot(time, anfis_thr, color="#00b4d8", lw=1.5, label="ANFIS Thrust")
        self.ax_thr.axhline(WEIGHT, color="yellow", lw=0.8, linestyle=":", alpha=0.7,
                            label=f"Hover ({WEIGHT:.1f} N)")
        self.ax_thr.set_ylabel("Thrust (N)", color="white")
        self.ax_thr.set_xlabel("Time (s)", color="white")
        self.ax_thr.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
        self.ax_thr.xaxis.label.set_color("white"); self.ax_thr.yaxis.label.set_color("white")

        # Status annotations
        def status_str(safe, final_vel, landing_t):
            icon  = "✓ SAFE" if safe else "✗ HARD"
            t_str = f"{landing_t * DT:.1f}s" if landing_t < N else "timeout"
            return f"PID: {icon} | v={final_vel:.2f} m/s | land@{t_str}"

        pid_fv   = pid_vel[max(0, landed_pid - 1)]
        anfis_fv = anfis_vel[max(0, landed_anfis - 1)]

        self.ax_alt.text(
            0.01, 0.94,
            f"PID:   {'SAFE ✓' if safe_pid else 'HARD ✗'}  v={pid_fv:.2f} m/s\n"
            f"ANFIS: {'SAFE ✓' if safe_anfis else 'HARD ✗'}  v={anfis_fv:.2f} m/s",
            transform=self.ax_alt.transAxes,
            fontsize=8, color="white",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f3460", alpha=0.8)
        )

    def show(self):
        plt.show()


if __name__ == "__main__":
    db = Dashboard()
    db.show()
