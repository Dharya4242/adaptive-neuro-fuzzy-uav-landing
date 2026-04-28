"""
Phase 3 — Real-Time Simulation Replay
Side-by-side PID vs ANFIS landing animation

This script creates an animated comparison showing both controllers
landing a drone simultaneously under identical wind conditions.

Output: phase3_simulation_replay.png (static multi-frame comparison)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle
import warnings
warnings.filterwarnings("ignore")

# Constants
GRAVITY = 9.81
MASS = 2.0
WEIGHT = MASS * GRAVITY
DT = 0.05
SAFE_VEL = 0.5
MAX_STEPS = 1200

# PID Controller
class PIDController:
    def __init__(self, Ka=0.18, Kp=7.0, Ki=0.5, Kd=0.3, Kw=0.15,
                 v_max=5.5, v_min=0.3, integral_clamp=40.0):
        self.Ka = Ka; self.Kp = Kp; self.Ki = Ki; self.Kd = Kd
        self.Kw = Kw; self.v_max = v_max; self.v_min = v_min
        self.clamp = integral_clamp
        self._integ = 0.0; self._prev = 0.0

    def reset(self):
        self._integ = 0.0; self._prev = 0.0

    def compute(self, alt, vel, wind):
        tgt = float(np.clip(self.Ka * alt, self.v_min, self.v_max))
        err = vel - tgt
        self._integ = float(np.clip(self._integ + err * DT, -self.clamp, self.clamp))
        deriv = (err - self._prev) / DT
        self._prev = err
        thrust = WEIGHT - self.Kp * err - self.Ki * self._integ - self.Kd * deriv + self.Kw * wind
        return float(np.clip(thrust, 0.0, 3.0 * WEIGHT))

# Load trained ANFIS model
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

def generate_wind(n, scale=2.0, tau=0.5):
    alpha = DT / (tau + DT)
    w = np.zeros(n)
    w[0] = np.random.normal(0, scale)
    for t in range(1, n):
        w[t] = alpha * np.random.normal(0, scale) + (1 - alpha) * w[t - 1]
    return np.clip(w, 0, 15)

def run_landing(controller, h0, wind_seq, controller_type="PID"):
    if controller_type == "PID":
        controller.reset()
    
    alt = np.zeros(MAX_STEPS)
    vel = np.zeros(MAX_STEPS)
    thr = np.zeros(MAX_STEPS)
    alt[0] = h0
    t_land = MAX_STEPS - 1
    
    for t in range(MAX_STEPS - 1):
        if controller_type == "PID":
            thr[t] = controller.compute(alt[t], vel[t], wind_seq[t])
        else:
            X = np.array([[alt[t], vel[t], wind_seq[t]]])
            thr[t] = controller.predict(X)[0]
        
        accel = (thr[t] - WEIGHT) / MASS
        vel[t + 1] = vel[t] + accel * DT
        alt[t + 1] = alt[t] - vel[t] * DT
        
        if alt[t + 1] <= 0:
            alt[t + 1:] = 0.0
            thr[t + 1:] = WEIGHT
            t_land = t + 1
            break
    
    return alt, vel, thr, t_land

# Run simulation
print("=" * 70)
print("PHASE 3 — REAL-TIME SIMULATION REPLAY")
print("=" * 70)

np.random.seed(42)
h0 = 35.0
wind_seq = generate_wind(MAX_STEPS, scale=2.5)

pid = PIDController()
anfis = load_anfis_model()

print(f"\nSimulating landing from {h0}m altitude...")
pid_alt, pid_vel, pid_thr, pid_tland = run_landing(pid, h0, wind_seq, "PID")
anfis_alt, anfis_vel, anfis_thr, anfis_tland = run_landing(anfis, h0, wind_seq, "ANFIS")

print(f"  PID landed at t={pid_tland * DT:.2f}s, vel={abs(pid_vel[pid_tland]):.3f} m/s")
print(f"  ANFIS landed at t={anfis_tland * DT:.2f}s, vel={abs(anfis_vel[anfis_tland]):.3f} m/s")

# Create visualization
max_t = max(pid_tland, anfis_tland) + 20
t_axis = np.arange(max_t) * DT

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0a0a1e')
gs = gridspec.GridSpec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.35, wspace=0.25)

colors = {
    'pid': '#ff6b6b',
    'anfis': '#4ecdc4',
    'safe': '#95e1d3',
    'unsafe': '#ffa07a',
    'bg': '#0a0a1e',
    'panel': '#1a1a2e',
    'grid': '#2a2a3e',
    'text': '#e0e0e0'
}

# Panel 1: Side-by-side drone visualization at 4 key moments
ax_visual = fig.add_subplot(gs[0, :])
ax_visual.set_facecolor(colors['bg'])
ax_visual.set_xlim(-2, 10)
ax_visual.set_ylim(-2, 42)
ax_visual.axis('off')

snapshots = [0, max_t // 3, 2 * max_t // 3, min(pid_tland, anfis_tland)]

for i, t_idx in enumerate(snapshots):
    x_offset = i * 2.5
    
    # PID drone
    pid_y = pid_alt[t_idx]
    ax_visual.add_patch(Circle((x_offset, pid_y), 0.3, color=colors['pid'], ec='white', lw=2, zorder=10))
    ax_visual.text(x_offset, pid_y + 1.5, 'PID', ha='center', color=colors['pid'], 
                   fontsize=8, fontweight='bold')
    ax_visual.plot([x_offset, x_offset], [0, pid_y], color=colors['pid'], 
                   linestyle='--', alpha=0.3, linewidth=1)
    
    # ANFIS drone
    anfis_y = anfis_alt[t_idx]
    ax_visual.add_patch(Circle((x_offset + 0.8, anfis_y), 0.3, color=colors['anfis'], 
                               ec='white', lw=2, zorder=10))
    ax_visual.text(x_offset + 0.8, anfis_y + 1.5, 'ANFIS', ha='center', 
                   color=colors['anfis'], fontsize=8, fontweight='bold')
    ax_visual.plot([x_offset + 0.8, x_offset + 0.8], [0, anfis_y], 
                   color=colors['anfis'], linestyle='--', alpha=0.3, linewidth=1)
    
    # Time label
    ax_visual.text(x_offset + 0.4, -1, f't={t_idx * DT:.1f}s', ha='center', 
                   color=colors['text'], fontsize=9, alpha=0.7)
    
    # Ground line
    ax_visual.plot([-0.5 + x_offset, 1.8 + x_offset], [0, 0], color='#3a3a4a', 
                   linewidth=4, solid_capstyle='round')

ax_visual.text(5, 40, 'Landing Sequence Comparison', ha='center', 
               color='white', fontsize=14, fontweight='bold')

# Panel 2: Altitude over time
ax_alt = fig.add_subplot(gs[1, 0])
ax_alt.set_facecolor(colors['panel'])
ax_alt.plot(t_axis, pid_alt[:max_t], color=colors['pid'], linewidth=2, label='PID', alpha=0.9)
ax_alt.plot(t_axis, anfis_alt[:max_t], color=colors['anfis'], linewidth=2, label='ANFIS', alpha=0.9)
ax_alt.axvline(pid_tland * DT, color=colors['pid'], linestyle=':', alpha=0.5)
ax_alt.axvline(anfis_tland * DT, color=colors['anfis'], linestyle=':', alpha=0.5)
ax_alt.set_xlabel('Time (s)', color=colors['text'])
ax_alt.set_ylabel('Altitude (m)', color=colors['text'])
ax_alt.set_title('Altitude Profile', color='white', fontweight='bold', pad=10)
ax_alt.legend(loc='upper right', facecolor=colors['panel'], edgecolor=colors['grid'], 
              labelcolor=colors['text'])
ax_alt.grid(True, color=colors['grid'], alpha=0.3, linestyle='--')
ax_alt.tick_params(colors=colors['text'])
for spine in ax_alt.spines.values():
    spine.set_color(colors['grid'])

# Panel 3: Velocity over time
ax_vel = fig.add_subplot(gs[1, 1])
ax_vel.set_facecolor(colors['panel'])
ax_vel.plot(t_axis, pid_vel[:max_t], color=colors['pid'], linewidth=2, label='PID', alpha=0.9)
ax_vel.plot(t_axis, anfis_vel[:max_t], color=colors['anfis'], linewidth=2, label='ANFIS', alpha=0.9)
ax_vel.axhline(SAFE_VEL, color=colors['safe'], linestyle='--', linewidth=2, 
               label=f'Safe ({SAFE_VEL} m/s)', alpha=0.7)
ax_vel.axvline(pid_tland * DT, color=colors['pid'], linestyle=':', alpha=0.5)
ax_vel.axvline(anfis_tland * DT, color=colors['anfis'], linestyle=':', alpha=0.5)
ax_vel.set_xlabel('Time (s)', color=colors['text'])
ax_vel.set_ylabel('Velocity (m/s)', color=colors['text'])
ax_vel.set_title('Descent Velocity', color='white', fontweight='bold', pad=10)
ax_vel.legend(loc='upper right', facecolor=colors['panel'], edgecolor=colors['grid'], 
              labelcolor=colors['text'])
ax_vel.grid(True, color=colors['grid'], alpha=0.3, linestyle='--')
ax_vel.tick_params(colors=colors['text'])
for spine in ax_vel.spines.values():
    spine.set_color(colors['grid'])

# Panel 4: Thrust output
ax_thr = fig.add_subplot(gs[2, 0])
ax_thr.set_facecolor(colors['panel'])
ax_thr.plot(t_axis, pid_thr[:max_t], color=colors['pid'], linewidth=2, label='PID', alpha=0.9)
ax_thr.plot(t_axis, anfis_thr[:max_t], color=colors['anfis'], linewidth=2, label='ANFIS', alpha=0.9)
ax_thr.axhline(WEIGHT, color='yellow', linestyle=':', linewidth=1.5, 
               label=f'Hover ({WEIGHT:.1f} N)', alpha=0.6)
ax_thr.set_xlabel('Time (s)', color=colors['text'])
ax_thr.set_ylabel('Thrust (N)', color=colors['text'])
ax_thr.set_title('Thrust Command', color='white', fontweight='bold', pad=10)
ax_thr.legend(loc='upper right', facecolor=colors['panel'], edgecolor=colors['grid'], 
              labelcolor=colors['text'])
ax_thr.grid(True, color=colors['grid'], alpha=0.3, linestyle='--')
ax_thr.tick_params(colors=colors['text'])
for spine in ax_thr.spines.values():
    spine.set_color(colors['grid'])

# Panel 5: Wind disturbance
ax_wind = fig.add_subplot(gs[2, 1])
ax_wind.set_facecolor(colors['panel'])
ax_wind.fill_between(t_axis, 0, wind_seq[:max_t], color='skyblue', alpha=0.4, label='Wind speed')
ax_wind.plot(t_axis, wind_seq[:max_t], color='dodgerblue', linewidth=1.5, alpha=0.8)
ax_wind.set_xlabel('Time (s)', color=colors['text'])
ax_wind.set_ylabel('Wind (m/s)', color=colors['text'])
ax_wind.set_title('Wind Disturbance (Shared)', color='white', fontweight='bold', pad=10)
ax_wind.grid(True, color=colors['grid'], alpha=0.3, linestyle='--')
ax_wind.tick_params(colors=colors['text'])
for spine in ax_wind.spines.values():
    spine.set_color(colors['grid'])

# Add summary statistics box
summary_text = f"""
PID:    Landing time = {pid_tland * DT:.2f}s  |  Final vel = {abs(pid_vel[pid_tland]):.3f} m/s  |  {'✓ SAFE' if abs(pid_vel[pid_tland]) <= SAFE_VEL else '✗ UNSAFE'}
ANFIS:  Landing time = {anfis_tland * DT:.2f}s  |  Final vel = {abs(anfis_vel[anfis_tland]):.3f} m/s  |  {'✓ SAFE' if abs(anfis_vel[anfis_tland]) <= SAFE_VEL else '✗ UNSAFE'}
"""
fig.text(0.5, 0.02, summary_text, ha='center', color='white', fontsize=10, 
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor=colors['panel'], 
         edgecolor=colors['grid'], alpha=0.8))

fig.suptitle('Phase 3 — Real-Time Landing Simulation | PID vs ANFIS', 
             color='white', fontsize=16, fontweight='bold', y=0.98)

plt.savefig('phase3_simulation_replay.png', dpi=150, bbox_inches='tight', facecolor=colors['bg'])
print("\n✓ phase3_simulation_replay.png saved")
print("\n" + "=" * 70)
print("Simulation replay complete!")
print("=" * 70)
plt.close()
