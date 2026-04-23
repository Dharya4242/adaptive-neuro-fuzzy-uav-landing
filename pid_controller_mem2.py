"""
Drone Landing PID Controller — Phase 2 Baseline
=================================================
Implements a standard PID controller that takes (altitude, velocity, wind)
and outputs thrust_adjustment. Evaluated against the Phase-1 dataset.

Physics convention (mirrors data_generator.py exactly)
-------------------------------------------------------
  • velocity  > 0  →  descending (downward is positive)
  • altitude(t+1)  = altitude(t) - velocity(t) * dt
  • velocity(t+1)  = velocity(t) + (thrust - weight) / mass * dt
  • To DECELERATE: set thrust < weight (reduces downward velocity)
  • To ACCELERATE: set thrust > weight

Two evaluation tracks
---------------------
  A. RMSE track — grid search to best match the dataset thrust_adjustment labels.
     The labels follow: weight + 0.5*alt + 0.3*(vel - 0.5)  (positive Kd coefficient).
     These gains recreate the training data distribution but produce fast, unsafe descents.

  B. Safe-landing track — cascaded PID tuned for operational safety.
     Outer loop: altitude → proportional target velocity.
     Inner loop: velocity error → thrust correction.
     Produces high safe-landing rate; RMSE against labels is intentionally higher.

Academic motivation
-------------------
  PID must choose between minimising RMSE and maximising operational safety — it cannot
  simultaneously achieve both with a fixed gain set.  This fundamental limitation is what
  ANFIS (Phase 3) is designed to overcome by learning the nonlinear control landscape.

Outputs
-------
  performance_metrics.csv   — one-row CSV with all benchmark numbers
  pid_evaluation.png        — 4-panel diagnostic dashboard
  pid_sample_landing.png    — representative safe landing trace
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import itertools
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS  (must match data_generator.py)
# ============================================================================

GRAVITY   = 9.81
MASS      = 2.0
WEIGHT    = MASS * GRAVITY           # 19.62 N
DT        = 0.05                     # 50 ms
SAFE_VEL  = 0.5                      # m/s
MAX_STEPS = 1200                     # 60 s horizon

np.random.seed(42)

# ============================================================================
# WIND MODEL  (identical to data_generator.py)
# ============================================================================

def gen_wind(n: int, scale: float = 2.0, tau: float = 0.5) -> np.ndarray:
    alpha = DT / (tau + DT)
    w     = np.zeros(n)
    w[0]  = np.random.normal(0, scale)
    for t in range(1, n):
        w[t] = alpha * np.random.normal(0, scale) + (1 - alpha) * w[t - 1]
    return np.clip(w, 0, 15)

# ============================================================================
# PID CONTROLLER CLASS
# ============================================================================

class PIDController:
    """
    Cascaded discrete-time PID for drone landing.

    Thrust law
    ----------
        target_vel = clip(Ka * altitude, v_min, v_max)
        error      = velocity - target_vel
        integral  += error * dt                          (anti-windup clamped)
        deriv       = (error - prev_error) / dt
        thrust      = weight
                      - Kp * error                       (negative feedback)
                      - Ki * integral
                      - Kd * deriv
                      + Kw * wind                        (feed-forward)

    Sign logic
    ----------
    When velocity > target (too fast): error > 0, -Kp*error < 0 → thrust
    drops below weight → net downward force reverses → drone decelerates.  (safe ✓)

    RMSE-matching mode
    ------------------
    The training labels have coefficient +0.3 on (vel - 0.5), i.e., more velocity
    → more thrust → more acceleration.  Fitting them requires Kp < 0 (sign flip),
    which makes the controller pro-acceleration and operationally unsafe.
    This tension is explicitly surfaced in the metrics and is the motivation for ANFIS.
    """

    def __init__(self, Ka: float, Kp: float, Ki: float, Kd: float,
                 Kw: float = 0.0, v_max: float = 12.0, v_min: float = 0.3,
                 integral_clamp: float = 40.0):
        self.Ka = Ka;  self.Kp = Kp
        self.Ki = Ki;  self.Kd = Kd
        self.Kw = Kw;  self.v_max = v_max;  self.v_min = v_min
        self.clamp = integral_clamp
        self._integ = 0.0;  self._prev = 0.0

    def reset(self):
        self._integ = 0.0;  self._prev = 0.0

    def compute(self, alt: float, vel: float, wind: float) -> float:
        tgt        = float(np.clip(self.Ka * alt, self.v_min, self.v_max))
        err        = vel - tgt
        self._integ = float(np.clip(self._integ + err * DT, -self.clamp, self.clamp))
        deriv      = (err - self._prev) / DT
        self._prev = err
        thrust     = WEIGHT - self.Kp * err - self.Ki * self._integ - self.Kd * deriv + self.Kw * wind
        return float(np.clip(thrust, 0.0, 3.0 * WEIGHT))

# ============================================================================
# PHYSICS SIMULATION
# ============================================================================

def run_episode(ctrl: PIDController, h0: float, wind_in: np.ndarray = None) -> dict:
    ctrl.reset()
    alt = np.zeros(MAX_STEPS);  vel = np.zeros(MAX_STEPS)
    thr = np.zeros(MAX_STEPS)
    wnd = wind_in if wind_in is not None else gen_wind(MAX_STEPS)
    alt[0] = h0;  t_land = MAX_STEPS - 1;  landed = False

    for t in range(MAX_STEPS - 1):
        thr[t]     = ctrl.compute(alt[t], vel[t], wnd[t])
        accel      = (thr[t] - WEIGHT) / MASS
        vel[t + 1] = vel[t] + accel * DT
        alt[t + 1] = alt[t] - vel[t] * DT
        if alt[t + 1] <= 0:
            alt[t + 1:] = 0.0;  thr[t + 1:] = WEIGHT
            t_land = t + 1;  landed = True;  break

    return dict(alt=alt, vel=vel, thr=thr, wnd=wnd,
                safe=landed and abs(vel[t_land]) <= SAFE_VEL,
                landed=landed,
                landing_time=t_land * DT,
                final_vel=abs(vel[t_land]))

# ============================================================================
# METRIC 1 — RMSE
# ============================================================================

def eval_rmse(ctrl: PIDController, df: pd.DataFrame):
    ctrl.reset()
    preds = [];  alts = df['altitude'].values
    for i, row in enumerate(df.itertuples(index=False)):
        if i > 0 and (row.altitude > alts[i - 1] + 5.0 or alts[i - 1] < 0.5):
            ctrl.reset()
        preds.append(ctrl.compute(row.altitude, row.velocity, row.wind))
    y  = df['thrust_adjustment'].values
    yh = np.array(preds)
    return float(np.sqrt(np.mean((y - yh) ** 2))), yh

# ============================================================================
# METRIC 2 — SAFE-LANDING RATE
# ============================================================================

def eval_safe_landing(ctrl: PIDController, n: int = 300) -> dict:
    h0s = np.random.uniform(10, 50, n)
    safe = 0;  landed = 0;  fv = [];  lt = []
    for h0 in h0s:
        r = run_episode(ctrl, h0)
        if r['landed']:
            landed += 1;  fv.append(r['final_vel']);  lt.append(r['landing_time'])
            if r['safe']:  safe += 1
    rate = safe / n * 100
    print(f"  Safe: {safe}/{n} ({rate:.1f}%)   landed: {landed}/{n}")
    return dict(n_episodes=n, n_safe=safe, n_landed=landed,
                safe_pct=round(rate, 2),
                mean_final_vel=round(float(np.mean(fv)), 4)  if fv else 0.0,
                mean_landing_time=round(float(np.mean(lt)), 4) if lt else 0.0)

# ============================================================================
# METRIC 3 — GUST RESPONSE
# ============================================================================

def eval_gust(ctrl: PIDController, h0: float = 25.0,
              gust_t: float = 1.0, gust_mag: float = 5.0,
              gust_dur: float = 0.5, band: float = 0.15) -> dict:
    wnd   = np.ones(MAX_STEPS) * 0.2
    onset = int(gust_t / DT)
    wnd[onset: onset + int(gust_dur / DT)] = gust_mag
    r     = run_episode(ctrl, h0, wind_in=wnd)
    vp    = r['vel'][onset:]
    tp    = np.array([np.clip(ctrl.Ka * r['alt'][onset + k], ctrl.v_min, ctrl.v_max)
                      for k in range(len(vp))])
    errs  = np.abs(vp - tp)
    peak  = float(np.max(errs))
    rt    = next((k * DT for k, e in enumerate(errs) if e < band), float(len(errs) * DT))
    print(f"  Peak vel error: {peak:.3f} m/s   Recovery: {rt:.3f} s")
    return dict(gust_mag=gust_mag, gust_dur=gust_dur,
                peak_vel_error=round(peak, 4), response_time_s=round(rt, 4),
                t_axis=np.arange(MAX_STEPS) * DT,
                vel=r['vel'], alt=r['alt'], wnd=wnd, onset=onset)

# ============================================================================
# GRID SEARCH
# ============================================================================

def grid_search(train_df: pd.DataFrame) -> tuple:
    Ka_v = [0.3, 0.5, 0.7]
    Kp_v = [-1.0, -0.5, -0.3, 0.0, 0.3, 0.5, 1.0]   # negative = pro-accel (unsafe)
    Ki_v = [0.0, 0.05, 0.1]
    Kd_v = [0.0, 0.05, 0.1, 0.2]
    Kw_v = [0.0, 0.05, 0.1]
    total = len(Ka_v)*len(Kp_v)*len(Ki_v)*len(Kd_v)*len(Kw_v)
    print(f"  Evaluating {total} combinations …")

    y  = train_df['thrust_adjustment'].values
    best_rmse = np.inf;  best_p = {}

    for Ka, Kp, Ki, Kd, Kw in itertools.product(Ka_v, Kp_v, Ki_v, Kd_v, Kw_v):
        ctrl = PIDController(Ka, Kp, Ki, Kd, Kw)
        yh   = np.array([ctrl.compute(r.altitude, r.velocity, r.wind)
                         for r in train_df.itertuples(index=False)])
        rmse = float(np.sqrt(np.mean((y - yh) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_p    = dict(Ka=Ka, Kp=Kp, Ki=Ki, Kd=Kd, Kw=Kw)

    print(f"  Best RMSE: {best_rmse:.4f} N   Gains: {best_p}")
    return best_p, best_rmse

# ============================================================================
# PLOTTING HELPERS
# ============================================================================

BG    = '#0f0f1a';  PBG   = '#1a1a2e';  BORDER = '#333355'
CYAN  = '#00e5ff';  RED   = '#ff4c4c';  GREEN  = '#39ff14'
ORG   = '#ffa500';  YLW   = '#ffe600';  LGREY  = '#cccccc'


def _style_ax(ax, title):
    ax.set_facecolor(PBG)
    ax.set_title(title, color=CYAN, fontsize=11, fontweight='bold', pad=8)
    ax.tick_params(colors=LGREY, labelsize=9)
    ax.xaxis.label.set_color(LGREY);  ax.yaxis.label.set_color(LGREY)
    for sp in ax.spines.values():  sp.set_edgecolor(BORDER)
    ax.grid(True, color='#252545', linestyle='--', lw=0.6)


def plot_evaluation(rmse_ctrl, safe_ctrl, test_df, preds, rmse_val,
                    gust, safe_d, rg, sg):
    fig = plt.figure(figsize=(16, 12));  fig.patch.set_facecolor(BG)
    y   = test_df['thrust_adjustment'].values

    # Panel A — Predicted vs True
    ax  = fig.add_subplot(2, 2, 1);  _style_ax(ax, 'A — Predicted vs. True  (RMSE Controller)')
    idx = np.random.choice(len(y), min(3000, len(y)), replace=False)
    ax.scatter(y[idx], preds[idx], s=4, alpha=0.30, color=CYAN, rasterized=True)
    mn, mx = min(y.min(), preds.min()), max(y.max(), preds.max())
    ax.plot([mn, mx], [mn, mx], '--', color=GREEN, lw=1.8, label='Perfect fit')
    r2 = 1.0 - np.var(y - preds) / np.var(y)
    ax.text(0.04, 0.95, f'RMSE = {rmse_val:.3f} N\nR² = {r2:.4f}',
            transform=ax.transAxes, color=YLW, fontsize=11,
            fontweight='bold', va='top')
    ax.set_xlabel('True Thrust (N)');  ax.set_ylabel('PID Predicted (N)')
    ax.legend(fontsize=9, facecolor=PBG, labelcolor=LGREY)

    # Panel B — Residuals
    ax2 = fig.add_subplot(2, 2, 2);  _style_ax(ax2, 'B — Residual Distribution  (RMSE Controller)')
    res = y - preds
    ax2.hist(res, bins=80, color=CYAN, edgecolor='none', alpha=0.80)
    ax2.axvline(0, color=GREEN, lw=2.0, ls='--', label='Zero error')
    ax2.axvline(res.mean(), color=ORG, lw=1.8, ls='-.',
                label=f'Mean = {res.mean():.3f} N')
    ax2.text(0.68, 0.95, f'σ = {res.std():.3f} N',
             transform=ax2.transAxes, color=YLW, fontsize=11,
             fontweight='bold', va='top')
    ax2.set_xlabel('Residual (N)');  ax2.set_ylabel('Count')
    ax2.legend(fontsize=9, facecolor=PBG, labelcolor=LGREY)

    # Panel C — Gust response
    ax3  = fig.add_subplot(2, 2, 3);  ax3b = ax3.twinx()
    ax3.set_facecolor(PBG)
    ax3.set_title('C — Gust Response  (Safety Controller)', color=CYAN,
                  fontsize=11, fontweight='bold', pad=8)
    end  = min(int(25 / DT), len(gust['t_axis']))
    t    = gust['t_axis'][:end]
    ax3.plot(t, gust['vel'][:end],  color=RED,  lw=2.0, label='Velocity (m/s)')
    ax3b.plot(t, gust['wnd'][:end], color=ORG,  lw=1.4, ls='--',
              alpha=0.8, label='Wind (m/s)')
    ax3.axhline(SAFE_VEL, color=GREEN, lw=1.8, ls='--',
                label=f'Safe threshold ({SAFE_VEL} m/s)')
    onset_t = gust['onset'] * DT
    ax3.axvline(onset_t, color=YLW, lw=1.2, ls=':', alpha=0.8)
    ax3.text(onset_t + 0.2, max(gust['vel'][:end]) * 0.88,
             '← Gust onset', color=YLW, fontsize=8)
    rec_t = onset_t + gust['response_time_s']
    if rec_t < t[-1]:
        ax3.axvline(rec_t, color=GREEN, lw=1.2, ls=':', alpha=0.8)
        ax3.text(rec_t + 0.2, max(gust['vel'][:end]) * 0.65,
                 f'← Recovered\n  in {gust["response_time_s"]:.2f} s',
                 color=GREEN, fontsize=8)
    ax3.set_xlabel('Time (s)', color=LGREY);  ax3.set_ylabel('Velocity (m/s)', color=RED)
    ax3b.set_ylabel('Wind (m/s)', color=ORG)
    ax3.tick_params(colors=LGREY, axis='both');  ax3.tick_params(axis='y', labelcolor=RED)
    ax3b.tick_params(axis='y', labelcolor=ORG)
    for sp in ax3.spines.values():  sp.set_edgecolor(BORDER)
    ax3.grid(True, color='#252545', ls='--', lw=0.6)
    l1, lb1 = ax3.get_legend_handles_labels()
    l2, lb2 = ax3b.get_legend_handles_labels()
    ax3.legend(l1 + l2, lb1 + lb2, fontsize=8, facecolor=PBG, labelcolor=LGREY)

    # Panel D — Summary table
    ax4 = fig.add_subplot(2, 2, 4);  ax4.set_facecolor(PBG);  ax4.axis('off')
    ax4.set_title('D — Performance Summary', color=CYAN, fontsize=11,
                  fontweight='bold', pad=8)
    for sp in ax4.spines.values():  sp.set_edgecolor(BORDER)
    mae = float(np.mean(np.abs(y - preds)))

    rows = [
        ('Metric',                'Value',                   'Unit'),
        ('── RMSE-OPTIMAL ─────────────────', '', ''),
        ('Test RMSE',             f'{rmse_val:.4f}',         'N'),
        ('Test MAE',              f'{mae:.4f}',              'N'),
        ('R²',                    f'{r2:.4f}',               '—'),
        (f'Ka={rg["Ka"]}  Kp={rg["Kp"]}', f'Ki={rg["Ki"]}  Kd={rg["Kd"]}', ''),
        ('  ⚠  Kp<0 → pro-accel (unsafe)', '', ''),
        ('── SAFETY-TUNED ─────────────────', '', ''),
        ('Safe Landing Rate',     f'{safe_d["safe_pct"]:.1f}', '%'),
        ('Safe / Episodes',       f'{safe_d["n_safe"]}/{safe_d["n_episodes"]}', ''),
        ('Mean Final Velocity',   f'{safe_d["mean_final_vel"]:.3f}', 'm/s'),
        ('Mean Landing Time',     f'{safe_d["mean_landing_time"]:.1f}', 's'),
        ('Gust Recovery Time',    f'{gust["response_time_s"]:.3f}', 's'),
        ('Peak Velocity Error',   f'{gust["peak_vel_error"]:.3f}', 'm/s'),
        (f'Ka={sg["Ka"]}  Kp={sg["Kp"]}', f'Ki={sg["Ki"]}  Kd={sg["Kd"]}', ''),
    ]

    cx = [0.02, 0.58, 0.90];  rh = 0.065;  y0 = 0.93
    for j, h in enumerate(rows[0]):
        ax4.text(cx[j], y0, h, color=YLW, fontsize=9.5,
                 fontweight='bold', transform=ax4.transAxes)
    ax4.add_line(mlines.Line2D([0.0, 1.0], [y0 - 0.018, y0 - 0.018],
                 transform=ax4.transAxes, color=BORDER, lw=0.8))
    for i, rv in enumerate(rows[1:]):
        y_ = y0 - rh * (i + 1)
        c  = CYAN if '──' in str(rv[0]) else (GREEN if i == 7 else LGREY)
        for j, v in enumerate(rv):
            ax4.text(cx[j], y_, str(v), color=c, fontsize=8.5,
                     transform=ax4.transAxes, va='center')

    fig.suptitle('PID Controller — Baseline Evaluation Dashboard\n'
                 'Phase 2 | Drone Landing Control | ANFIS Benchmark Reference',
                 color='white', fontsize=13, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.963])
    plt.savefig('pid_evaluation.png', dpi=150, bbox_inches='tight', facecolor=BG)
    print("  Saved → pid_evaluation.png")
    plt.close()


def plot_sample_landing(ctrl: PIDController):
    r  = run_episode(ctrl, h0=35.0)
    tl = min(int(r['landing_time'] / DT) + 8, MAX_STEPS)
    t  = np.arange(tl) * DT

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor(BG)
    specs = [('alt', 'Altitude (m)', CYAN, True),
             ('vel', 'Velocity (m/s)', RED, False),
             ('thr', 'Thrust (N)', ORG, False)]
    titles = ['Altitude Profile', 'Velocity Profile', 'Thrust Output']
    for ax, (k, lbl, c, fill), ttl in zip(axes, specs, titles):
        ax.set_facecolor(PBG);  ax.plot(t, r[k][:tl], color=c, lw=2.2)
        if fill:  ax.fill_between(t, 0, r[k][:tl], alpha=0.12, color=c)
        if k == 'vel':
            ax.axhline(SAFE_VEL, color=GREEN, lw=1.8, ls='--',
                       label=f'Safe threshold ({SAFE_VEL} m/s)')
            ax.legend(fontsize=9, facecolor=PBG, labelcolor=LGREY)
        if k == 'thr':
            ax.axhline(WEIGHT, color=GREEN, lw=1.2, ls=':',
                       label=f'Hover ({WEIGHT:.1f} N)')
            ax.legend(fontsize=9, facecolor=PBG, labelcolor=LGREY)
        ax.set_xlabel('Time (s)', color=LGREY);  ax.set_ylabel(lbl, color=LGREY)
        ax.set_title(ttl, color=CYAN, fontweight='bold')
        ax.tick_params(colors=LGREY)
        ax.grid(True, color='#252545', ls='--', lw=0.6)
        for sp in ax.spines.values():  sp.set_edgecolor(BORDER)

    st = '✓  SAFE' if r['safe'] else '✗  UNSAFE'
    fig.suptitle(f'PID Safety Controller — Sample Landing | h₀=35 m | '
                 f'Final vel={r["final_vel"]:.3f} m/s | {st}',
                 color='white', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pid_sample_landing.png', dpi=150, bbox_inches='tight', facecolor=BG)
    print("  Saved → pid_sample_landing.png")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 68)
    print("  DRONE LANDING — PID BASELINE CONTROLLER  (Phase 2)")
    print("=" * 68)

    # 1. Load dataset
    print("\n[1] Loading dataset …")
    df = pd.read_csv('dataset.csv')
    if 'S.No' in df.columns:  df = df.drop(columns=['S.No'])
    print(f"    {len(df):,} rows × {len(df.columns)} features")

    # 2. Split (time-ordered 80/20)
    s        = int(0.8 * len(df))
    train_df = df.iloc[:s].reset_index(drop=True)
    test_df  = df.iloc[s:].reset_index(drop=True)
    print(f"\n[2] Split → train: {len(train_df):,}  |  test: {len(test_df):,}")

    # ── TRACK A: RMSE-OPTIMAL ────────────────────────────────────────────────
    print("\n[3] Grid search for RMSE-optimal gains …")
    rg, _      = grid_search(train_df)
    rmse_ctrl  = PIDController(**rg)
    rmse_v, yh = eval_rmse(rmse_ctrl, test_df)
    y          = test_df['thrust_adjustment'].values
    mae_v      = float(np.mean(np.abs(y - yh)))
    r2_v       = 1.0 - float(np.sum((y - yh)**2)) / float(np.sum((y - y.mean())**2))
    print(f"\n    RMSE={rmse_v:.4f} N   MAE={mae_v:.4f} N   R²={r2_v:.4f}")
    unsafe_flag = rg['Kp'] < 0
    print(f"    Kp={rg['Kp']}  {'⚠  negative Kp = pro-acceleration (unsafe landings)' if unsafe_flag else ''}")

    # ── TRACK B: SAFETY-TUNED ────────────────────────────────────────────────
    # Cascaded: target_vel = Ka*alt, then velocity-error PID.
    # Verified to produce safe landings from 10–50 m.
    sg       = dict(Ka=0.18, Kp=7.0, Ki=0.5, Kd=0.3, Kw=0.15, v_max=5.5, v_min=0.3)
    safe_ctrl = PIDController(**sg)

    print(f"\n[4] Safety controller gains → {sg}")
    print("\n[5] Evaluating safe-landing rate (300 episodes) …")
    sd = eval_safe_landing(safe_ctrl, n=300)

    print("\n[6] Evaluating gust response …")
    gd = eval_gust(safe_ctrl)

    # ── SAVE METRICS ─────────────────────────────────────────────────────────
    metrics = dict(
        controller='PID',
        # RMSE track
        rmse_Ka=rg['Ka'], rmse_Kp=rg['Kp'], rmse_Ki=rg['Ki'],
        rmse_Kd=rg['Kd'], rmse_Kw=rg.get('Kw', 0.0),
        test_rmse_N=round(rmse_v, 6),
        test_mae_N=round(mae_v, 6),
        test_r2=round(r2_v, 6),
        rmse_gains_pro_acceleration=int(unsafe_flag),
        # Safety track
        safe_Ka=sg['Ka'], safe_Kp=sg['Kp'], safe_Ki=sg['Ki'],
        safe_Kd=sg['Kd'], safe_Kw=sg['Kw'],
        safe_landing_pct=sd['safe_pct'],
        n_episodes=sd['n_episodes'],
        n_safe=sd['n_safe'],
        n_landed=sd['n_landed'],
        mean_final_vel_ms=sd['mean_final_vel'],
        mean_landing_time_s=sd['mean_landing_time'],
        # Gust
        gust_magnitude_ms=gd['gust_mag'],
        gust_duration_s=gd['gust_dur'],
        gust_response_time_s=gd['response_time_s'],
        gust_peak_vel_error_ms=gd['peak_vel_error'],
        # Dataset info
        train_rows=len(train_df),
        test_rows=len(test_df),
        note=('RMSE gains minimise label-matching error but are operationally unsafe '
              '(Kp < 0 is pro-acceleration). Safety gains ensure operational safety '
              'at cost of higher RMSE. ANFIS target: beat BOTH metrics simultaneously.')
    )
    pd.DataFrame([metrics]).to_csv('performance_metrics.csv', index=False)
    print("\n[7] performance_metrics.csv saved.")

    # ── PRINT SUMMARY ─────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  PERFORMANCE SUMMARY")
    print("=" * 68)
    print(f"  RMSE-OPTIMAL  (Ka={rg['Ka']}, Kp={rg['Kp']}, Ki={rg['Ki']}, Kd={rg['Kd']})")
    print(f"    Test RMSE           :  {rmse_v:.4f} N")
    print(f"    Test MAE            :  {mae_v:.4f} N")
    print(f"    R² on test set      :  {r2_v:.4f}")
    print(f"    ⚠  Kp<0 → pro-acceleration; unsafe landings")
    print()
    print(f"  SAFETY-TUNED  (Ka={sg['Ka']}, Kp={sg['Kp']}, Ki={sg['Ki']}, Kd={sg['Kd']}, Kw={sg['Kw']})")
    print(f"    Safe landing rate   :  {sd['safe_pct']}%  ({sd['n_safe']}/{sd['n_episodes']})")
    print(f"    Mean final velocity :  {sd['mean_final_vel']} m/s")
    print(f"    Mean landing time   :  {sd['mean_landing_time']} s")
    print(f"    Gust response time  :  {gd['response_time_s']} s")
    print(f"    Peak velocity error :  {gd['peak_vel_error']} m/s")
    print("=" * 68)
    print("  → ANFIS target: low RMSE AND high safe-landing rate")
    print("=" * 68)

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    print("\n[8] Generating plots …")
    plot_evaluation(rmse_ctrl, safe_ctrl, test_df, yh, rmse_v,
                    gd, sd, rg, sg)
    plot_sample_landing(safe_ctrl)

    print("\n✅  All outputs ready:")
    print("      pid_controller.py")
    print("      performance_metrics.csv")
    print("      pid_evaluation.png")
    print("      pid_sample_landing.png")


if __name__ == '__main__':
    main()