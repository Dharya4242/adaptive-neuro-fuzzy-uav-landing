"""
Drone Landing Data Generator - CORRECTED VERSION
Generates synthetic dataset for ANFIS neuro-fuzzy controller training.

Features:
  • Physics-based simulation (gravity, mass, thrust)
  • Correlated wind model (not white noise)
  • Ground truth thrust labels (PD-law)
  • 10,000 rows with diverse landing scenarios
  • S.No column for row identification
  • CORRECTED visualization showing complete landing trajectories

Physics: altitude, velocity, gravity, wind, and thrust adjustments
Dataset: ~10,000 rows covering varied landing scenarios
Output: dataset.csv with columns [S.No, altitude, velocity, wind, thrust_adjustment]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# ============================================================================
# PHYSICS PARAMETERS
# ============================================================================

GRAVITY = 9.81  # m/s^2
DRONE_MASS = 2.0  # kg (affects thrust-to-accel ratio)
DT = 0.05  # time step (50 ms)
SAFE_LANDING_VELOCITY = 0.5  # m/s — threshold for "safe" landing
INITIAL_ALTITUDE = 50.0  # m — start height

# ============================================================================
# WIND MODEL: Correlated Gaussian (low-pass filtered)
# ============================================================================

def generate_correlated_wind(duration_steps, dt, wind_scale=2.0, correlation_tau=0.5):
    """
    Generate realistic wind with temporal correlation.
    Uses first-order low-pass filter on white noise.
    
    Args:
        duration_steps: number of time steps
        dt: time step duration (s)
        wind_scale: standard deviation of wind speed (m/s)
        correlation_tau: time constant for low-pass filter (s)
    
    Returns:
        wind array of shape (duration_steps,)
    """
    # Filter coefficient: smaller tau → more correlation (slower changes)
    alpha = dt / (correlation_tau + dt)
    
    # Generate white noise
    white_noise = np.random.normal(0, wind_scale, duration_steps)
    
    # Low-pass filter to create gusts with duration
    wind = np.zeros(duration_steps)
    wind[0] = white_noise[0]
    for t in range(1, duration_steps):
        wind[t] = alpha * white_noise[t] + (1 - alpha) * wind[t-1]
    
    # Clip to [0, 15] m/s (no negative wind in this model)
    wind = np.clip(wind, 0, 15)
    
    return wind


# ============================================================================
# DRONE DYNAMICS
# ============================================================================

def simulate_landing(initial_altitude, controller_fn, num_steps=1000, dt=DT, initial_velocity=0.0):
    """
    Simulate a single drone landing scenario.

    Args:
        initial_altitude: starting height (m)
        controller_fn: function that takes (altitude, velocity, wind) and returns thrust (N)
        num_steps: simulation steps
        dt: time step (s)
        initial_velocity: starting descent speed (m/s), default 0

    Returns:
        arrays: altitude, velocity, wind, thrust, thrust_adjustment (ground truth label)
    """
    altitude = np.zeros(num_steps)
    velocity = np.zeros(num_steps)
    wind = np.zeros(num_steps)
    thrust = np.zeros(num_steps)
    thrust_adjustment = np.zeros(num_steps)

    # Initial conditions
    altitude[0] = initial_altitude
    velocity[0] = float(initial_velocity)

    # Generate wind for this scenario
    # Three wind regimes so ANFIS trains on the full 0-15 m/s range.
    # The low-pass filter suppresses variance, so we compensate with a base offset.
    regime = np.random.randint(3)
    if regime == 0:          # calm (0-3 m/s)
        wind_signal = generate_correlated_wind(num_steps, dt, wind_scale=2.0, correlation_tau=0.5)
    elif regime == 1:        # moderate (3-8 m/s)
        wind_signal = np.clip(
            generate_correlated_wind(num_steps, dt, wind_scale=3.0, correlation_tau=0.5) + 4.0,
            0, 15)
    else:                    # gusty (6-12 m/s)
        wind_signal = np.clip(
            generate_correlated_wind(num_steps, dt, wind_scale=3.0, correlation_tau=0.5) + 8.0,
            0, 15)
    wind[:] = wind_signal

    landing_step = num_steps  # track where landing occurs

    # Simulate
    for t in range(num_steps - 1):
        # Get controller output: desired thrust adjustment
        desired_thrust = controller_fn(altitude[t], velocity[t], wind[t])
        thrust[t] = desired_thrust

        # Label = actual controller thrust (consistent with the dynamics that produced this row)
        thrust_adjustment[t] = desired_thrust

        # Update dynamics
        weight = DRONE_MASS * GRAVITY
        net_force = desired_thrust - weight
        accel = net_force / DRONE_MASS

        # Simple Euler integration
        velocity[t+1] = velocity[t] + accel * dt
        altitude[t+1] = altitude[t] - velocity[t] * dt  # positive velocity = descending

        # Stop if hit ground
        if altitude[t+1] <= 0:
            landing_step = t + 1
            break

    return altitude, velocity, wind, thrust, thrust_adjustment, landing_step


# ============================================================================
# CONTROLLER FUNCTIONS (for data generation)
# ============================================================================

def naive_controller(altitude, velocity, wind):
    """Naive constant-thrust controller (hover)."""
    return DRONE_MASS * GRAVITY


def pd_controller(altitude, velocity, wind):
    """PD controller with negative velocity feedback."""
    kp = 0.6
    kd = 0.4
    weight = DRONE_MASS * GRAVITY
    return weight + kp * altitude - kd * velocity


def smart_controller(altitude, velocity, wind):
    """PD with wind feed-forward."""
    kp = 0.7
    kd = 0.5
    kw = 0.1
    weight = DRONE_MASS * GRAVITY
    return weight + kp * altitude - kd * velocity + kw * wind


def safety_pd_controller(altitude, velocity, wind):
    """
    Safety controller matching Phase 3 PID steady-state behavior.
    Cascaded: target_vel = clip(Ka*alt, v_min, v_max), then P on velocity error.
    Produces safe landings (touchdown < 0.5 m/s) and includes wind feed-forward.
    Gains mirror the safety-tuned PID in phase3_evaluation.py.
    """
    Ka, Kp, Kw = 0.18, 7.0, 0.15
    v_min, v_max = 0.3, 5.5
    weight = DRONE_MASS * GRAVITY
    vel_target = float(np.clip(Ka * altitude, v_min, v_max))
    thrust = weight - Kp * (velocity - vel_target) + Kw * wind
    return float(np.clip(thrust, 0.0, 3.0 * weight))


# ============================================================================
# SCENARIO GENERATORS
# ============================================================================

def generate_stable_landing_scenario(num_steps=500):
    """Ideal scenario: high altitude, calm wind."""
    altitude, velocity, wind, thrust, thrust_adj, _ = simulate_landing(
        initial_altitude=45.0,
        controller_fn=safety_pd_controller,
        num_steps=num_steps,
        dt=DT
    )
    return altitude, velocity, wind, thrust_adj


def generate_gusty_landing_scenario(num_steps=500):
    """Challenging: moderate altitude, significant wind gusts."""
    altitude, velocity, wind, thrust, thrust_adj, _ = simulate_landing(
        initial_altitude=35.0,
        controller_fn=safety_pd_controller,
        num_steps=num_steps,
        dt=DT
    )
    return altitude, velocity, wind, thrust_adj


def generate_low_altitude_recovery_scenario(num_steps=500):
    """Emergency: low altitude with non-zero initial velocity (near-crash recovery)."""
    initial_vel = np.random.uniform(2.0, 5.0)  # start with some descent speed
    altitude, velocity, wind, thrust, thrust_adj, _ = simulate_landing(
        initial_altitude=np.random.uniform(5.0, 15.0),
        controller_fn=safety_pd_controller,
        num_steps=num_steps,
        dt=DT,
        initial_velocity=initial_vel,
    )
    return altitude, velocity, wind, thrust_adj


def generate_random_scenario(num_steps=500):
    """Randomised: random initial altitude with safety controller."""
    initial_alt = np.random.uniform(10, 50)
    altitude, velocity, wind, thrust, thrust_adj, _ = simulate_landing(
        initial_altitude=initial_alt,
        controller_fn=safety_pd_controller,
        num_steps=num_steps,
        dt=DT
    )
    return altitude, velocity, wind, thrust_adj


# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_dataset(num_rows=10000, output_file='dataset.csv'):
    """
    Generate full dataset by simulating multiple landing scenarios.
    
    Args:
        num_rows: target number of rows
        output_file: CSV filename to save
    
    Returns:
        DataFrame with columns: S.No, altitude, velocity, wind, thrust_adjustment
    """
    data_list = []
    
    # Estimate steps per scenario
    scenarios_needed = max(1, num_rows // 300)
    
    print(f"Generating {scenarios_needed} scenarios to reach ~{num_rows} rows...")
    
    scenario_generators = [
        generate_stable_landing_scenario,
        generate_gusty_landing_scenario,
        generate_low_altitude_recovery_scenario,
        generate_low_altitude_recovery_scenario,  # double weight for near-crash coverage
        generate_random_scenario,
    ]
    
    for scenario_idx in range(scenarios_needed):
        # Pick a scenario type (cycle through them for balance)
        scenario_type = scenario_generators[scenario_idx % len(scenario_generators)]
        
        # Generate scenario with randomized length
        num_steps = np.random.randint(300, 600)
        altitude, velocity, wind, thrust_adj = scenario_type(num_steps=num_steps)

        # Add only in-flight rows (stop at landing, drop final 5 near-ground steps)
        cutoff = num_steps - 5
        for t in range(len(altitude)):
            if altitude[t] <= 0.0 or t >= cutoff:
                break
            data_list.append({
                'altitude': altitude[t],
                'velocity': velocity[t],
                'wind': wind[t],
                'thrust_adjustment': thrust_adj[t],
            })
        
        if (scenario_idx + 1) % 50 == 0:
            print(f"  ... {scenario_idx + 1} scenarios, {len(data_list)} rows so far")
    
    df = pd.DataFrame(data_list)
    
    # Trim to exactly num_rows
    df = df.iloc[:num_rows].reset_index(drop=True)
    
    # Add S.No column (serial number) as first column
    df.insert(0, 'S.No', range(1, len(df) + 1))
    
    print(f"\nDataset complete: {len(df)} rows")
    print(f"\nColumn structure:")
    print(f"  • S.No (Serial Number): 1 to {len(df)}")
    print(f"  • Altitude (m): {df['altitude'].min():.2f} to {df['altitude'].max():.2f}")
    print(f"  • Velocity (m/s): {df['velocity'].min():.2f} to {df['velocity'].max():.2f}")
    print(f"  • Wind (m/s): {df['wind'].min():.2f} to {df['wind'].max():.2f}")
    print(f"  • Thrust Adjustment (N): {df['thrust_adjustment'].min():.2f} to {df['thrust_adjustment'].max():.2f}")
    
    print(f"\nStatistics (excluding S.No):")
    print(df[['altitude', 'velocity', 'wind', 'thrust_adjustment']].describe())
    
    # Verify data quality
    print(f"\nData Quality Checks:")
    print(f"  ✓ Total rows: {len(df)}")
    print(f"  ✓ Total columns: {len(df.columns)}")
    print(f"  ✓ Missing values: {df.isnull().sum().sum()}")
    print(f"  ✓ S.No range: {df['S.No'].min()} to {df['S.No'].max()}")
    print(f"  ✓ Column order: {list(df.columns)}")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
    
    return df


# ============================================================================
# VISUALIZATION - CORRECTED VERSION
# ============================================================================

def plot_sample_trajectories(output_file='dataset.csv', num_samples=3):
    """
    Plot sample trajectories from the generated dataset.
    Shows COMPLETE landing sequences: altitude, velocity, and wind over time.
    CORRECTED to show full 10-second descent, not just end-of-landing phase.
    """
    df = pd.read_csv(output_file)
    
    print("\n" + "="*80)
    print("GENERATING CORRECTED VISUALIZATION")
    print("="*80)
    
    # CORRECTED: Extract complete landing scenarios properly
    # A complete scenario starts from high altitude and ends at ground
    scenarios = []
    current_scenario = []
    max_altitude_in_scenario = 0
    
    for idx, row in df.iterrows():
        altitude = row['altitude']
        
        # Track maximum altitude in current scenario
        if altitude > max_altitude_in_scenario:
            max_altitude_in_scenario = altitude
            current_scenario.append(row)
        # If altitude decreases and we had high altitude before, it's a descent
        elif max_altitude_in_scenario > 1 and altitude < 1:
            # End of descent - save this scenario
            current_scenario.append(row)
            if len(current_scenario) > 50:  # Only keep substantial scenarios
                scenarios.append(pd.DataFrame(current_scenario))
            current_scenario = []
            max_altitude_in_scenario = 0
        else:
            # Continue current scenario
            if altitude < max_altitude_in_scenario:
                current_scenario.append(row)
    
    if current_scenario and len(current_scenario) > 50:
        scenarios.append(pd.DataFrame(current_scenario))
    
    print(f"\nFound {len(scenarios)} complete landing scenarios")
    print(f"Average scenario length: {np.mean([len(s) for s in scenarios]):.0f} rows")
    
    # CORRECTED: Select scenarios that have good altitude range (not just ground hover)
    good_scenarios = []
    for scenario in scenarios:
        alt_range = scenario['altitude'].max() - scenario['altitude'].min()
        if alt_range > 30:  # Must show at least 30m descent
            good_scenarios.append(scenario)
    
    if len(good_scenarios) == 0:
        print("WARNING: No complete descent scenarios found. Using best available.")
        good_scenarios = sorted(scenarios, key=lambda s: s['altitude'].max() - s['altitude'].min(), reverse=True)[:num_samples]
    
    print(f"Selected {len(good_scenarios)} scenarios with altitude range > 30m")
    
    # Plot samples
    fig, axes = plt.subplots(num_samples, 3, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Select diverse scenarios
    sample_indices = np.random.choice(len(good_scenarios), size=min(num_samples, len(good_scenarios)), replace=False)
    
    for plot_idx, scenario_idx in enumerate(sample_indices):
        scenario_df = good_scenarios[scenario_idx]
        t_axis = np.arange(len(scenario_df)) * DT
        
        # Calculate altitude range and duration
        alt_min = scenario_df['altitude'].min()
        alt_max = scenario_df['altitude'].max()
        duration = len(scenario_df) * DT
        
        print(f"\nScenario {plot_idx + 1}:")
        print(f"  Altitude range: {alt_min:.2f} → {alt_max:.2f} m (drop: {alt_max - alt_min:.2f} m)")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Velocity range: {scenario_df['velocity'].min():.2f} → {scenario_df['velocity'].max():.2f} m/s")
        print(f"  Wind range: {scenario_df['wind'].min():.2f} → {scenario_df['wind'].max():.2f} m/s")
        print(f"  Thrust range: {scenario_df['thrust_adjustment'].min():.2f} → {scenario_df['thrust_adjustment'].max():.2f} N")
        
        # ===== LEFT COLUMN: ALTITUDE =====
        axes[plot_idx, 0].plot(t_axis, scenario_df['altitude'].values, 'b-', linewidth=2.5)
        axes[plot_idx, 0].fill_between(t_axis, 0, scenario_df['altitude'].values, alpha=0.2, color='blue')
        axes[plot_idx, 0].set_xlabel('Time (s)', fontsize=11)
        axes[plot_idx, 0].set_ylabel('Altitude (m)', fontsize=11, fontweight='bold')
        axes[plot_idx, 0].set_title(f'Scenario {scenario_idx + 1}: Complete Descent\n({alt_max:.1f}m → {alt_min:.1f}m in {duration:.1f}s)', 
                                     fontsize=12, fontweight='bold')
        axes[plot_idx, 0].grid(True, alpha=0.4, linestyle='--')
        axes[plot_idx, 0].set_xlim(0, t_axis.max())
        axes[plot_idx, 0].set_ylim(-2, alt_max * 1.1)
        
        # ===== MIDDLE COLUMN: VELOCITY =====
        axes[plot_idx, 1].plot(t_axis, scenario_df['velocity'].values, 'r-', linewidth=2.5, label='Descent velocity')
        axes[plot_idx, 1].axhline(SAFE_LANDING_VELOCITY, color='g', linestyle='--', linewidth=2, label=f'Safe threshold ({SAFE_LANDING_VELOCITY} m/s)')
        axes[plot_idx, 1].axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        axes[plot_idx, 1].set_xlabel('Time (s)', fontsize=11)
        axes[plot_idx, 1].set_ylabel('Velocity (m/s)', fontsize=11, fontweight='bold')
        axes[plot_idx, 1].set_title(f'Velocity Control\nAcceleration → Safe Landing', fontsize=12, fontweight='bold')
        axes[plot_idx, 1].legend(loc='best', fontsize=10)
        axes[plot_idx, 1].grid(True, alpha=0.4, linestyle='--')
        axes[plot_idx, 1].set_xlim(0, t_axis.max())
        
        # ===== RIGHT COLUMN: WIND & THRUST =====
        ax_wind = axes[plot_idx, 2]
        ax_thrust = ax_wind.twinx()
        
        line1 = ax_wind.plot(t_axis, scenario_df['wind'].values, 'g-', linewidth=2.5, label='Wind (correlated)')
        line2 = ax_thrust.plot(t_axis, scenario_df['thrust_adjustment'].values, 'orange', linewidth=2.5, 
                               label='Thrust adjustment', alpha=0.9)
        
        ax_wind.set_xlabel('Time (s)', fontsize=11)
        ax_wind.set_ylabel('Wind Speed (m/s)', fontsize=11, fontweight='bold', color='g')
        ax_thrust.set_ylabel('Thrust Adjustment (N)', fontsize=11, fontweight='bold', color='orange')
        ax_wind.set_title(f'Wind & Thrust Coupling\nAdaptive Control', fontsize=12, fontweight='bold')
        
        ax_wind.tick_params(axis='y', labelcolor='g')
        ax_thrust.tick_params(axis='y', labelcolor='orange')
        ax_wind.grid(True, alpha=0.4, linestyle='--')
        ax_wind.set_xlim(0, t_axis.max())
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_wind.legend(lines, labels, loc='upper left', fontsize=10)
        
        # Add text annotation
        ax_wind.text(0.98, 0.02, f'Duration: {duration:.1f}s\nAlt drop: {alt_max-alt_min:.1f}m',
                     transform=ax_wind.transAxes, fontsize=9,
                     verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('DRONE LANDING SIMULATION: Complete Descent Trajectories\n(Physics-Based with Correlated Wind)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('sample_trajectories.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Sample trajectories plot saved to sample_trajectories.png")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Generate dataset
    print("=" * 80)
    print("DRONE LANDING DATA GENERATOR - CORRECTED VERSION")
    print("=" * 80)
    
    df = generate_dataset(num_rows=10000, output_file='dataset.csv')
    
    # Plot sample trajectories (CORRECTED visualization)
    print("\nGenerating CORRECTED sample trajectory plots...")
    plot_sample_trajectories('dataset.csv', num_samples=3)
    
    print("\n" + "=" * 80)
    print("COMPLETE: dataset.csv and sample_trajectories.png ready for Phase 2")
    print("=" * 80)