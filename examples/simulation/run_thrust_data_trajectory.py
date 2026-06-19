"""
Replay real motor thrust data through the simulator and record the trajectory.

Usage
-----
    python examples/simulation/run_thrust_data_trajectory.py

The script expects a CSV with one column per motor and one row per timestep.
Edit THRUST_CSV_PATH and the per-column mapping below to match your file.
The ground plane constant GROUND_Z_NED sets the z-floor (NED, so positive = below
origin). When the drone is on or below the ground the state is snapped back to
ground level and attitude is forced flat (level hover pose).

Outputs
-------
- Console: timestep-by-timestep summary
- Matplotlib 3D plot of the trajectory
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401 (registers 3D projection)

from airevolve.simulator.simulation.drone_simulator import DroneSimulator
from airevolve.simulator.simulation.propeller_data import get_propeller_specs

# ---------------------------------------------------------------------------
# Configuration — edit these to match your hardware and data file
# ---------------------------------------------------------------------------

# Path to your CSV file. Each row is one timestep; columns are per-motor thrust
# in Newtons. Leave as None to run with synthetic demo data instead.
THRUST_CSV_PATH = None  # e.g. "data/flight_01_thrusts.csv"

# Column indices in the CSV for each motor (adjust if your CSV has extra cols).
# Order must match the propeller layout defined below.
MOTOR_COLUMN_ORDER = [0, 1, 2, 3]  # [motor1_col, motor2_col, motor3_col, motor4_col]

# Simulation timestep. Should match the rate at which your data was recorded.
DT = 0.005  # seconds

# Propeller / airframe description — a standard 5-inch quad at 110 mm arms.
ARM_LENGTH = 0.11   # metres
PROP_SIZE  = 5      # inches

PROPELLERS = [
    {"loc": [ ARM_LENGTH * np.cos(np.pi/4), ARM_LENGTH * np.sin(np.pi/4), 0], "dir": [0, 0, -1, "ccw"], "propsize": PROP_SIZE},
    {"loc": [ ARM_LENGTH * np.cos(3*np.pi/4), ARM_LENGTH * np.sin(3*np.pi/4), 0], "dir": [0, 0, -1, "cw"],  "propsize": PROP_SIZE},
    {"loc": [ ARM_LENGTH * np.cos(5*np.pi/4), ARM_LENGTH * np.sin(5*np.pi/4), 0], "dir": [0, 0, -1, "ccw"], "propsize": PROP_SIZE},
    {"loc": [ ARM_LENGTH * np.cos(7*np.pi/4), ARM_LENGTH * np.sin(7*np.pi/4), 0], "dir": [0, 0, -1, "cw"],  "propsize": PROP_SIZE},
]

# Ground plane in NED coordinates (z positive = downward).
# The drone is considered "on the ground" when state[2] >= GROUND_Z_NED.
GROUND_Z_NED = 0.0  # 0 = world origin is the ground surface

# Initial conditions (NED)
INITIAL_POSITION = [0.0, 0.0, -0.5]  # start 0.5 m above ground
INITIAL_VELOCITY = [0.0, 0.0, 0.0]
INITIAL_ATTITUDE = [0.0, 0.0, 0.0]   # level, nose-north

# ---------------------------------------------------------------------------
# Thrust → action conversion
# ---------------------------------------------------------------------------

def thrust_to_action(thrust_N: float, k_f: float, w_min: float, w_max: float, k: float) -> float:
    """Convert a measured motor thrust (N) to the simulator action in [-1, 1].

    The simulator action pipeline is:
        action u ∈ [-1, 1]
        → U = (u + 1) / 2 ∈ [0, 1]
        → Wc = (w_max - w_min) · sqrt(k · U² + (1-k) · U) + w_min   (rad/s)
        → T  = k_f · Wc²

    Invert: T → Wc → U → u.
    """
    thrust_N = float(np.clip(thrust_N, 0.0, k_f * w_max ** 2))

    # Wc from thrust
    Wc = np.sqrt(max(thrust_N / k_f, 0.0))
    Wc = float(np.clip(Wc, w_min, w_max))

    # Invert sqrt-poly: Wc = (w_max - w_min) · sqrt(k·U² + (1-k)·U) + w_min
    z = (Wc - w_min) / (w_max - w_min)          # ∈ [0, 1]
    z2 = z * z
    if abs(k) < 1e-12:
        U = z2
    else:
        disc = (1.0 - k) ** 2 + 4.0 * k * z2
        U = (-(1.0 - k) + np.sqrt(disc)) / (2.0 * k)

    return float(np.clip(2.0 * U - 1.0, -1.0, 1.0))


def thrusts_to_actions(thrusts: np.ndarray, k_f: float, w_min: float, w_max: float, k: float) -> np.ndarray:
    return np.array([thrust_to_action(t, k_f, w_min, w_max, k) for t in thrusts])


# ---------------------------------------------------------------------------
# Ground plane enforcement
# ---------------------------------------------------------------------------

def apply_ground_plane(state: np.ndarray, ground_z: float) -> np.ndarray:
    """If the drone is at or below the ground, snap it back and lay it flat.

    NED convention: z increases downward, so 'below ground' means state[2] >= ground_z.
    When grounded:
    - Position z is clamped to ground_z.
    - Vertical velocity (vz) is zeroed if downward.
    - Attitude is reset to flat (roll=pitch=0; yaw preserved).
    - Angular rates are zeroed.
    """
    if state[2] >= ground_z:
        state = state.copy()
        state[2] = ground_z           # snap to ground surface
        state[5] = min(state[5], 0.0) # remove downward velocity (vz ≥ 0 = downward in NED)
        state[6] = 0.0                # roll  = 0
        state[7] = 0.0                # pitch = 0
        # state[8] yaw is preserved
        state[9]  = 0.0               # p
        state[10] = 0.0               # q
        state[11] = 0.0               # r
    return state


# ---------------------------------------------------------------------------
# Synthetic demo data (used when THRUST_CSV_PATH is None)
# ---------------------------------------------------------------------------

def make_demo_thrusts(drone: DroneSimulator, duration: float = 5.0) -> np.ndarray:
    """Generate a simple thrust sequence: spool up → hover → slow descent."""
    steps = int(duration / DT)
    n = drone.num_motors
    m = drone.mass

    # Hover thrust per motor
    hover_thrust = m * 9.81 / n

    thrusts = np.zeros((steps, n))
    for step in range(steps):
        t = step * DT
        if t < 1.0:
            # Ramp from 0 → hover
            frac = t / 1.0
            thrusts[step] = hover_thrust * frac
        elif t < 3.5:
            # Hover (slight over-thrust to climb then settle)
            climb = 1.1 if t < 2.0 else 1.0
            thrusts[step] = hover_thrust * climb
        else:
            # Controlled descent
            thrusts[step] = hover_thrust * 0.7

    return thrusts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Build simulator
    drone = DroneSimulator(propellers=PROPELLERS, dt=DT)

    # Fetch motor model constants for thrust→action conversion
    prop_specs = get_propeller_specs(PROP_SIZE)
    k_f    = prop_specs["constants"][0]
    w_min  = prop_specs["w_min"]
    w_max  = float(prop_specs["wmax"])
    k_poly = prop_specs["k"]

    print(f"Drone mass : {drone.mass:.3f} kg")
    print(f"Hover thrust / motor: {drone.mass * 9.81 / drone.num_motors:.4f} N")
    print(f"k_f        : {k_f:.3e} N/(rad/s)²")
    print(f"Motor range: {w_min:.1f} – {w_max:.1f} rad/s")
    print()

    # Load or synthesise thrust data
    if THRUST_CSV_PATH is not None:
        raw = np.loadtxt(THRUST_CSV_PATH, delimiter=",")
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        thrust_data = raw[:, MOTOR_COLUMN_ORDER]  # shape (T, N)
        print(f"Loaded {len(thrust_data)} timesteps from {THRUST_CSV_PATH}")
    else:
        thrust_data = make_demo_thrusts(drone, duration=5.0)
        print(f"Using synthetic demo data: {len(thrust_data)} timesteps × {drone.num_motors} motors")

    # Set initial conditions
    drone.set_state(
        position=INITIAL_POSITION,
        velocity=INITIAL_VELOCITY,
        attitude=INITIAL_ATTITUDE,
        angular_velocity=[0.0, 0.0, 0.0],
    )

    # Run simulation
    trajectory = []
    times = []
    on_ground_count = 0

    for step, thrusts in enumerate(thrust_data):
        t = step * DT

        # Enforce ground plane before stepping (clamp state if needed)
        drone.state = apply_ground_plane(drone.state, GROUND_Z_NED)

        # Convert real thrusts → simulator actions
        n_motors = drone.num_motors
        if len(thrusts) < n_motors:
            # Pad with zeros if fewer columns than motors
            thrusts = np.pad(thrusts, (0, n_motors - len(thrusts)))
        actions = thrusts_to_actions(thrusts[:n_motors], k_f, w_min, w_max, k_poly)

        try:
            drone.step(actions)
        except RuntimeError as exc:
            print(f"Simulation diverged at t={t:.3f}s: {exc}")
            break

        state = drone.get_state()
        pos = state["position"]
        att = state["attitude"]

        on_ground = pos[2] >= GROUND_Z_NED
        if on_ground:
            on_ground_count += 1

        trajectory.append(pos.copy())
        times.append(t)

        if step % 50 == 0 or on_ground:
            label = " [GROUND]" if on_ground else ""
            print(
                f"t={t:6.3f}s  pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]  "
                f"att=[{np.degrees(att[0]):+5.1f}°, {np.degrees(att[1]):+5.1f}°, {np.degrees(att[2]):+5.1f}°]"
                f"{label}"
            )

    trajectory = np.array(trajectory)
    times = np.array(times)

    print(f"\nSimulation complete. Steps on ground: {on_ground_count}/{len(times)}")

    # -----------------------------------------------------------------------
    # Plot trajectory
    # -----------------------------------------------------------------------
    # NED → display: flip z so up is positive on the plot
    x_plot = trajectory[:, 0]
    y_plot = trajectory[:, 1]
    z_plot = -trajectory[:, 2]  # NED z negated for intuitive "up" display

    fig = plt.figure(figsize=(12, 5))

    # 3D trajectory
    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.plot(x_plot, y_plot, z_plot, "b-", linewidth=1.5, label="trajectory")
    ax3d.scatter(x_plot[0], y_plot[0], z_plot[0], c="g", s=60, label="start", zorder=5)
    ax3d.scatter(x_plot[-1], y_plot[-1], z_plot[-1], c="r", s=60, label="end", zorder=5)

    # Ground plane patch
    x_range = max(np.ptp(x_plot), 0.5)
    y_range = max(np.ptp(y_plot), 0.5)
    gx = np.linspace(x_plot.min() - 0.1 * x_range, x_plot.max() + 0.1 * x_range, 2)
    gy = np.linspace(y_plot.min() - 0.1 * y_range, y_plot.max() + 0.1 * y_range, 2)
    gX, gY = np.meshgrid(gx, gy)
    gZ = np.zeros_like(gX) - GROUND_Z_NED  # ground at z=0 in display coords
    ax3d.plot_surface(gX, gY, gZ, alpha=0.15, color="brown", label="ground")

    ax3d.set_xlabel("x (m, North)")
    ax3d.set_ylabel("y (m, East)")
    ax3d.set_zlabel("z (m, up)")
    ax3d.set_title("3D Trajectory")
    ax3d.legend(fontsize=8)

    # Altitude vs time
    ax2 = fig.add_subplot(122)
    ax2.plot(times, z_plot, "b-", linewidth=1.5, label="altitude")
    ax2.axhline(-GROUND_Z_NED, color="brown", linestyle="--", linewidth=1, label="ground")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("altitude (m, up positive)")
    ax2.set_title("Altitude vs Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("trajectory.png", dpi=150)
    print("Plot saved to trajectory.png")
    plt.show()


if __name__ == "__main__":
    main()
