"""
Diagnostic script: Verify coordinate system consistency between hover check and RL simulator.

This script creates a known-good quadcopter configuration, runs it through both:
1. The dronehover hover check (get_sim → Hover)
2. The RL simulator (DroneSimulator via _convert_individual_to_propellers)

Then simulates with constant thrust to see which direction the drone moves,
exposing any coordinate system mismatches.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np


def _convert_genome_to_propellers_ned(genome):
    """
    Convert genome to propellers using the FIXED NED conversion
    (same logic as _convert_individual_to_propellers after the fix).
    """
    valid_rows = ~np.isnan(genome).any(axis=1)
    individual_clean = genome[valid_rows]

    propellers = []
    for row in individual_clean:
        magnitude, arm_yaw, arm_pitch, mot_yaw, mot_pitch, direction = row

        # Position: spherical to ENU cartesian
        enu_x = magnitude * np.cos(arm_pitch) * np.cos(arm_yaw)
        enu_y = magnitude * np.cos(arm_pitch) * np.sin(arm_yaw)
        enu_z = magnitude * np.sin(arm_pitch)

        # ENU to NED: (x, y, z) → (y, x, -z)
        x, y, z = enu_y, enu_x, -enu_z

        # Thrust direction in NED frame (matches orientation_to_unit_vector)
        sp, cp = np.sin(mot_pitch), np.cos(mot_pitch)
        sy, cy = np.sin(mot_yaw), np.cos(mot_yaw)
        thrust_x = -sy * sp
        thrust_y = -cy * sp
        thrust_z = cp

        rotation = "cw" if direction > 0.5 else "ccw"
        propellers.append({
            "loc": [x, y, z],
            "dir": [thrust_x, thrust_y, thrust_z, rotation],
            "propsize": 2
        })
    return propellers


# =============================================================================
# PART 1: Compare how the same genome is converted by each path
# =============================================================================

def compare_conversions():
    """Compare get_sim vs _convert_individual_to_propellers for the same genome."""
    from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import (
        get_sim, orientation_to_unit_vector
    )
    import airevolve.evolution_tools.inspection_tools.utils as u

    print("=" * 80)
    print("PART 1: Compare conversion paths for the same genome")
    print("=" * 80)

    genome = np.array([
        [0.08,  np.pi/4,   0.0, 0.0, 0.0, 0.0],
        [0.08,  3*np.pi/4, 0.0, 0.0, 0.0, 1.0],
        [0.08, -3*np.pi/4, 0.0, 0.0, 0.0, 0.0],
        [0.08, -np.pi/4,   0.0, 0.0, 0.0, 1.0],
    ])

    print(f"\nGenome (mot_pitch=0, mot_yaw=0 for all motors):")
    print(f"  [mag, arm_yaw, arm_pitch, mot_yaw, mot_pitch, dir]")
    for i, row in enumerate(genome):
        print(f"  Arm {i}: {row}")

    print("\n--- Path A: Hover check (get_sim / hovering_info.py) ---")
    print("  Applies: convert_to_cartesian → ENU_to_NED for positions")
    print("  Applies: orientation_to_unit_vector (with internal ENU→NED) for thrust")

    for i, (mag, arm_yaw, arm_pitch, mot_yaw, mot_pitch, direction) in enumerate(genome):
        gx, gy, gz = u.convert_to_cartesian(mag, arm_yaw, arm_pitch)
        nx, ny, nz = u.ENU_to_NED(gx, gy, gz)
        thrust_ned = orientation_to_unit_vector(0, mot_pitch, mot_yaw)
        print(f"  Arm {i}: pos_NED=({nx:.4f}, {ny:.4f}, {nz:.4f}), thrust_NED=({thrust_ned[0]:.4f}, {thrust_ned[1]:.4f}, {thrust_ned[2]:.4f})")

    print("\n--- Path B: RL simulator (_convert_individual_to_propellers, FIXED) ---")
    print("  Now applies: spherical→ENU→NED for positions, orientation_to_unit_vector formula for thrust")

    props = _convert_genome_to_propellers_ned(genome)
    for i, p in enumerate(props):
        loc = p["loc"]
        d = p["dir"][:3]
        print(f"  Arm {i}: pos_NED=({loc[0]:.4f}, {loc[1]:.4f}, {loc[2]:.4f}), thrust_NED=({d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f}), rot={p['dir'][3]}")

    # Check match
    print("\n--- Match check ---")
    hover_sim = get_sim(genome)
    hover_props = hover_sim.drone.props
    all_match = True
    for i, (hp, rp) in enumerate(zip(hover_props, props)):
        pos_match = np.allclose(hp["loc"], rp["loc"], atol=1e-6)
        dir_match = np.allclose(hp["dir"][:3], rp["dir"][:3], atol=1e-6)
        status = "MATCH" if (pos_match and dir_match) else "MISMATCH"
        if not (pos_match and dir_match):
            all_match = False
        print(f"  Arm {i}: {status}  (pos: {'ok' if pos_match else 'DIFF'}, dir: {'ok' if dir_match else 'DIFF'})")
    if all_match:
        print("  All arms MATCH between hover check and RL simulator.")
    else:
        print("  WARNING: Some arms do NOT match!")


# =============================================================================
# PART 2: Run hover check on the genome
# =============================================================================

def run_hover_check():
    """Run the dronehover hover check and show results."""
    from airevolve.evolution_tools.genome_handlers.repair_workflow import (
        stage2_hover_check, stage3_hover_repair
    )

    print("\n" + "=" * 80)
    print("PART 2: Run hover check on standard quadcopter genome")
    print("=" * 80)

    genome_default = np.array([
        [0.08,  np.pi/4,   0.0, 0.0, 0.0, 0.0],
        [0.08,  3*np.pi/4, 0.0, 0.0, 0.0, 1.0],
        [0.08, -3*np.pi/4, 0.0, 0.0, 0.0, 0.0],
        [0.08, -np.pi/4,   0.0, 0.0, 0.0, 1.0],
    ])

    print("\n--- Genome with mot_pitch=0 (default) ---")
    can_hover, msg = stage2_hover_check(genome_default, verbose=True)
    print(f"  Result: can_hover={can_hover}, msg='{msg}'")

    genome_flipped = np.array([
        [0.08,  np.pi/4,   0.0, 0.0, np.pi, 0.0],
        [0.08,  3*np.pi/4, 0.0, 0.0, np.pi, 1.0],
        [0.08, -3*np.pi/4, 0.0, 0.0, np.pi, 0.0],
        [0.08, -np.pi/4,   0.0, 0.0, np.pi, 1.0],
    ])

    print("\n--- Genome with mot_pitch=pi (flipped) ---")
    can_hover, msg = stage2_hover_check(genome_flipped, verbose=True)
    print(f"  Result: can_hover={can_hover}, msg='{msg}'")

    for label, genome in [("mot_pitch=0", genome_default), ("mot_pitch=pi", genome_flipped)]:
        can_hover, _ = stage2_hover_check(genome, verbose=False)
        if can_hover:
            print(f"\n--- Hover repair on '{label}' genome ---")
            repaired, msg = stage3_hover_repair(genome, coordinate_system='spherical', verbose=True)
            if repaired is not None:
                print(f"  Repaired genome:\n{repaired}")
                return repaired, label

    return None, None


# =============================================================================
# PART 3: Simulate in DroneSimulator to see which way it goes
# =============================================================================

def simulate_with_thrust(genome, label):
    """Simulate the drone with constant thrust and observe z-axis movement."""
    from airevolve.simulator.simulation.drone_simulator import DroneSimulator

    print("\n" + "=" * 80)
    print(f"PART 3: Simulate drone in DroneSimulator (genome: {label})")
    print("=" * 80)

    # Convert using the FIXED NED conversion
    propellers = _convert_genome_to_propellers_ned(genome)

    print(f"\nPropellers passed to DroneSimulator (NED conversion):")
    for i, p in enumerate(propellers):
        print(f"  Motor {i}: loc={[f'{v:.4f}' for v in p['loc']]}, "
              f"dir={[f'{v:.4f}' for v in p['dir'][:3]]}, rot={p['dir'][3]}")

    sim = DroneSimulator(propellers=propellers, dt=0.001)

    print(f"\nDroneSimulator properties:")
    print(f"  Mass: {sim.mass:.4f} kg")
    print(f"  Gravity: {sim.g} m/s^2 (applied as +z force)")
    print(f"  Bf (force allocation matrix):\n{sim.Bf}")
    total_force_dir = sim.Bf.sum(axis=1)
    print(f"  Total force at all motors=1: [{total_force_dir[0]:.4f}, {total_force_dir[1]:.4f}, {total_force_dir[2]:.4f}]")
    print(f"  For hover, need Bf @ U = [0, 0, {-sim.g * sim.mass:.4f}]")

    # Test 1: No thrust - should fall under gravity (+z in NED)
    print("\n--- Test 1: Zero thrust (expect: z increases = falls in NED) ---")
    sim.reset()
    sim.set_state(position=[0, 0, -1.5])

    for step in range(100):
        sim.step(np.zeros(sim.num_motors))

    state = sim.get_state()
    z_change = state['position'][2] - (-1.5)
    print(f"  After 0.1s: z changed by {z_change:+.6f}, vel_z = {state['velocity'][2]:+.4f}")
    print(f"  Interpretation: {'FALLS (+z = down in NED)' if z_change > 0 else 'RISES (-z = up in NED)'}")

    # Test 2: Full thrust
    print("\n--- Test 2: Full thrust (all motors at 1.0) ---")
    sim.reset()
    sim.set_state(position=[0, 0, -1.5])

    for step in range(100):
        sim.step(np.ones(sim.num_motors))

    state = sim.get_state()
    z_change = state['position'][2] - (-1.5)
    print(f"  After 0.1s: z changed by {z_change:+.6f}, vel_z = {state['velocity'][2]:+.4f}")

    thrust_z_sign = np.sign(total_force_dir[2])
    gravity_z_sign = +1  # gravity is always +z in DroneSimulator

    if thrust_z_sign * gravity_z_sign > 0:
        print(f"  WARNING: Thrust and gravity both push in +z! Drone accelerates downward!")
    else:
        print(f"  OK: Thrust opposes gravity (thrust in -z, gravity in +z)")
        if z_change < 0:
            print(f"  Drone RISES (thrust overcomes gravity)")
        else:
            print(f"  Drone FALLS (thrust insufficient to overcome gravity)")

    # Test 3: Standard quad reference
    print("\n--- Test 3: Standard quad reference (dir=[0,0,-1]) ---")
    standard_props = [
        {"loc": [0.11, 0.11, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
        {"loc": [-0.11, 0.11, 0], "dir": [0, 0, -1, "cw"], "propsize": 5},
        {"loc": [-0.11, -0.11, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
        {"loc": [0.11, -0.11, 0], "dir": [0, 0, -1, "cw"], "propsize": 5}
    ]
    std_sim = DroneSimulator(propellers=standard_props, dt=0.001)
    std_total = std_sim.Bf.sum(axis=1)
    print(f"  Total force direction: {std_total}")

    std_sim.set_state(position=[0, 0, -1.5])
    for step in range(100):
        std_sim.step(np.ones(std_sim.num_motors) * 0.5)

    state = std_sim.get_state()
    z_change = state['position'][2] - (-1.5)
    print(f"  After 0.1s at 50% thrust: z changed by {z_change:+.6f}")
    print(f"  Standard quad: thrust in -z opposes gravity in +z → {'RISES' if z_change < 0 else 'FALLS'}")


# =============================================================================
# PART 4: Direct comparison of Bf matrices from both paths
# =============================================================================

def compare_allocation_matrices():
    """Compare Bf allocation matrices from hover check vs RL simulator."""
    from airevolve.evolution_tools.inspection_tools.morphological_descriptors.hovering_info import get_sim
    from airevolve.simulator.simulation.drone_simulator import DroneSimulator

    print("\n" + "=" * 80)
    print("PART 4: Compare Bf allocation matrices from both paths")
    print("=" * 80)

    genome = np.array([
        [0.08,  np.pi/4,   0.0, 0.0, 0.0, 0.0],
        [0.08,  3*np.pi/4, 0.0, 0.0, 0.0, 1.0],
        [0.08, -3*np.pi/4, 0.0, 0.0, 0.0, 0.0],
        [0.08, -np.pi/4,   0.0, 0.0, 0.0, 1.0],
    ])

    # Path A: Hover check (dronehover)
    print("\n--- Path A: dronehover Hover ---")
    hover_sim = get_sim(genome)
    if hover_sim is not None:
        hover_sim.compute_hover(verbose=False)
        print(f"  Bf:\n{hover_sim.Bf}")
        print(f"  static_success: {hover_sim.static_success}")
        if hasattr(hover_sim, 'eta') and hover_sim.eta is not None:
            total_force = hover_sim.Bf @ hover_sim.eta
            print(f"  Total hover force (Bf @ eta): {total_force}")

    # Path B: RL simulator (with NED fix)
    print("\n--- Path B: DroneSimulator (FIXED NED conversion) ---")
    propellers = _convert_genome_to_propellers_ned(genome)
    rl_sim = DroneSimulator(propellers=propellers, dt=0.01)
    print(f"  Bf:\n{rl_sim.Bf}")
    total_force = rl_sim.Bf.sum(axis=1)
    print(f"  Total force (all motors at 1): {total_force}")

    # Note: Bf values differ in magnitude because dronehover and DroneSimulator
    # use different propeller data (kf, wmax), but the DIRECTION should match
    if hover_sim is not None:
        hover_dir = hover_sim.Bf.sum(axis=1)
        hover_dir_norm = hover_dir / np.linalg.norm(hover_dir)
        rl_dir_norm = total_force / np.linalg.norm(total_force)
        print(f"\n  Force direction (normalized):")
        print(f"    dronehover:     {hover_dir_norm}")
        print(f"    DroneSimulator: {rl_dir_norm}")
        if np.allclose(hover_dir_norm, rl_dir_norm, atol=0.01):
            print(f"    MATCH: Both point in the same direction")
        else:
            print(f"    MISMATCH: Force directions differ!")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("COORDINATE SYSTEM DIAGNOSTIC")
    print("Checking consistency between hover check and RL simulator\n")

    compare_conversions()
    repaired_genome, label = run_hover_check()
    compare_allocation_matrices()

    if repaired_genome is not None:
        simulate_with_thrust(repaired_genome, f"repaired from {label}")
    else:
        genome_default = np.array([
            [0.08,  np.pi/4,   0.0, 0.0, 0.0, 0.0],
            [0.08,  3*np.pi/4, 0.0, 0.0, 0.0, 1.0],
            [0.08, -3*np.pi/4, 0.0, 0.0, 0.0, 0.0],
            [0.08, -np.pi/4,   0.0, 0.0, 0.0, 1.0],
        ])
        simulate_with_thrust(genome_default, "mot_pitch=0 (default)")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
