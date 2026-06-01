import sys
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from airevolve.simulator.simulation.battery_model import LiPoBatteryModel
from airevolve.evolution_tools.evaluators.drone_gate_env_power import PowerAwareDroneEnv
from airevolve.evolution_tools.evaluators.gate_train import figure8

SEPARATOR = "=" * 70
SUB_SEP   = "-" * 70


def _make_hex_individual():
    r = 0.11
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    rows = []
    for i, a in enumerate(angles):
        rows.append([r, a, 0.0, 0.0, 0.0, 1.0 if i % 2 == 0 else 0.0])
    return np.array(rows, dtype=np.float32)


def _make_env(experiment_type, sparse_weight, overdraw_penalty_weight,
              strict_voltage_kill, max_steps):
    cfg = figure8
    individual = _make_hex_individual()
    return PowerAwareDroneEnv(
        num_envs=1,
        individual=individual,
        gates_pos=cfg.gate_pos,
        gate_yaw=cfg.gate_yaw,
        start_pos=cfg.starting_pos,
        x_bounds=cfg.x_bounds,
        y_bounds=cfg.y_bounds,
        z_bounds=cfg.z_bounds,
        gates_ahead=1,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device="cpu",
        max_steps=max_steps,
        experiment_type=experiment_type,
        penalty_weights={"sparse_weight": sparse_weight},
        randomize_soc=False,
        initialize_at_random_gates=False,
        strict_voltage_kill=strict_voltage_kill,
        overdraw_penalty_weight=overdraw_penalty_weight,
    )


def _run_to_done(env, fixed_action):
    obs = env.reset()
    done = False
    last_reward = None
    last_info = None
    while not done:
        action = np.tile(fixed_action, (1, 1))
        env.step_async(action)
        obs, rewards, dones, infos = env.step_wait()
        last_reward = float(rewards[0])
        last_info = infos[0]
        done = bool(dones[0])
    return last_reward, last_info


def test_rl_ppo_branch():
    print(SEPARATOR)
    print("TEST 1 — RL / PPO Branch: Timestep Overdraw + Episodic Sparse Penalty")
    print(SEPARATOR)

    OVERDRAW_WEIGHT  = 0.01
    SPARSE_WEIGHT    = 0.002
    MAX_STEPS        = 8
    BATTERY_MAX_A    = LiPoBatteryModel.BATTERY_MAX_CURRENT
    N_MOTORS         = 6

    print(f"  overdraw_penalty_weight : {OVERDRAW_WEIGHT}")
    print(f"  sparse_weight           : {SPARSE_WEIGHT}")
    print(f"  strict_voltage_kill     : False  (PPO training mode)")
    print(f"  max_steps               : {MAX_STEPS}")
    print()

    print(SUB_SEP)
    print("  1A — Timestep Overdraw Penalty")
    print(SUB_SEP)

    bat_ref      = LiPoBatteryModel()
    full_action  = np.ones(N_MOTORS, dtype=np.float32)
    i_theoretical = bat_ref.compute_current_from_rpms(full_action, max_rpm=1.0)
    delta_i      = max(0.0, i_theoretical - BATTERY_MAX_A)
    expected_step_penalty = OVERDRAW_WEIGHT * delta_i

    env_with = _make_env(2, SPARSE_WEIGHT, OVERDRAW_WEIGHT, False, MAX_STEPS)
    env_zero = _make_env(2, 0.0,           0.0,             False, MAX_STEPS)

    np.random.seed(0)
    env_with.reset()
    np.random.seed(0)
    env_zero.reset()

    action_batch = np.tile(full_action, (1, 1))

    env_with.step_async(action_batch)
    _, rewards_with, dones_with, infos_with = env_with.step_wait()

    env_zero.step_async(action_batch)
    _, rewards_zero, dones_zero, infos_zero = env_zero.step_wait()

    actual_current  = env_with.bat_current[0]
    terminal_voltage = env_with.bat_voltage[0]
    battery_alive   = not env_with.bat_is_depleted[0]

    measured_step_penalty = float(rewards_zero[0]) - float(rewards_with[0])

    print(f"  Requested action (all motors) : {full_action.tolist()}")
    print(f"  Theoretical current           : {i_theoretical:.4f} A")
    print(f"  Battery current limit         : {BATTERY_MAX_A} A")
    print(f"  Actual current drawn (clamped): {actual_current:.4f} A")
    print(f"  Limiter fired                 : {'YES ✓' if i_theoretical > BATTERY_MAX_A else 'NO ✗'}")
    print()
    print(f"  Terminal voltage              : {terminal_voltage:.4f} V")
    print(f"  battery.is_depleted           : {env_with.bat_is_depleted[0]}  "
          f"({'alive ✓' if battery_alive else 'dead ✗'})")
    print()
    print(f"  Delta I (overdraw)            : {delta_i:.4f} A")
    print(f"  Expected step penalty         : {OVERDRAW_WEIGHT} × {delta_i:.4f} = {expected_step_penalty:.6f}")
    print(f"  Measured reward reduction     : {measured_step_penalty:.6f}  "
          f"{'✓' if abs(measured_step_penalty - expected_step_penalty) < 1e-6 else '✗'}")

    step_ok = (
        actual_current <= BATTERY_MAX_A + 2.0
        and battery_alive
        and abs(measured_step_penalty - expected_step_penalty) < 1e-6
    )
    print()
    print(f"  1A RESULT: Limiter OK, Battery alive, Step penalty correct → "
          f"{'PASS ✓' if step_ok else 'FAIL ✗'}")

    env_with.close()
    env_zero.close()

    print()
    print(SUB_SEP)
    print("  1B — Episodic Sparse Energy Penalty")
    print(SUB_SEP)

    env_sparse   = _make_env(2, SPARSE_WEIGHT, 0.0, False, MAX_STEPS)
    env_baseline = _make_env(2, 0.0,           0.0, False, MAX_STEPS)

    idle_action  = np.zeros(N_MOTORS, dtype=np.float32)
    r_sparse,   info_sparse   = _run_to_done(env_sparse,   idle_action)
    r_baseline, info_baseline = _run_to_done(env_baseline, idle_action)

    ep_energy_j      = info_sparse["battery_energy_j"]
    expected_ep_penalty = SPARSE_WEIGHT * ep_energy_j
    measured_ep_penalty = r_baseline - r_sparse

    print(f"  Action used (idle — no overdraw) : {idle_action.tolist()}")
    print(f"  Episode length                   : {MAX_STEPS} steps")
    print()
    print(f"  Total energy consumed (J)        : {ep_energy_j:.6f} J")
    print(f"  Expected sparse penalty          : {SPARSE_WEIGHT} × {ep_energy_j:.6f} = {expected_ep_penalty:.6f}")
    print(f"  Baseline terminal reward         : {r_baseline:.6f}")
    print(f"  Penalized terminal reward        : {r_sparse:.6f}")
    print(f"  Measured reward reduction        : {measured_ep_penalty:.6f}  "
          f"{'✓' if abs(measured_ep_penalty - expected_ep_penalty) < 1e-6 else '✗'}")

    ep_ok = abs(measured_ep_penalty - expected_ep_penalty) < 1e-6
    print()
    print(f"  1B RESULT: Episodic sparse penalty correct → {'PASS ✓' if ep_ok else 'FAIL ✗'}")

    env_sparse.close()
    env_baseline.close()

    all_pass = step_ok and ep_ok
    print()
    print(f"  {'>>> TEST 1 PASSED <<<' if all_pass else '>>> TEST 1 FAILED <<<'}")
    return all_pass


def test_ea_nsga2_branch():
    print()
    print(SEPARATOR)
    print("TEST 2 — EA / NSGA-II Branch: Strict Voltage Kill Switch")
    print(SEPARATOR)

    THROTTLE          = 0.90
    DT                = 0.01
    MAX_STEPS         = 1200
    VOLTAGE_THRESHOLD = LiPoBatteryModel.VOLTAGE_DEPLETED
    SOC_THRESHOLD     = LiPoBatteryModel.SOC_DEPLETION_THRESHOLD
    motor_rpms        = np.full(6, THROTTLE)

    bat_ref      = LiPoBatteryModel()
    i_at_throttle = bat_ref.compute_current_from_rpms(motor_rpms, max_rpm=1.0)

    print(f"  strict_voltage_kill : True  (NSGA-II eval mode)")
    print(f"  Throttle            : {THROTTLE * 100:.0f}% × 6 motors")
    print(f"  Current at throttle : {i_at_throttle:.3f} A")
    print(f"  Simulation window   : {MAX_STEPS} steps @ dt={DT}s  ({MAX_STEPS * DT:.1f} s max)")
    print(f"  Kill thresholds     : SoC < {SOC_THRESHOLD*100:.0f}%  OR  V_terminal < {VOLTAGE_THRESHOLD} V")
    print()

    battery = LiPoBatteryModel(strict_voltage_kill=True)
    battery.reset()

    death_step    = None
    death_time    = None
    death_voltage = None
    death_soc     = None

    for step in range(MAX_STEPS):
        result = battery.step(DT, motor_rpms, max_rpm=1.0)
        if battery.is_depleted:
            death_step    = step + 1
            death_time    = result["time"] + DT
            death_voltage = result["voltage"]
            death_soc     = result["soc"]
            break

    total_energy_j = battery.get_total_energy_consumed()

    if death_step is not None:
        voltage_kill = death_voltage < VOLTAGE_THRESHOLD
        soc_kill     = death_soc    < SOC_THRESHOLD

        print(f"  ── Depletion Event ──────────────────────────────────────────")
        print(f"  Killed at step      : {death_step} / {MAX_STEPS}")
        print(f"  Simulation time     : {death_time:.3f} s  (of {MAX_STEPS * DT:.1f} s window)")
        print(f"  Terminal voltage    : {death_voltage:.4f} V  (threshold < {VOLTAGE_THRESHOLD} V)  "
              f"{'→ VOLTAGE KILL ✓' if voltage_kill else ''}")
        print(f"  State of Charge     : {death_soc * 100:.2f}%  (threshold < {SOC_THRESHOLD*100:.0f}%)  "
              f"{'→ SOC KILL ✓' if soc_kill else ''}")
        print()
        print(f"  ── Energy Budget ────────────────────────────────────────────")
        print(f"  Total energy consumed : {total_energy_j:.4f} J")
        print(f"  Average power draw    : {total_energy_j / death_time:.4f} W")
        print()

        killed_correctly = voltage_kill or soc_kill
        died_mid_eval    = death_step < MAX_STEPS

        print(f"  RESULT: Killed by threshold={killed_correctly}  |  "
              f"Died before end of window={died_mid_eval}")
        all_pass = killed_correctly and died_mid_eval
        print(f"  {'>>> TEST 2 PASSED <<<' if all_pass else '>>> TEST 2 FAILED <<<'}")
        return all_pass
    else:
        print(f"  Battery survived all {MAX_STEPS} steps — kill switch did NOT fire.")
        print(f"  >>> TEST 2 FAILED: expected depletion did not occur <<<")
        return False


if __name__ == "__main__":
    print()
    print(SEPARATOR)
    print("  Airevolve Power Architecture Verification")
    print("  Motor-Level ECM | Current Limiter | Timestep + Sparse Penalties | Kill Switch")
    print(SEPARATOR)
    print()

    t1 = test_rl_ppo_branch()
    t2 = test_ea_nsga2_branch()

    print()
    print(SEPARATOR)
    overall = t1 and t2
    print(f"  OVERALL: {'ALL TESTS PASSED ✓' if overall else 'SOME TESTS FAILED ✗'}")
    print(SEPARATOR)
    print()

    sys.exit(0 if overall else 1)
