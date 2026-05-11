"""
LiPo Battery Model for Airevolve Drone Physics Simulation.

Models the Tattu R-Line 750mAh 14.8V 95C 4S / Tattu 450mAh 14.8V 75C 4S lithium polymer battery using an
Equivalent Circuit Model (ECM) with SoC-based voltage curve and dynamic
flight-mode classification for realistic current-draw estimation.

COMPATIBILITY NOTE FOR MORPHOLOGY / URDF GENERATORS:
  External code that builds drone geometry or URDF files MUST use:
    - BATTERY_MASS = 0.082 kg  (defined in propeller_data.py)
    - BATTERY_POS  = [0.0, 0.0, 0.02]  (NED frame, defined in drone_configuration.py)
  These constants are what the inertial model in DroneConfiguration uses.
  Changing either without updating this class (or vice-versa) will introduce
  a mismatch between the electrical model and the physical dynamics model.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


class LiPoBatteryModel:
    """
    Supports two calling conventions for step():
      Legacy:  step(dt, current_draw_amps: float)   — explicit current in amps
      Modal:   step(dt, kinematic_state: array-like) — auto-classifies flight mode

    Integration hooks
    -----------------
    get_rl_observations()       -> np.ndarray [soc, voltage, power]  for PPO obs space
    get_total_energy_consumed() -> float  Joules  for EA fitness
    get_mass_and_dimensions()   -> (mass_g, (L_mm, W_mm, H_mm))  for URDF builder
    get_history()               -> pd.DataFrame or dict  for plotting / logging
    """

    # ------------------------------------------------------------------ #
    # Battery specifications — Tattu R-Line 750mAh 14.8V 95C 4S          #
    # ------------------------------------------------------------------ #
    BATTERY_NAME = "Tattu 450mAh 14.8V 75C 4S"
    NUM_CELLS = 4

    # Electrical
    CAPACITY_MAH = 450.0          # mAh  750 or 450
    CAPACITY_AH  = 0.450          # Ah  (= CAPACITY_MAH / 1000)
    VOLTAGE_FULL     = 16.8       # V   (4 × 4.20 V/cell, fully charged)
    VOLTAGE_NOMINAL  = 14.8       # V   (4 × 3.70 V/cell, nominal)
    VOLTAGE_DEPLETED = 12.8       # V   (4 × 3.20 V/cell, lower cutoff)
    C_RATING     = 75             # 95 or 75
    MAX_CURRENT  = 33.75          # A   71.25 or 33.75

    # ECM internal resistance — 60 mΩ is a validated value for a high-discharge
    # 4S racing pack.  At 18 A takeoff this causes 1.08 V sag (≈6% of nominal),
    # accurately capturing the under-voltage dip that differentiates idle from
    # full-throttle flight in the energy model.
    INTERNAL_RESISTANCE = 0.06    # Ω

    # Physical
    # NOTE: MASS_KG must equal BATTERY_MASS in propeller_data.py (currently 0.082 kg).
    # NOTE: DIMENSIONS_MM must match any URDF / CAD generator geometry.
    MASS_G         = 55           # grams  82 or 55
    MASS_KG        = 0.055        # kg   — matches BATTERY_MASS in propeller_data.py
    DIMENSIONS_MM  = (45, 24, 27) # mm     (60 x 31 x 28) or (45 x 24 x 27)

    # Depletion thresholds
    SOC_DEPLETION_THRESHOLD     = 0.15  # 15 % SoC commercial safety reserve
    VOLTAGE_DEPLETION_THRESHOLD = 13.2  # V  (4 × 3.30 V/cell)

    # ------------------------------------------------------------------ #
    # Flight modes — hardware-measured current draws for the sub-250 g   #
    # hexacopter: Flywoo GN405 FC, EMAX ECO 1404 3700KV × 6,            #
    # Gemfan 2609 props, Tattu R-Line 750mAh 14.8V 4S 95C battery.      #
    # ------------------------------------------------------------------ #
    FLIGHT_MODE_CURRENTS: dict[str, float] = {
        "Ground":  0.15,    # A — FC + ESC idle, props stationary
        "Hover":   4.50,    # A — altitude hold, minimal translation
        "Move":    6.50,    # A — lateral / forward flight during figure-8
        "Takeoff": 18.00,   # A — full-throttle climb at mission start
    }

    # Mode-detection thresholds for _classify_mode() (state in NED frame).
    # In NED: z < 0 is airborne, z ≈ 0 is ground level, vz < 0 is ascending.
    _ALTITUDE_GROUND_THRESHOLD = 0.05   # m  : altitude ≤ this → Ground mode
    _TAKEOFF_VZ_THRESHOLD      = 0.30   # m/s: |vz| > this AND airborne → Takeoff
    _MOVE_VXY_THRESHOLD        = 0.30   # m/s: √(vx²+vy²) > this → Move

    # Legacy Open-Circuit Voltage lookup table — kept for reference; ECM formula is the live path.
    _OCV_SOC = np.array([0.00, 0.05, 0.20, 0.80, 1.00])
    _OCV_V   = np.array([12.80, 13.40, 14.40, 16.00, 16.80])

    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all state to a fully-charged battery."""
        self._soc: float = 1.0
        self._capacity_remaining_ah: float = self.CAPACITY_AH
        self._total_time_s: float = 0.0
        self._total_energy_j: float = 0.0
        self.is_depleted: bool = False

        # Last-step cache (used by get_rl_observations before first step)
        self._last_voltage: float = self.VOLTAGE_FULL
        self._last_power: float = 0.0
        self._last_mode: str = "Ground"

        # Per-step history lists — converted to arrays / DataFrame at get_history()
        self._hist_time:         list[float] = []
        self._hist_mode:         list[str]   = []
        self._hist_voltage:      list[float] = []
        self._hist_soc:          list[float] = []
        self._hist_capacity_mah: list[float] = []
        self._hist_power:        list[float] = []
        self._hist_current:      list[float] = []
        self._hist_energy:       list[float] = []

    # ------------------------------------------------------------------ #
    # Core ECM physics                                                    #
    # ------------------------------------------------------------------ #

    def _compute_ecm_voltage(self, soc: float, current: float) -> float:
        """
        ECM terminal voltage using a square-root SoC curve and R_internal sag.

          V_ideal  = 13.0 + (16.8 − 13.0) × √SoC
          V_actual = V_ideal − I × R_internal   (clamped to VOLTAGE_DEPLETED)

        The √SoC shape follows the empirical discharge profile of a 4S LiPo:
        fast drop near full charge, long flat mid-range plateau, steep knee
        at low SoC.  R_internal = 0.06 Ω models the dominant DC series resistance
        of the pack at room temperature under high-rate discharge.
        """
        soc_clamped = max(0.0, min(1.0, soc))
        v_ideal  = 13.0 + (16.8 - 13.0) * math.pow(soc_clamped, 0.5)
        v_actual = v_ideal - (current * self.INTERNAL_RESISTANCE)
        return max(self.VOLTAGE_DEPLETED, v_actual)

    def _compute_ocv(self, soc: float) -> float:
        """Piecewise-linear OCV from the lookup table (legacy reference path)."""
        return float(np.interp(np.clip(soc, 0.0, 1.0), self._OCV_SOC, self._OCV_V))

    def _classify_mode(self, state) -> str:
        """
        Classify flight mode from a kinematic state vector (NED frame).

        State layout (from DroneGateEnv.world_states):
            [0:3]  position  (x, y, z_NED)   — z < 0 when airborne
            [3:6]  velocity  (vx, vy, vz)    — vz < 0 when ascending
            [6:9]  attitude  (phi, theta, psi)
            [9:12] ang. rate (p, q, r)
            [12:]  motor speeds

        Priority ladder (highest-power mode wins ties):
            Ground  → altitude ≤ _ALTITUDE_GROUND_THRESHOLD
            Takeoff → airborne AND |vz| > _TAKEOFF_VZ_THRESHOLD
            Move    → airborne AND √(vx²+vy²) > _MOVE_VXY_THRESHOLD
            Hover   → airborne, low velocity
        """
        z_ned = float(state[2])
        vx    = float(state[3])
        vy    = float(state[4])
        vz    = float(state[5])

        # altitude is positive when the drone is airborne (NED z is negative when up)
        altitude  = -z_ned
        horiz_spd = math.sqrt(vx * vx + vy * vy)

        if altitude <= self._ALTITUDE_GROUND_THRESHOLD:
            return "Ground"
        if abs(vz) > self._TAKEOFF_VZ_THRESHOLD:
            return "Takeoff"
        if horiz_spd > self._MOVE_VXY_THRESHOLD:
            return "Move"
        return "Hover"

    def step(self, dt: float, state_or_current) -> dict:
        """
        Advance the battery model by one timestep.

        Two calling conventions
        -----------------------
        Legacy (direct current draw):
            step(dt, current_draw_amps: float)
            The caller supplies the exact current in amps.  Useful for fixed-
            profile test scripts (e.g. test_battery_model.py phase replays).
            The mode field in the returned dict and history is empty string "".

        Modal (kinematic state → automatic mode classification):
            step(dt, kinematic_state: array-like)
            Accepts the full drone state vector from DroneGateEnv.world_states[i]
            and maps it to a FLIGHT_MODE_CURRENTS entry via _classify_mode().

        Parameters
        ----------
        dt : float
            Timestep in seconds.  Matches DroneSimulator / DroneGateEnv dt
            (typically 0.005–0.01 s).
        state_or_current : float | array-like
            A scalar current [A] (legacy path) or the drone's 12+N element
            kinematic state vector (modal path).

        Returns
        -------
        dict with keys:
            time          — simulation time at the START of this step [s]
            mode          — flight mode string ("Ground"/"Hover"/"Move"/"Takeoff"
                            or "" for the legacy scalar path)
            soc           — State of Charge in [0, 1]
            voltage       — ECM terminal voltage [V]
            current       — actual current drawn this step [A]
            capacity_mah  — remaining capacity [mAh]
            power         — instantaneous power [W]
            total_energy_j — cumulative energy consumed [J]
        """
        t_now = self._total_time_s

        # ── Dispatch on argument type ──────────────────────────────────── #
        if isinstance(state_or_current, (int, float, np.floating)):
            # Legacy path: caller provides explicit current [A]
            mode    = ""
            current = float(np.clip(state_or_current, 0.0, self.MAX_CURRENT))
        else:
            # Modal path: classify from kinematic state, look up current
            mode    = self._classify_mode(state_or_current)
            current = self.FLIGHT_MODE_CURRENTS[mode]

        # BMS cutoff — no further drain once the safety threshold is breached
        if self.is_depleted:
            current = 0.0

        # Capacity drain: Δ_Ah = I [A] × dt [s] / 3600 [s/h]
        # Unit note: dt is in seconds; dividing by 3600 converts to hours so
        # the product with current (Amps) yields capacity consumed in Amp-hours.
        delta_ah = current * (dt / 3600.0)
        self._capacity_remaining_ah = max(0.0, self._capacity_remaining_ah - delta_ah)

        # SoC update
        self._soc = float(np.clip(
            self._capacity_remaining_ah / self.CAPACITY_AH, 0.0, 1.0
        ))

        # ECM terminal voltage (SoC curve + R_internal sag)
        v_terminal = self._compute_ecm_voltage(self._soc, current)

        # Instantaneous power [W] and cumulative energy [J]
        power_w = v_terminal * current
        self._total_energy_j += power_w * dt

        # Advance simulation clock
        self._total_time_s += dt

        # Depletion check — evaluated AFTER update so the breaching step is recorded
        if (self._soc < self.SOC_DEPLETION_THRESHOLD or
                v_terminal < self.VOLTAGE_DEPLETION_THRESHOLD):
            self.is_depleted = True

        # Persist for property accessors and RL observation cache
        self._last_voltage = v_terminal
        self._last_power   = power_w
        self._last_mode    = mode

        # Append to per-step history
        # capacity_mah: Ah × 1000 → mAh
        self._hist_time.append(t_now)
        self._hist_mode.append(mode)
        self._hist_voltage.append(v_terminal)
        self._hist_soc.append(self._soc)
        self._hist_capacity_mah.append(self._capacity_remaining_ah * 1000.0)
        self._hist_power.append(power_w)
        self._hist_current.append(current)
        self._hist_energy.append(self._total_energy_j)

        return {
            'time':           t_now,
            'mode':           mode,
            'soc':            self._soc,
            'voltage':        v_terminal,
            'current':        current,
            'capacity_mah':   self._capacity_remaining_ah * 1000.0,
            'power':          power_w,
            'total_energy_j': self._total_energy_j,
        }

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def soc(self) -> float:
        """Current State of Charge in [0, 1]."""
        return self._soc

    @property
    def voltage(self) -> float:
        """Current ECM terminal voltage in Volts."""
        return self._last_voltage

    @property
    def mode(self) -> str:
        """Last classified flight mode string."""
        return self._last_mode

    # ------------------------------------------------------------------ #
    # Integration hooks                                                   #
    # ------------------------------------------------------------------ #

    def get_rl_observations(self) -> np.ndarray:
        """
        Battery observations for the PPO agent's observation space.

        Returns
        -------
        np.ndarray, shape (3,), dtype float32
            [current_soc, current_voltage_V, instantaneous_power_W]

        The observation size is fixed (mode is not included) so that previously
        trained policies remain compatible without retraining.
        """
        return np.array([self._soc, self._last_voltage, self._last_power], dtype=np.float32)

    def get_total_energy_consumed(self) -> float:
        """Return total energy consumed since last reset, in Joules."""
        return self._total_energy_j

    def get_mass_and_dimensions(self) -> tuple:
        """
        Physical battery specs for CAD / URDF builders.

        Returns (mass_grams: int, dimensions_mm: tuple).
        """
        return (self.MASS_G, self.DIMENSIONS_MM)

    def get_history(self):
        """
        Return per-step simulation history.

        Returns a pandas DataFrame if pandas is available, otherwise a plain
        dict of lists.  Keys / columns:
            'time'          — simulation time at step start [s]
            'mode'          — flight mode string (or "" for legacy scalar calls)
            'voltage'       — ECM terminal voltage [V]
            'soc'           — State of Charge [0, 1]
            'capacity_mah'  — remaining capacity [mAh]
            'power'         — instantaneous power [W]
            'current'       — current drawn [A]
            'total_energy_j'— cumulative energy consumed [J]
        """
        data = {
            'time':           self._hist_time,
            'mode':           self._hist_mode,
            'voltage':        self._hist_voltage,
            'soc':            self._hist_soc,
            'capacity_mah':   self._hist_capacity_mah,
            'power':          self._hist_power,
            'current':        self._hist_current,
            'total_energy_j': self._hist_energy,
        }
        if _PANDAS_AVAILABLE:
            return pd.DataFrame(data)
        return data

    # ------------------------------------------------------------------ #
    # Dunder                                                              #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"LiPoBatteryModel({self.BATTERY_NAME}) | "
            f"mode={self._last_mode!r} | "
            f"SoC={self._soc*100:.1f}% | "
            f"V={self._last_voltage:.3f}V | "
            f"E={self._total_energy_j:.1f}J | "
            f"depleted={self.is_depleted}"
        )
