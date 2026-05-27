"""
LiPo Battery Model for Airevolve Drone Physics Simulation.

Models the Tattu R-Line 750mAh 14.8V 95C 4S / Tattu 450mAh 14.8V 75C 4S lithium polymer battery using a
high-fidelity motor-level Equivalent Circuit Model (ECM) with SoC-based voltage curve and RPM-based
current estimation for a 295g hexacopter (6 motors).

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
    Motor-level Equivalent Circuit Model (ECM) for a 295 g hexacopter (6 motors).

    step() signature:
        step(dt, motor_rpms, max_rpm)
            RPM-to-current quadratic curve per motor, summed with FC baseline,
            then ECM voltage sag applied.

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
    BATTERY_NAME = "Tattu 750mAh 14.8V 95C 4S"
    NUM_CELLS = 4

    # Electrical
    CAPACITY_MAH = 750.0          # mAh  750 or 450
    CAPACITY_AH  = 0.750          # Ah  (= CAPACITY_MAH / 1000)
    VOLTAGE_FULL     = 16.8       # V   (4 × 4.20 V/cell, fully charged)
    VOLTAGE_NOMINAL  = 14.8       # V   (4 × 3.70 V/cell, nominal)
    VOLTAGE_DEPLETED = 12.8       # V   (4 × 3.20 V/cell, lower cutoff)
    C_RATING     = 95             # 95 or 75
    MAX_CURRENT  = 71.25          # A   71.25 or 33.75

    # ECM internal resistance — 60 mΩ is a validated value for a high-discharge
    # 4S racing pack.  At max current (71.25 A) this causes ~4.3 V sag,
    # accurately modelling brownout risk at peak throttle.
    INTERNAL_RESISTANCE = 0.06    # Ω

    # ------------------------------------------------------------------ #
    # Motor-level ECM constants — 295 g hexacopter (6 motors)            #
    # ------------------------------------------------------------------ #
    MOTOR_IDLE_CURRENT  = 0.5     # A — per-motor current at zero throttle
    MOTOR_MAX_CURRENT   = 14.0    # A — per-motor current at max RPM
    FC_BASELINE_CURRENT = 0.15    # A — flight controller + ESC idle draw
    BATTERY_MAX_CURRENT = 71.25   # A — battery continuous discharge limit

    # Physical
    # NOTE: MASS_KG must equal BATTERY_MASS in propeller_data.py (currently 0.082 kg).
    # NOTE: DIMENSIONS_MM must match any URDF / CAD generator geometry.
    MASS_G         = 82           # grams  82 or 55
    MASS_KG        = 0.082        # kg   — matches BATTERY_MASS in propeller_data.py
    DIMENSIONS_MM  = (60, 31, 28) # mm     (60 x 31 x 28) or (45 x 24 x 27)

    # Depletion thresholds
    SOC_DEPLETION_THRESHOLD     = 0.15  # 15 % SoC commercial safety reserve
    VOLTAGE_DEPLETION_THRESHOLD = 12.8  # V  — same as VOLTAGE_DEPLETED (hard cutoff)

    # Legacy Open-Circuit Voltage lookup table — kept for reference; ECM formula is the live path.
    _OCV_SOC = np.array([0.00, 0.05, 0.20, 0.80, 1.00])
    _OCV_V   = np.array([12.80, 13.40, 14.40, 16.00, 16.80])

    # ------------------------------------------------------------------ #

    def __init__(self, strict_voltage_kill: bool = True, track_history: bool = True) -> None:
        """
        Parameters
        ----------
        strict_voltage_kill : bool
            If True (default / NSGA-II eval): depletion fires on
            SoC < 0.15 OR V_terminal < 12.8 V — physically accurate.
            If False (PPO training): depletion fires ONLY on SoC < 0.15,
            preventing transient voltage sags from instantly killing
            episodes before the agent has learned throttle control.
        """
        self.strict_voltage_kill = strict_voltage_kill
        self.track_history = track_history
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
        self._last_current: float = 0.0

        # Per-step history lists — converted to arrays / DataFrame at get_history()
        self._hist_time:         list[float] = []
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

          V_ideal    = 13.0 + 3.8 × √SoC
          V_terminal = V_ideal − I_total × R_internal

        The √SoC shape follows the empirical discharge profile of a 4S LiPo:
        fast drop near full charge, long flat mid-range plateau, steep knee
        at low SoC.  R_internal = 0.06 Ω models the dominant DC series resistance
        of the pack at room temperature under high-rate discharge.

        NOTE: V_terminal is NOT clamped here so that the depletion check can
        accurately detect sub-threshold voltages (strict_voltage_kill path).
        """
        soc_clamped = max(0.0, min(1.0, soc))
        v_ideal    = 13.0 + 3.8 * math.sqrt(soc_clamped)
        v_terminal = v_ideal - (current * self.INTERNAL_RESISTANCE)
        return v_terminal

    def _compute_ocv(self, soc: float) -> float:
        """Piecewise-linear OCV from the lookup table (legacy reference path)."""
        return float(np.interp(np.clip(soc, 0.0, 1.0), self._OCV_SOC, self._OCV_V))

    def compute_current_from_rpms(
        self, motor_rpms: np.ndarray, max_rpm: float
    ) -> float:
        """
        Compute total battery current from motor RPMs using the quadratic ECM.

        Per-motor model:
            I_m = MOTOR_IDLE_CURRENT + (MOTOR_MAX_CURRENT - MOTOR_IDLE_CURRENT)
                  × (RPM_i / max_rpm)²

        Total current:
            I_total = FC_BASELINE_CURRENT + Σ I_m   (over all 6 motors)

        Parameters
        ----------
        motor_rpms : np.ndarray
            Array of per-motor RPM values (length = number of motors, typically 6).
        max_rpm : float
            Maximum achievable RPM for the motor set.

        Returns
        -------
        float
            Total battery current draw in Amperes.
        """
        rpm_ratio = np.clip(np.asarray(motor_rpms, dtype=float) / max_rpm, 0.0, 1.0)
        motor_currents = (
            self.MOTOR_IDLE_CURRENT
            + (self.MOTOR_MAX_CURRENT - self.MOTOR_IDLE_CURRENT) * rpm_ratio ** 2
        )
        return float(self.FC_BASELINE_CURRENT + np.sum(motor_currents))

    def step(self, dt: float, motor_rpms: np.ndarray, max_rpm: float) -> dict:
        """
        Advance the battery model by one timestep using motor RPMs.

        Parameters
        ----------
        dt : float
            Timestep in seconds.  Matches DroneSimulator / DroneGateEnv dt
            (typically 0.005–0.01 s).
        motor_rpms : np.ndarray
            Per-motor RPM values, shape (n_motors,).  Typically 6 for the
            295 g hexacopter.
        max_rpm : float
            Maximum achievable RPM used to normalise the quadratic curve.

        Returns
        -------
        dict with keys:
            time          — simulation time at the START of this step [s]
            soc           — State of Charge in [0, 1]
            voltage       — ECM terminal voltage [V]
            current       — total battery current [A]
            capacity_mah  — remaining capacity [mAh]
            power         — instantaneous power [W]
            total_energy_j — cumulative energy consumed [J]
        """
        t_now = self._total_time_s

        # ── RPM → Current ─────────────────────────────────────────────── #
        # Per-motor quadratic: I_m = idle + (max - idle) × (RPM/max_rpm)²
        # Total: FC baseline + Σ motor currents
        current = self.compute_current_from_rpms(motor_rpms, max_rpm)

        # BMS cutoff — no further drain once the safety threshold is breached
        if self.is_depleted:
            current = 0.0

        # Capacity drain: Δ_Ah = I [A] × dt [s] / 3600 [s/h]
        delta_ah = current * (dt / 3600.0)
        self._capacity_remaining_ah = max(0.0, self._capacity_remaining_ah - delta_ah)

        # SoC update
        self._soc = float(np.clip(
            self._capacity_remaining_ah / self.CAPACITY_AH, 0.0, 1.0
        ))

        # ECM terminal voltage: V_ideal = 13.0 + 3.8 × √SoC,  V_term = V_ideal - I×R
        v_terminal = self._compute_ecm_voltage(self._soc, current)

        # Instantaneous power [W] = V_terminal × I_total
        power_w = v_terminal * current
        self._total_energy_j += power_w * dt

        # Advance simulation clock
        self._total_time_s += dt

        # ── Depletion check ───────────────────────────────────────────── #
        # Evaluated AFTER update so the breaching step is still recorded.
        # strict_voltage_kill=True  → SoC OR voltage cutoff  (NSGA-II eval)
        # strict_voltage_kill=False → SoC cutoff ONLY         (PPO training)
        soc_depleted = self._soc < self.SOC_DEPLETION_THRESHOLD
        if self.strict_voltage_kill:
            v_depleted = v_terminal < self.VOLTAGE_DEPLETION_THRESHOLD
        else:
            v_depleted = False
        if soc_depleted or v_depleted:
            self.is_depleted = True

        # Persist for property accessors and RL observation cache
        self._last_voltage = v_terminal
        self._last_power   = power_w
        self._last_current = current

        # Append to per-step history
        if self.track_history:
            self._hist_time.append(t_now)
            self._hist_voltage.append(v_terminal)
            self._hist_soc.append(self._soc)
            self._hist_capacity_mah.append(self._capacity_remaining_ah * 1000.0)
            self._hist_power.append(power_w)
            self._hist_current.append(current)
            self._hist_energy.append(self._total_energy_j)

        return {
            'time':           t_now,
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
    def current(self) -> float:
        """Total battery current from the last step, in Amperes."""
        return self._last_current

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
            'voltage'       — ECM terminal voltage [V]
            'soc'           — State of Charge [0, 1]
            'capacity_mah'  — remaining capacity [mAh]
            'power'         — instantaneous power [W]
            'current'       — total battery current [A]
            'total_energy_j'— cumulative energy consumed [J]
        """
        data = {
            'time':           self._hist_time,
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
            f"strict_kill={self.strict_voltage_kill} | "
            f"SoC={self._soc*100:.1f}% | "
            f"V={self._last_voltage:.3f}V | "
            f"I={self._last_current:.2f}A | "
            f"E={self._total_energy_j:.1f}J | "
            f"depleted={self.is_depleted}"
        )
