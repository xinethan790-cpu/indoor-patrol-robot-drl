"""Stateful, action-free monitoring of omnidirectional LiDAR proximity.

The monitor consumes scans in the project's policy order: beam zero points
forward and positive beam indices turn toward the robot's left.  It classifies
critical proximity and estimates closing motion, but deliberately neither
claims physical contact nor modifies a policy action.  It is separate from the
evidence-bound R2a safety shield.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CriticalProximityConfig:
    """Validated thresholds for :class:`CriticalProximityMonitor`."""

    critical_threshold_m: float
    minimum_beam_count: int = 8
    approaching_epsilon_mps: float = 1e-6
    threat_association_max_angle_rad: float = math.radians(8.0)
    maximum_plausible_closing_rate_mps: float = 0.50
    approaching_confirmation_s: float = 0.20

    def __post_init__(self) -> None:
        threshold = _finite_float(
            self.critical_threshold_m,
            label="critical_threshold_m",
        )
        if threshold <= 0.0:
            raise ValueError("critical_threshold_m must be positive.")
        object.__setattr__(self, "critical_threshold_m", threshold)
        if isinstance(self.minimum_beam_count, bool) or not isinstance(
            self.minimum_beam_count, int
        ):
            raise TypeError("minimum_beam_count must be an integer.")
        if self.minimum_beam_count < 5:
            raise ValueError(
                "minimum_beam_count must be at least five for the local median."
            )
        epsilon = _finite_float(
            self.approaching_epsilon_mps,
            label="approaching_epsilon_mps",
        )
        if epsilon < 0.0:
            raise ValueError("approaching_epsilon_mps cannot be negative.")
        object.__setattr__(self, "approaching_epsilon_mps", epsilon)
        association_angle = _finite_float(
            self.threat_association_max_angle_rad,
            label="threat_association_max_angle_rad",
        )
        if not 0.0 < association_angle <= math.pi:
            raise ValueError(
                "threat_association_max_angle_rad must be in the interval (0, pi]."
            )
        object.__setattr__(
            self,
            "threat_association_max_angle_rad",
            association_angle,
        )
        plausible_rate = _finite_float(
            self.maximum_plausible_closing_rate_mps,
            label="maximum_plausible_closing_rate_mps",
        )
        if plausible_rate <= 0.0:
            raise ValueError("maximum_plausible_closing_rate_mps must be positive.")
        object.__setattr__(
            self,
            "maximum_plausible_closing_rate_mps",
            plausible_rate,
        )
        confirmation = _finite_float(
            self.approaching_confirmation_s,
            label="approaching_confirmation_s",
        )
        if confirmation <= 0.0:
            raise ValueError("approaching_confirmation_s must be positive.")
        object.__setattr__(self, "approaching_confirmation_s", confirmation)

    @classmethod
    def from_lidar_min_range(
        cls,
        lidar_min_range_m: float,
        *,
        margin_m: float = 0.01,
        minimum_beam_count: int = 8,
        approaching_epsilon_mps: float = 1e-6,
        threat_association_max_angle_rad: float = math.radians(8.0),
        maximum_plausible_closing_rate_mps: float = 0.50,
        approaching_confirmation_s: float = 0.20,
    ) -> CriticalProximityConfig:
        """Build the conventional ``LiDAR minRange + 0.01 m`` threshold.

        The caller supplies the live device limit, keeping this module free of
        Webots or other hardware APIs.  ``margin_m`` remains explicit and may
        be overridden for a separately identified application profile.
        """

        lidar_minimum = _finite_float(
            lidar_min_range_m,
            label="lidar_min_range_m",
        )
        margin = _finite_float(margin_m, label="margin_m")
        if lidar_minimum <= 0.0:
            raise ValueError("lidar_min_range_m must be positive.")
        if margin < 0.0:
            raise ValueError("margin_m cannot be negative.")
        threshold = lidar_minimum + margin
        if not math.isfinite(threshold):
            raise ValueError("lidar_min_range_m + margin_m must be finite.")
        return cls(
            critical_threshold_m=threshold,
            minimum_beam_count=minimum_beam_count,
            approaching_epsilon_mps=approaching_epsilon_mps,
            threat_association_max_angle_rad=threat_association_max_angle_rad,
            maximum_plausible_closing_rate_mps=(
                maximum_plausible_closing_rate_mps
            ),
            approaching_confirmation_s=approaching_confirmation_s,
        )


@dataclass(frozen=True, slots=True)
class CriticalProximityAssessment:
    """One all-angle proximity classification and temporal estimate."""

    minimum_distance_m: float
    minimum_beam_index: int
    minimum_angle_rad: float
    beam_count: int
    critical_threshold_m: float
    critical: bool
    consecutive_critical_frames: int
    closing_rate_mps: float | None
    ttc_s: float | None
    tracked_distance_m: float | None = None
    threat_association_valid: bool | None = None
    closing_rate_valid: bool | None = None
    approaching_confirmation_elapsed_s: float | None = None
    approaching_confirmation_required_s: float | None = None
    approaching_confirmed: bool | None = None
    ttc_valid: bool | None = None

    def __post_init__(self) -> None:
        minimum = _finite_float(self.minimum_distance_m, label="minimum_distance_m")
        threshold = _finite_float(
            self.critical_threshold_m,
            label="critical_threshold_m",
        )
        if minimum <= 0.0 or threshold <= 0.0:
            raise ValueError("Proximity distances and thresholds must be positive.")
        beam_count = _validated_integer(self.beam_count, label="beam_count", minimum=2)
        beam_index = _validated_integer(
            self.minimum_beam_index,
            label="minimum_beam_index",
            minimum=0,
        )
        if beam_index >= beam_count:
            raise ValueError("minimum_beam_index must be below beam_count.")
        angle = _finite_float(self.minimum_angle_rad, label="minimum_angle_rad")
        expected_angle = _policy_beam_angle(beam_index, beam_count)
        if not math.isclose(angle, expected_angle, abs_tol=1e-12):
            raise ValueError(
                "minimum_angle_rad does not match minimum_beam_index and beam_count."
            )
        if not isinstance(self.critical, bool):
            raise TypeError("critical must be a bool.")
        expected_critical = minimum <= threshold
        if self.critical is not expected_critical:
            raise ValueError("critical must equal minimum_distance_m <= critical_threshold_m.")
        consecutive = _validated_integer(
            self.consecutive_critical_frames,
            label="consecutive_critical_frames",
            minimum=0,
        )
        if self.critical != (consecutive > 0):
            raise ValueError(
                "consecutive_critical_frames must be positive exactly when critical."
            )

        closing_rate = self.closing_rate_mps
        if closing_rate is not None:
            closing_rate = _finite_float(closing_rate, label="closing_rate_mps")
        ttc = self.ttc_s
        if ttc is not None:
            ttc = _finite_float(ttc, label="ttc_s")
            if ttc < 0.0:
                raise ValueError("ttc_s cannot be negative.")
        audit_values = (
            self.tracked_distance_m,
            self.threat_association_valid,
            self.closing_rate_valid,
            self.approaching_confirmation_elapsed_s,
            self.approaching_confirmation_required_s,
            self.approaching_confirmed,
            self.ttc_valid,
        )
        audit_supplied = any(value is not None for value in audit_values)
        if audit_supplied and any(value is None for value in audit_values):
            raise ValueError(
                "Proximity validity and confirmation audit fields must be supplied "
                "together."
            )

        tracked_distance: float | None = None
        confirmation_elapsed: float | None = None
        confirmation_required: float | None = None
        if audit_supplied:
            tracked_distance = _finite_float(
                self.tracked_distance_m,
                label="tracked_distance_m",
            )
            if tracked_distance <= 0.0:
                raise ValueError("tracked_distance_m must be positive.")
            if tracked_distance < minimum:
                raise ValueError(
                    "tracked_distance_m cannot be below the raw minimum distance."
                )
            for label, value in (
                ("threat_association_valid", self.threat_association_valid),
                ("closing_rate_valid", self.closing_rate_valid),
                ("approaching_confirmed", self.approaching_confirmed),
                ("ttc_valid", self.ttc_valid),
            ):
                if not isinstance(value, bool):
                    raise TypeError(f"{label} must be a bool when audit fields are used.")
            confirmation_elapsed = _finite_float(
                self.approaching_confirmation_elapsed_s,
                label="approaching_confirmation_elapsed_s",
            )
            confirmation_required = _finite_float(
                self.approaching_confirmation_required_s,
                label="approaching_confirmation_required_s",
            )
            if confirmation_elapsed < 0.0:
                raise ValueError(
                    "approaching_confirmation_elapsed_s cannot be negative."
                )
            if confirmation_required <= 0.0:
                raise ValueError(
                    "approaching_confirmation_required_s must be positive."
                )
            if not self.threat_association_valid:
                if self.closing_rate_valid:
                    raise ValueError(
                        "closing_rate_valid requires a valid threat association."
                    )
                if confirmation_elapsed != 0.0:
                    raise ValueError(
                        "An unassociated threat cannot retain confirmation time."
                    )
            if not self.closing_rate_valid and confirmation_elapsed != 0.0:
                raise ValueError(
                    "An invalid closing-rate sample cannot retain confirmation time."
                )
            if self.approaching_confirmed:
                if not self.closing_rate_valid:
                    raise ValueError(
                        "approaching_confirmed requires a valid closing-rate sample."
                    )
                if confirmation_elapsed < confirmation_required:
                    raise ValueError(
                        "approaching_confirmed requires the configured confirmation time."
                    )
                if closing_rate is None or closing_rate <= 0.0:
                    raise ValueError(
                        "approaching_confirmed requires a positive closing_rate_mps."
                    )
                if ttc is None or not self.ttc_valid:
                    raise ValueError("approaching_confirmed requires a valid ttc_s.")
            else:
                if closing_rate is not None or ttc is not None or self.ttc_valid:
                    raise ValueError(
                        "Unconfirmed approach must suppress closing_rate_mps and ttc_s."
                    )
            if self.ttc_valid is not (ttc is not None):
                raise ValueError("ttc_valid must exactly describe whether ttc_s is emitted.")
            if self.critical and ttc is not None and ttc != 0.0:
                raise ValueError("Critical proximity can only report ttc_s=0.0.")
            if not self.critical and ttc == 0.0:
                raise ValueError("Non-critical proximity cannot report ttc_s=0.0.")
        elif self.critical:
            if ttc != 0.0:
                raise ValueError("Critical proximity must report ttc_s=0.0.")
        elif closing_rate is not None and closing_rate > 0.0:
            if ttc is None:
                raise ValueError("Positive closing_rate_mps requires ttc_s.")
        elif ttc is not None:
            raise ValueError("Non-critical non-approaching proximity cannot report ttc_s.")

        object.__setattr__(self, "minimum_distance_m", minimum)
        object.__setattr__(self, "minimum_beam_index", beam_index)
        object.__setattr__(self, "minimum_angle_rad", angle)
        object.__setattr__(self, "beam_count", beam_count)
        object.__setattr__(self, "critical_threshold_m", threshold)
        object.__setattr__(self, "consecutive_critical_frames", consecutive)
        object.__setattr__(self, "closing_rate_mps", closing_rate)
        object.__setattr__(self, "ttc_s", ttc)
        object.__setattr__(self, "tracked_distance_m", tracked_distance)
        object.__setattr__(
            self,
            "approaching_confirmation_elapsed_s",
            confirmation_elapsed,
        )
        object.__setattr__(
            self,
            "approaching_confirmation_required_s",
            confirmation_required,
        )

    @property
    def approaching(self) -> bool:
        """Whether a confirmed local threat track is closing."""

        return self.closing_rate_mps is not None and self.closing_rate_mps > 0.0


class CriticalProximityMonitor:
    """Track full-scan critical proximity without changing robot commands."""

    def __init__(self, config: CriticalProximityConfig) -> None:
        if not isinstance(config, CriticalProximityConfig):
            raise TypeError("config must be a CriticalProximityConfig value.")
        self.config = config
        self._previous_tracked_distance_m: float | None = None
        self._previous_minimum_angle_rad: float | None = None
        self._beam_count: int | None = None
        self._consecutive_critical_frames = 0
        self._approaching_confirmation_elapsed_s = 0.0

    def reset(self) -> None:
        """Discard temporal state before a new run or sensor reconfiguration."""

        self._previous_tracked_distance_m = None
        self._previous_minimum_angle_rad = None
        self._beam_count = None
        self._consecutive_critical_frames = 0
        self._approaching_confirmation_elapsed_s = 0.0

    def update(
        self,
        policy_order_lidar: np.ndarray | list[float],
        *,
        dt_s: float,
    ) -> CriticalProximityAssessment:
        """Classify one scan and update consecutive-frame and closing estimates.

        The hard ``critical`` result always uses the raw global minimum.  The
        temporal estimate instead follows the current nearest beam with a
        circular five-beam median, rejects cross-angle and implausible jumps,
        and emits ``closing_rate_mps``/``ttc_s`` only after a continuously
        approaching track reaches its confirmation duration.
        """

        scan = _validated_scan(
            policy_order_lidar,
            minimum_beam_count=self.config.minimum_beam_count,
        )
        dt = _finite_float(dt_s, label="dt_s")
        if dt <= 0.0:
            raise ValueError("dt_s must be positive.")
        if self._beam_count is not None and scan.size != self._beam_count:
            raise ValueError(
                "LiDAR beam count changed without resetting the proximity monitor: "
                f"{self._beam_count} -> {scan.size}."
            )

        beam_index = int(np.argmin(scan))
        minimum_distance = float(scan[beam_index])
        minimum_angle = _policy_beam_angle(beam_index, int(scan.size))
        tracked_distance = _circular_local_median(scan, beam_index)
        critical = minimum_distance <= self.config.critical_threshold_m
        consecutive = self._consecutive_critical_frames + 1 if critical else 0

        association_valid = False
        closing_rate_valid = False
        confirmation_elapsed = 0.0
        closing_rate: float | None = None
        if (
            self._previous_tracked_distance_m is not None
            and self._previous_minimum_angle_rad is not None
        ):
            association_valid = (
                _circular_angle_distance(
                    minimum_angle,
                    self._previous_minimum_angle_rad,
                )
                <= self.config.threat_association_max_angle_rad
            )
        if association_valid:
            raw_closing_rate = (
                self._previous_tracked_distance_m - tracked_distance
            ) / dt
            if not math.isfinite(raw_closing_rate):
                raise ValueError("LiDAR distance change and dt_s produced a non-finite rate.")
            candidate_closing_rate = (
                0.0
                if abs(raw_closing_rate) <= self.config.approaching_epsilon_mps
                else raw_closing_rate
            )
            maximum_rate = self.config.maximum_plausible_closing_rate_mps
            if math.isclose(
                abs(candidate_closing_rate),
                maximum_rate,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                candidate_closing_rate = math.copysign(
                    maximum_rate,
                    candidate_closing_rate,
                )
            closing_rate_valid = (
                abs(candidate_closing_rate) <= maximum_rate
            )
            if closing_rate_valid and candidate_closing_rate > 0.0:
                confirmation_elapsed = (
                    self._approaching_confirmation_elapsed_s + dt
                )
                if (
                    confirmation_elapsed
                    >= self.config.approaching_confirmation_s
                ):
                    closing_rate = candidate_closing_rate
            else:
                confirmation_elapsed = 0.0
        approaching_confirmed = closing_rate is not None
        if approaching_confirmed:
            if critical:
                ttc: float | None = 0.0
            else:
                ttc = (
                    tracked_distance - self.config.critical_threshold_m
                ) / closing_rate
            if not math.isfinite(ttc):
                raise ValueError("LiDAR distance and closing rate produced a non-finite TTC.")
        else:
            ttc = None

        assessment = CriticalProximityAssessment(
            minimum_distance_m=minimum_distance,
            minimum_beam_index=beam_index,
            minimum_angle_rad=minimum_angle,
            beam_count=int(scan.size),
            critical_threshold_m=self.config.critical_threshold_m,
            critical=critical,
            consecutive_critical_frames=consecutive,
            closing_rate_mps=closing_rate,
            ttc_s=ttc,
            tracked_distance_m=tracked_distance,
            threat_association_valid=association_valid,
            closing_rate_valid=closing_rate_valid,
            approaching_confirmation_elapsed_s=confirmation_elapsed,
            approaching_confirmation_required_s=(
                self.config.approaching_confirmation_s
            ),
            approaching_confirmed=approaching_confirmed,
            ttc_valid=ttc is not None,
        )
        self._previous_tracked_distance_m = tracked_distance
        self._previous_minimum_angle_rad = minimum_angle
        self._beam_count = int(scan.size)
        self._consecutive_critical_frames = consecutive
        self._approaching_confirmation_elapsed_s = confirmation_elapsed
        return assessment


def _validated_scan(
    policy_order_lidar: np.ndarray | list[float],
    *,
    minimum_beam_count: int,
) -> np.ndarray:
    try:
        scan = np.asarray(policy_order_lidar, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("policy_order_lidar must contain finite distances.") from error
    if scan.ndim != 1:
        raise ValueError("policy_order_lidar must be a one-dimensional scan.")
    if scan.size < minimum_beam_count:
        raise ValueError(
            "policy_order_lidar has too few beams: "
            f"expected at least {minimum_beam_count}, got {scan.size}."
        )
    if not np.isfinite(scan).all():
        raise ValueError("policy_order_lidar distances must be finite.")
    if np.any(scan <= 0.0):
        raise ValueError("policy_order_lidar distances must be positive.")
    return scan


def _policy_beam_angle(beam_index: int, beam_count: int) -> float:
    angle = beam_index * (2.0 * math.pi / beam_count)
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _circular_local_median(scan: np.ndarray, beam_index: int) -> float:
    indices = [
        (beam_index + offset) % int(scan.size)
        for offset in (-2, -1, 0, 1, 2)
    ]
    return float(np.median(scan[indices]))


def _circular_angle_distance(first_rad: float, second_rad: float) -> float:
    difference = (first_rad - second_rad + math.pi) % (2.0 * math.pi) - math.pi
    return abs(difference)


def _finite_float(value: float, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a finite number, not a boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def _validated_integer(value: int, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer.")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return value
