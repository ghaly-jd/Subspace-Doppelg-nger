from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared.config import get_nested


RITUAL_MOTION_TYPE = "Motion Ritual"


@dataclass(frozen=True)
class RitualPhase:
    id: str
    label: str
    prompt: str
    duration_seconds: float
    profile: str = "full_body"


DEFAULT_RITUAL_PHASES = (
    RitualPhase(
        id="calibration_pose",
        label="Calibration Pose",
        prompt="Stand still.",
        duration_seconds=3.0,
    ),
    RitualPhase(
        id="lateral_step",
        label="Lateral Step Scan",
        prompt="Step left. Step right. Return center.",
        duration_seconds=4.0,
    ),
    RitualPhase(
        id="arm_raise",
        label="Arm Raise Scan",
        prompt="Raise both arms. Lower them.",
        duration_seconds=3.0,
        profile="upper_body",
    ),
    RitualPhase(
        id="squat",
        label="Squat Scan",
        prompt="Squat down. Stand up.",
        duration_seconds=4.0,
    ),
    RitualPhase(
        id="rotation",
        label="Rotation Scan",
        prompt="Turn left. Turn right.",
        duration_seconds=4.0,
    ),
    RitualPhase(
        id="balance_hold",
        label="Balance Hold",
        prompt="Lift one knee. Hold. Return.",
        duration_seconds=4.0,
    ),
    RitualPhase(
        id="wave_hello",
        label="Wave Hello",
        prompt="Wave hello.",
        duration_seconds=3.0,
        profile="upper_body",
    ),
    RitualPhase(
        id="clap_twice",
        label="Clap Twice",
        prompt="Clap twice.",
        duration_seconds=2.0,
        profile="upper_body",
    ),
    RitualPhase(
        id="point_forward",
        label="Point Forward",
        prompt="Point forward.",
        duration_seconds=2.0,
        profile="upper_body",
    ),
)


def is_ritual_motion(motion_type: str) -> bool:
    normalized = "".join(character for character in motion_type.lower() if character.isalnum())
    return normalized in {"motionritual", "ritual", "grassmannmirror"}


def ritual_motion_label(config: dict[str, Any] | None = None) -> str:
    if config is None:
        return RITUAL_MOTION_TYPE
    return str(get_nested(config, "ritual", "name", default=RITUAL_MOTION_TYPE))


def ritual_phases_from_config(config: dict[str, Any]) -> tuple[RitualPhase, ...]:
    configured = get_nested(config, "ritual", "phases", default=None)
    if not isinstance(configured, list) or not configured:
        return DEFAULT_RITUAL_PHASES

    phases: list[RitualPhase] = []
    defaults_by_id = {phase.id: phase for phase in DEFAULT_RITUAL_PHASES}
    for item in configured:
        if not isinstance(item, dict):
            continue
        phase_id = str(item.get("id", "")).strip()
        if not phase_id:
            continue
        default = defaults_by_id.get(phase_id)
        phases.append(
            RitualPhase(
                id=phase_id,
                label=str(item.get("label", default.label if default else phase_id.replace("_", " ").title())),
                prompt=str(item.get("prompt", default.prompt if default else "")),
                duration_seconds=float(
                    item.get(
                        "duration_seconds",
                        default.duration_seconds if default else 3.0,
                    )
                ),
                profile=str(item.get("profile", default.profile if default else "full_body")),
            )
        )
    return tuple(phases) if phases else DEFAULT_RITUAL_PHASES


def ritual_total_seconds(config: dict[str, Any]) -> float:
    configured = get_nested(config, "ritual", "total_seconds", default=None)
    if configured is not None:
        return float(configured)
    return float(sum(phase.duration_seconds for phase in ritual_phases_from_config(config)))


def ritual_scoring_weights(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        "overall": {
            "full_coordinate_subspace": float(
                get_nested(
                    config,
                    "ritual_scoring",
                    "overall",
                    "full_coordinate_subspace",
                    default=0.35,
                )
            ),
            "average_phase": float(
                get_nested(config, "ritual_scoring", "overall", "average_phase", default=0.20)
            ),
            "full_velocity_subspace": float(
                get_nested(
                    config,
                    "ritual_scoring",
                    "overall",
                    "full_velocity_subspace",
                    default=0.15,
                )
            ),
            "rhythm": float(get_nested(config, "ritual_scoring", "overall", "rhythm", default=0.10)),
            "joint_angles": float(
                get_nested(config, "ritual_scoring", "overall", "joint_angles", default=0.10)
            ),
            "energy_balance": float(
                get_nested(config, "ritual_scoring", "overall", "energy_balance", default=0.10)
            ),
        },
        "phase": {
            "coordinate_subspace": float(
                get_nested(config, "ritual_scoring", "phase", "coordinate_subspace", default=0.35)
            ),
            "velocity_subspace": float(
                get_nested(config, "ritual_scoring", "phase", "velocity_subspace", default=0.20)
            ),
            "speed_dtw": float(
                get_nested(config, "ritual_scoring", "phase", "speed_dtw", default=0.20)
            ),
            "rhythm": float(get_nested(config, "ritual_scoring", "phase", "rhythm", default=0.10)),
            "joint_angles": float(
                get_nested(config, "ritual_scoring", "phase", "joint_angles", default=0.15)
            ),
        },
    }
