from dataclasses import dataclass, field
from typing import Any


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class CameraCommand:
    action: str
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    amount: float = 0.0
    camera: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CameraCommand":
        return cls(
            action=str(payload.get("action", "")),
            dx=float(payload.get("dx", 0.0)),
            dy=float(payload.get("dy", 0.0)),
            dz=float(payload.get("dz", 0.0)),
            amount=float(payload.get("amount", 0.0)),
            camera=payload.get("camera"),
        )


@dataclass
class CameraState:
    mode: str = "free"
    fixed_camera: str | None = None
    azimuth: float = 150.0
    elevation: float = -25.0
    distance: float = 4.0
    lookat: list[float] = field(default_factory=lambda: [0.0, 0.3, 0.6])
    min_distance: float = 0.2
    max_distance: float = 12.0
    min_elevation: float = -89.0
    max_elevation: float = 89.0

    def apply(self, command: CameraCommand) -> None:
        if command.action == "orbit":
            self.mode = "free"
            self.azimuth += command.dx
            self.elevation = _clamp(
                self.elevation + command.dy,
                self.min_elevation,
                self.max_elevation,
            )
            return

        if command.action == "pan":
            self.mode = "free"
            self.lookat[0] += command.dx
            self.lookat[1] += command.dy
            self.lookat[2] += command.dz
            return

        if command.action == "zoom":
            self.mode = "free"
            self.distance = _clamp(
                self.distance + command.amount,
                self.min_distance,
                self.max_distance,
            )
            return

        if command.action == "set_fixed":
            self.mode = "fixed"
            self.fixed_camera = command.camera
            return

        if command.action == "reset":
            self.mode = "free"
            self.fixed_camera = None
            self.azimuth = 150.0
            self.elevation = -25.0
            self.distance = 4.0
            self.lookat = [0.0, 0.3, 0.6]

    def to_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "fixed_camera": self.fixed_camera,
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "distance": self.distance,
            "lookat": list(self.lookat),
        }

