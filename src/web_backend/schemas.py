from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str = "sta-web-backend"


class ControlRequest(BaseModel):
    action: Literal["pause", "resume", "reset"]


class CameraRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: str
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    amount: float = 0.0
    camera: str | None = None


class SimulationStatus(BaseModel):
    state: str
    sim_time: float = 0.0
    frame_index: int = 0
    resolution: list[int] = Field(default_factory=lambda: [0, 0])
    fps: int = 0
    camera: dict[str, Any] = Field(default_factory=dict)
    available_cameras: list[str] = Field(default_factory=list)
    active_task: dict[str, Any] | None = None


class TaskDispatchRequest(BaseModel):
    instruction: str = ""
    scene: str = "scene5_glare.xml"
    mode: str = "实时仿真"
    total_time: float = 300.0
    fps: int = 20
    record_tcp: bool = True
    plan: dict[str, Any] | None = None


class TaskRecord(BaseModel):
    id: str
    instruction: str
    scene: str
    mode: str
    status: str
    created_at: str
    message: str = ""


class LogRecord(BaseModel):
    id: str
    timestamp: str
    level: str
    message: str
