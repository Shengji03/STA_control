from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str = "sta-web-backend"


class ControlRequest(BaseModel):
    action: Literal["pause", "resume", "reset", "stop"]


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
    last_task: dict[str, Any] | None = None
    scene: str = ''
    telemetry: dict[str, Any] | None = None
    overlay: dict[str, bool] = Field(default_factory=dict)
    replay: bool = False
    message: str = ''


class TaskDispatchRequest(BaseModel):
    instruction: str = ""
    scene: str = "scene5_glare.xml"
    mode: Literal['实时仿真', '仅规划'] = "实时仿真"
    total_time: float = Field(default=300.0, ge=2, le=1800)
    fps: int = Field(default=20, ge=5, le=30)
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
    optimization: dict[str, Any] | None = None
    execution_effects: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    completed_at: str | None = None
    sample_count: int = 0


class ReplayRequest(BaseModel):
    index: int = Field(default=0, ge=0)


class OverlayRequest(BaseModel):
    planned: bool = True
    actual: bool = True
    L: bool = True
    R: bool = True


class LogRecord(BaseModel):
    id: str
    timestamp: str
    level: str
    message: str
