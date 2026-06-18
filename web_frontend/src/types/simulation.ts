export type SimulationState = "running" | "paused" | "disabled" | string;

export type ControlAction = "pause" | "resume" | "reset";

export interface CameraState {
  mode?: string;
  fixed_camera?: string | null;
  azimuth?: number;
  elevation?: number;
  distance?: number;
  lookat?: number[];
}

export interface SimulationStatus {
  state: SimulationState;
  sim_time: number;
  frame_index: number;
  resolution: [number, number] | number[];
  fps: number;
  camera: CameraState;
  available_cameras: string[];
  active_task?: {
    id: string;
    instruction: string;
    status?: string;
    message?: string;
    progress: number;
    runner_state?: string;
    phase_index?: number;
    phase_count?: number;
  } | null;
}

export interface ControlMessage {
  type: "control";
  action: ControlAction;
}

export interface CameraCommand {
  type?: "camera";
  action: string;
  dx?: number;
  dy?: number;
  dz?: number;
  amount?: number;
  camera?: string;
}

export interface StatusMessage {
  type: "status";
  payload: SimulationStatus;
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export type ServerMessage = StatusMessage | ErrorMessage;

export interface TaskDispatchRequest {
  instruction: string;
  scene: string;
  mode: string;
  total_time: number;
  fps: number;
  record_tcp: boolean;
  plan?: Record<string, unknown> | null;
}

export interface TaskRecord {
  id: string;
  instruction: string;
  scene: string;
  mode: string;
  status: string;
  created_at: string;
  message: string;
}

export interface LogRecord {
  id: string;
  timestamp: string;
  level: string;
  message: string;
}
