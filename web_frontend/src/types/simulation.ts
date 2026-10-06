export type ControlAction = "pause" | "resume" | "reset" | "stop";
export interface CameraCommand {
  action: string;
  dx?: number;
  dy?: number;
  dz?: number;
  amount?: number;
  camera?: string;
  type?: string;
}
export interface CameraState {
  mode?: string;
  fixed_camera?: string | null;
  azimuth?: number;
  elevation?: number;
  distance?: number;
  lookat?: number[];
}
export interface ArmTelemetry {
  actual_q: number[];
  desired_q: number[];
  error_q: number[];
  actual_tcp: number[];
  reference_tcp: number[];
  tcp_error_m: number;
  force: number[];
  moment: number[];
  torque: number[];
  skill: string;
  skill_index: number;
  skill_count: number;
}
export interface Sample {
  t: number;
  phase: number;
  arms: Record<string, ArmTelemetry>;
  qpos?: number[];
}
export interface Optimization {
  objective?: number;
  main_candidate_count?: number;
  auxiliary_candidate_count?: number;
  auxiliary_method?: string;
  rejected_policy_count?: number;
  assignments?: {
    goal_id: string;
    arm: string;
    phase_id: number;
    score: number;
  }[];
  auxiliaries?: { arm: string; covers: string[]; position: number[] }[];
  elapsed_ms?: number;
  gain_over_preference?: number | null;
}
export interface ExecutionEffects {
  verified?: boolean;
  goals?: {
    goal_id: string;
    object: string;
    verified: boolean;
    angle_error_rad?: number;
    shade_fraction?: number | null;
    board_return_error_m?: number;
    reason?: string;
  }[];
}
export interface ActiveTask {
  id: string;
  instruction: string;
  status: string;
  message: string;
  progress: number;
  runner_state: string;
  phase_index: number;
  phase_count: number;
  optimization?: Optimization;
  execution_effects?: ExecutionEffects;
}
export interface SimulationStatus {
  state: string;
  sim_time: number;
  frame_index: number;
  resolution: number[];
  fps: number;
  camera: CameraState;
  available_cameras: string[];
  active_task?: ActiveTask | null;
  last_task?: ActiveTask | null;
  scene?: string;
  telemetry?: Sample | null;
  overlay?: Record<string, boolean>;
  replay?: boolean;
  message?: string;
}
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
  completed_at?: string;
  message: string;
  optimization?: Optimization;
  execution_effects?: ExecutionEffects;
  sample_count?: number;
  metrics?: Record<string, Metrics>;
}
export interface Metrics {
  tcp_rmse_m: number;
  joint_rmse_rad: number;
  peak_force_N: number;
  peak_torque_Nm: number;
}
export interface PlanStep {
  arm: string;
  skill: string;
  description?: string;
  goal_id?: string;
  role?: string;
  params: Record<string, unknown>;
}
export interface ExecutionPhase {
  id: number;
  mode: string;
  nav?: { target: number[]; yaw?: number | null } | null;
  goal_ids: string[];
  steps: PlanStep[];
  cleanup_R: PlanStep[];
}
export interface TaskDetail {
  task?: TaskRecord;
  task_id?: string;
  scene?: string;
  plan?: {
    reasoning?: string;
    goals?: {
      id: string;
      object: string;
      operation: string;
      shade: string;
      angle: number;
    }[];
    execution_phases?: ExecutionPhase[];
    plan?: PlanStep[];
  };
  preview?: {
    duration: number;
    paths: Record<string, number[][]>;
    timeline: { t: number; phase: number; L: number[]; R: number[] }[];
  };
  samples: Sample[];
  optimization?: Optimization;
  execution_effects?: ExecutionEffects;
  metrics?: Record<string, Metrics>;
}
export interface Scenario {
  id: string;
  name: string;
  description: string;
  instruction: string;
  scene: string;
  plan: Record<string, unknown>;
}
export interface LogRecord {
  id: string;
  timestamp: string;
  level: string;
  message: string;
}
export type ServerMessage =
  | { type: "status"; payload: SimulationStatus }
  | { type: "error"; message: string };
export function statusLabel(value?: string): string {
  return (
    (
      {
        planning: "规划中",
        planned: "待执行",
        running: "执行中",
        paused: "已暂停",
        completed: "已完成",
        failed: "失败",
        cancelled: "已停止",
        interrupted: "已中断",
        disabled: "未启用",
        replay: "回放中",
      } as Record<string, string>
    )[value || ""] || "待命"
  );
}

export function verificationLabel(
  effects?: ExecutionEffects | null,
  status?: string,
): string {
  if (!effects?.goals?.length)
    return status === "completed" ? "无需操作校验" : "未测量";
  return effects.verified ? "通过" : "未通过";
}
