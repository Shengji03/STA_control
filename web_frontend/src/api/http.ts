import axios from "axios";
import type {
  CameraCommand,
  ControlAction,
  LogRecord,
  SimulationStatus,
  TaskDispatchRequest,
  TaskRecord,
  TaskDetail,
  Scenario,
} from "../types/simulation";
export const http = axios.create({ baseURL: "/api", timeout: 15000 });
export async function fetchSimulationStatus(): Promise<SimulationStatus> {
  return (await http.get("/simulation/status")).data;
}
export async function postSimulationControl(
  action: ControlAction,
): Promise<SimulationStatus> {
  return (await http.post("/simulation/control", { action })).data;
}
export async function postCameraCommand(
  command: CameraCommand,
): Promise<SimulationStatus> {
  return (await http.post("/simulation/camera", command)).data;
}
export async function dispatchTask(
  payload: TaskDispatchRequest,
): Promise<TaskRecord> {
  return (await http.post("/tasks/dispatch", payload)).data;
}
export async function fetchTaskHistory(): Promise<TaskRecord[]> {
  return (await http.get("/tasks/history")).data;
}
export async function fetchSystemLogs(): Promise<LogRecord[]> {
  return (await http.get("/logs")).data;
}
export async function fetchScenarios(): Promise<Scenario[]> {
  return (await http.get("/scenarios")).data;
}
export async function fetchTaskDetail(id: string): Promise<TaskDetail> {
  return (await http.get(`/tasks/${encodeURIComponent(id)}/detail`)).data;
}
export async function executeTask(id: string): Promise<void> {
  await http.post(`/tasks/${encodeURIComponent(id)}/execute`);
}
export async function replayTask(id: string, index: number): Promise<void> {
  await http.post(`/tasks/${encodeURIComponent(id)}/replay`, { index });
}
export async function setOverlay(
  overlay: Record<string, boolean>,
): Promise<void> {
  await http.post("/simulation/overlay", overlay);
}
export function errorMessage(error: unknown): string {
  const e = error as {
    response?: { data?: { detail?: unknown } };
    message?: string;
  };
  const d = e.response?.data?.detail;
  return typeof d === "string"
    ? d
    : Array.isArray(d)
      ? "请检查任务参数的范围与格式"
      : e.message || "请求失败，请检查后端连接";
}
