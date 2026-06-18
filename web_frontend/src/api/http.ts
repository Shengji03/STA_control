import axios from "axios";

import type {
  CameraCommand,
  ControlAction,
  LogRecord,
  SimulationStatus,
  TaskDispatchRequest,
  TaskRecord,
} from "../types/simulation";

export const http = axios.create({
  baseURL: "/api",
  timeout: 8000,
});

export async function fetchSimulationStatus(): Promise<SimulationStatus> {
  const response = await http.get<SimulationStatus>("/simulation/status");
  return response.data;
}

export async function postSimulationControl(action: ControlAction): Promise<SimulationStatus> {
  const response = await http.post<SimulationStatus>("/simulation/control", { action });
  return response.data;
}

export async function postCameraCommand(command: CameraCommand): Promise<SimulationStatus> {
  const response = await http.post<SimulationStatus>("/simulation/camera", command);
  return response.data;
}

export async function dispatchTask(payload: TaskDispatchRequest): Promise<TaskRecord> {
  const response = await http.post<TaskRecord>("/tasks/dispatch", payload);
  return response.data;
}

export async function fetchTaskHistory(): Promise<TaskRecord[]> {
  const response = await http.get<TaskRecord[]>("/tasks/history");
  return response.data;
}

export async function fetchSystemLogs(): Promise<LogRecord[]> {
  const response = await http.get<LogRecord[]>("/logs");
  return response.data;
}
