import { defineStore } from "pinia";
import { computed, ref, shallowRef } from "vue";
import {
  fetchSimulationStatus,
  postCameraCommand,
  postSimulationControl,
  fetchTaskDetail,
  setOverlay,
  errorMessage,
} from "../api/http";
import { createWebSocketUrl } from "../api/urls";
import type {
  CameraCommand,
  ControlAction,
  ServerMessage,
  SimulationStatus,
  Sample,
  TaskDetail,
} from "../types/simulation";
export const useSimulationStore = defineStore("simulation", () => {
  const status = ref<SimulationStatus>({
    state: "disabled",
    sim_time: 0,
    frame_index: 0,
    resolution: [0, 0],
    fps: 20,
    camera: {},
    available_cameras: [],
    active_task: null,
  });
  const frameUrl = ref("");
  const connectionState = ref("idle");
  const lastError = ref("");
  const measuredFps = ref(0);
  const samples = shallowRef<Sample[]>([]);
  const detail = shallowRef<TaskDetail | null>(null);
  const task = computed(
    () => status.value.active_task || status.value.last_task,
  );
  const cameraName = computed(() =>
    status.value.camera.mode === "fixed"
      ? status.value.camera.fixed_camera
      : "free",
  );
  const isOnline = computed(() => connectionState.value === "online");
  let socket: WebSocket | null = null,
    reconnectTimer: number | undefined;
  let wanted = false,
    loadingId = "",
    detailKey = "",
    requestKey = "",
    sampleTask = "",
    oldUrl = "",
    count = 0,
    started = performance.now();
  function liveWindow(rows: Sample[]) {
    const end = rows.at(-1)?.t ?? 0;
    return rows.filter((s) => s.t >= end - 60).slice(-1200);
  }
  function accept(payload: SimulationStatus) {
    status.value = payload;
    if (payload.state === "error" && payload.message)
      lastError.value = payload.message;
    const active = payload.active_task || payload.last_task;
    if (!active && sampleTask) {
      sampleTask = "";
      samples.value = [];
      detail.value = null;
      detailKey = "";
    }
    if (active && active.id !== sampleTask) {
      sampleTask = active.id;
      samples.value = [];
      detail.value = null;
      detailKey = "";
    }
    if (payload.replay && detail.value)
      samples.value = liveWindow(
        detail.value.samples.filter((s) => s.t <= payload.sim_time),
      );
    const sample = payload.telemetry;
    if (sample && !payload.replay && sample.t !== samples.value.at(-1)?.t)
      samples.value = liveWindow([...samples.value, sample]);
    const key = active
      ? active.id +
        ":" +
        (active.status === "planning"
          ? "planning"
          : active.status === "planned"
            ? "planned"
            : active.status === "completed" ||
                active.status === "failed" ||
                active.status === "cancelled"
              ? "finished"
              : "running")
      : "";
    if (active && key !== detailKey && loadingId !== key) {
      loadingId = key;
      requestKey = key;
      void fetchTaskDetail(active.id)
        .then((d) => {
          if (sampleTask === active.id && requestKey === key) {
            detail.value = d;
            detailKey = key;
            if (d.samples.length)
              samples.value = liveWindow(
                status.value.replay
                  ? d.samples.filter((s) => s.t <= status.value.sim_time)
                  : d.samples,
              );
          }
        })
        .catch(() => {})
        .finally(() => {
          if (loadingId === key) loadingId = "";
        });
    }
  }
  async function refreshStatus() {
    try {
      accept(await fetchSimulationStatus());
    } catch (e) {
      lastError.value = errorMessage(e);
    }
  }
  function connect() {
    wanted = true;
    if (
      socket &&
      (socket.readyState === WebSocket.OPEN ||
        socket.readyState === WebSocket.CONNECTING)
    )
      return;
    connectionState.value = "connecting";
    const current = new WebSocket(createWebSocketUrl("/ws/simulation"));
    socket = current;
    current.binaryType = "blob";
    current.onopen = () => {
      if (socket !== current) return;
      connectionState.value = "online";
      lastError.value = "";
    };
    current.onmessage = (event) => {
      if (socket !== current) return;
      if (typeof event.data === "string") {
        try {
          const data = JSON.parse(event.data) as ServerMessage;
          if (data.type === "status") accept(data.payload);
          else lastError.value = data.message;
        } catch {
          lastError.value = "收到无法解析的状态消息";
        }
        return;
      }
      const next = URL.createObjectURL(event.data as Blob);
      const previous = oldUrl;
      oldUrl = next;
      frameUrl.value = next;
      if (previous) window.setTimeout(() => URL.revokeObjectURL(previous), 250);
      count++;
      const elapsed = performance.now() - started;
      if (elapsed > 1000) {
        measuredFps.value = Math.round((count * 1000) / elapsed);
        count = 0;
        started = performance.now();
      }
    };
    current.onclose = () => {
      if (socket !== current) return;
      socket = null;
      connectionState.value = "offline";
      if (wanted && reconnectTimer === undefined)
        reconnectTimer = window.setTimeout(() => {
          reconnectTimer = undefined;
          connect();
        }, 1500);
    };
    current.onerror = () => {
      if (socket === current) lastError.value = "画面连接中断，正在尝试重连";
    };
  }
  function disconnect() {
    wanted = false;
    if (reconnectTimer !== undefined) {
      clearTimeout(reconnectTimer);
      reconnectTimer = undefined;
    }
    const old = socket;
    socket = null;
    old?.close();
    if (oldUrl) URL.revokeObjectURL(oldUrl);
    frameUrl.value = "";
    oldUrl = "";
    connectionState.value = "idle";
  }
  async function sendControl(action: ControlAction) {
    try {
      accept(await postSimulationControl(action));
      lastError.value = "";
    } catch (e) {
      lastError.value = errorMessage(e);
    }
  }
  async function sendCamera(command: CameraCommand) {
    try {
      accept(await postCameraCommand(command));
    } catch (e) {
      lastError.value = errorMessage(e);
    }
  }
  async function toggleOverlay(key: string) {
    const value: Record<string, boolean> = {
      planned: true,
      actual: true,
      L: true,
      R: true,
      ...status.value.overlay,
    };
    value[key] = !value[key];
    try {
      await setOverlay(value);
      await refreshStatus();
    } catch (e) {
      lastError.value = errorMessage(e);
    }
  }
  return {
    status,
    frameUrl,
    connectionState,
    lastError,
    measuredFps,
    samples,
    detail,
    task,
    cameraName,
    isOnline,
    refreshStatus,
    connect,
    disconnect,
    sendControl,
    sendCamera,
    toggleOverlay,
  };
});
