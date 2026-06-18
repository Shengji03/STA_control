import { defineStore } from "pinia";
import { computed, ref } from "vue";

import { fetchSimulationStatus, postCameraCommand, postSimulationControl } from "../api/http";
import { createWebSocketUrl } from "../api/urls";
import type { CameraCommand, ControlAction, ServerMessage, SimulationStatus } from "../types/simulation";

const disabledStatus: SimulationStatus = {
  state: "disabled",
  sim_time: 0,
  frame_index: 0,
  resolution: [0, 0],
  fps: 0,
  camera: {},
  available_cameras: [],
  active_task: null,
};

export const useSimulationStore = defineStore("simulation", () => {
  const status = ref<SimulationStatus>(disabledStatus);
  const frameUrl = ref<string>("");
  const connectionState = ref<"idle" | "connecting" | "online" | "offline">("idle");
  const lastError = ref<string>("");
  const measuredFps = ref(0);

  let socket: WebSocket | null = null;
  let reconnectTimer: number | undefined;
  let previousFrameUrl = "";
  let frameCount = 0;
  let fpsWindowStart = performance.now();

  const isOnline = computed(() => connectionState.value === "online");
  const cameraName = computed(() => status.value.camera?.fixed_camera || "free");

  async function refreshStatus(): Promise<void> {
    status.value = await fetchSimulationStatus();
  }

  function connect(): void {
    disconnect(false);
    connectionState.value = "connecting";
    socket = new WebSocket(createWebSocketUrl("/ws/simulation"));
    socket.binaryType = "blob";

    socket.addEventListener("open", () => {
      connectionState.value = "online";
      lastError.value = "";
    });

    socket.addEventListener("close", () => {
      connectionState.value = "offline";
      scheduleReconnect();
    });

    socket.addEventListener("error", () => {
      lastError.value = "WebSocket connection error";
      connectionState.value = "offline";
    });

    socket.addEventListener("message", (event) => {
      if (typeof event.data === "string") {
        handleTextMessage(event.data);
        return;
      }
      handleFrame(event.data as Blob);
    });
  }

  function disconnect(clearFrame = true): void {
    if (reconnectTimer !== undefined) {
      window.clearTimeout(reconnectTimer);
      reconnectTimer = undefined;
    }
    if (socket) {
      socket.close();
      socket = null;
    }
    if (clearFrame && previousFrameUrl) {
      URL.revokeObjectURL(previousFrameUrl);
      previousFrameUrl = "";
      frameUrl.value = "";
    }
  }

  async function sendControl(action: ControlAction): Promise<void> {
    const message = { type: "control", action };
    if (sendJson(message)) {
      return;
    }
    status.value = await postSimulationControl(action);
  }

  async function sendCamera(command: CameraCommand): Promise<void> {
    const payload = { ...command, type: "camera" as const };
    if (sendJson(payload)) {
      return;
    }
    status.value = await postCameraCommand(payload);
  }

  function sendJson(payload: object): boolean {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return false;
    }
    socket.send(JSON.stringify(payload));
    return true;
  }

  function handleTextMessage(raw: string): void {
    const parsed = JSON.parse(raw) as ServerMessage;
    if (parsed.type === "status") {
      status.value = parsed.payload;
      return;
    }
    if (parsed.type === "error") {
      lastError.value = parsed.message;
    }
  }

  function handleFrame(blob: Blob): void {
    const nextUrl = URL.createObjectURL(blob);
    if (previousFrameUrl) {
      URL.revokeObjectURL(previousFrameUrl);
    }
    previousFrameUrl = nextUrl;
    frameUrl.value = nextUrl;
    updateMeasuredFps();
  }

  function updateMeasuredFps(): void {
    frameCount += 1;
    const now = performance.now();
    const elapsed = now - fpsWindowStart;
    if (elapsed < 1000) {
      return;
    }
    measuredFps.value = Math.round((frameCount * 1000) / elapsed);
    frameCount = 0;
    fpsWindowStart = now;
  }

  function scheduleReconnect(): void {
    if (reconnectTimer !== undefined) {
      return;
    }
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = undefined;
      connect();
    }, 1200);
  }

  return {
    status,
    frameUrl,
    connectionState,
    lastError,
    measuredFps,
    isOnline,
    cameraName,
    refreshStatus,
    connect,
    disconnect,
    sendControl,
    sendCamera,
  };
});
