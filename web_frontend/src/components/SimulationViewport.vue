<script setup lang="ts">
import { computed, ref } from "vue";

import { useSimulationStore } from "../stores/simulation";

const simulation = useSimulationStore();
const isDragging = ref(false);
const lastPoint = ref<{ x: number; y: number } | null>(null);

const connectionText = computed(() => {
  if (simulation.connectionState === "online") return "已连接";
  if (simulation.connectionState === "connecting") return "连接中";
  if (simulation.connectionState === "offline") return "离线";
  return "未连接";
});

function onPointerDown(event: PointerEvent): void {
  isDragging.value = true;
  lastPoint.value = { x: event.clientX, y: event.clientY };
  (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
}

function onPointerMove(event: PointerEvent): void {
  if (!isDragging.value || !lastPoint.value) return;
  const dx = event.clientX - lastPoint.value.x;
  const dy = event.clientY - lastPoint.value.y;
  lastPoint.value = { x: event.clientX, y: event.clientY };
  void simulation.sendCamera({ action: "orbit", dx: dx * 0.35, dy: dy * 0.25 });
}

function onPointerUp(event: PointerEvent): void {
  isDragging.value = false;
  lastPoint.value = null;
  (event.currentTarget as HTMLElement).releasePointerCapture(event.pointerId);
}

function onWheel(event: WheelEvent): void {
  event.preventDefault();
  void simulation.sendCamera({ action: "zoom", amount: event.deltaY * 0.003 });
}
</script>

<template>
  <section class="viewport-shell" aria-label="MuJoCo 仿真画面">
    <header class="viewport-header">
      <div>
        <h1>STA Control</h1>
        <p :data-state="simulation.connectionState">{{ connectionText }}</p>
      </div>
      <div class="metrics">
        <span class="state-pill" :data-state="simulation.status.state">{{ simulation.status.state }}</span>
        <span><strong>{{ simulation.status.sim_time.toFixed(4) }}</strong>s</span>
        <span><strong>{{ simulation.measuredFps || simulation.status.fps }}</strong> FPS</span>
      </div>
    </header>

    <div
      class="stream-surface"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
      @pointercancel="onPointerUp"
      @wheel="onWheel"
    >
      <img v-if="simulation.frameUrl" :src="simulation.frameUrl" alt="MuJoCo realtime render" draggable="false" />
      <div v-else class="empty-frame">等待仿真画面</div>
    </div>
  </section>
</template>

