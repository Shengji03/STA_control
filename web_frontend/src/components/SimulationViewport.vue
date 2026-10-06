<script setup lang="ts">
import { ref } from "vue";
import { useSimulationStore } from "../stores/simulation";
import AppIcon from "./AppIcon.vue";
const simulation = useSimulationStore();
let point: { x: number; y: number } | null = null;
let lastSend = 0;
const dragging = ref(false);
function down(e: PointerEvent) {
  point = { x: e.clientX, y: e.clientY };
  dragging.value = true;
  (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
}
function move(e: PointerEvent) {
  if (!point) return;
  const now = performance.now();
  if (now - lastSend < 65) return;
  lastSend = now;
  const dx = e.clientX - point.x,
    dy = e.clientY - point.y;
  point = { x: e.clientX, y: e.clientY };
  void simulation.sendCamera({ action: "orbit", dx: dx * 0.35, dy: dy * 0.25 });
}
function up(e: PointerEvent) {
  point = null;
  dragging.value = false;
  const el = e.currentTarget as HTMLElement;
  if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
}
function wheel(e: WheelEvent) {
  void simulation.sendCamera({ action: "zoom", amount: e.deltaY * 0.003 });
}
const labels: Record<string, string> = {
  planned: "规划轨迹",
  actual: "实际轨迹",
  L: "L 臂",
  R: "R 臂",
};
</script>
<template>
  <section class="panel viewport-shell" aria-label="MuJoCo 仿真画面">
    <div class="panel-heading">
      <div>
        <AppIcon name="monitor" />
        <h2>MuJoCo 实时仿真</h2>
      </div>
      <span
        class="badge"
        :data-status="simulation.isOnline ? 'completed' : 'failed'"
        >{{ simulation.isOnline ? "LIVE" : "连接中" }}</span
      >
    </div>
    <div
      class="stream-surface"
      :class="{ dragging }"
      @pointerdown="down"
      @pointermove="move"
      @pointerup="up"
      @pointercancel="up"
      @wheel.prevent="wheel"
    >
      <img
        v-if="simulation.frameUrl"
        :src="simulation.frameUrl"
        alt="MuJoCo 双臂机器人仿真画面与轨迹"
        draggable="false"
      />
      <div v-else class="empty-state">
        <AppIcon name="robot" :size="44" /><strong>正在建立画面连接</strong>
        <p>请确认仿真后端已启动</p>
      </div>
      <span class="viewport-label">{{
        simulation.status.replay
          ? "历史状态回放"
          : simulation.status.scene === "scene4_pipeline.xml"
            ? "管道作业场景"
            : "化工遮光场景"
      }}</span>
    </div>
    <div class="viewport-toolbar">
      <div class="toggle-group">
        <button
          v-for="(label, key) in labels"
          :key="key"
          :class="{ selected: simulation.status.overlay?.[key] }"
          :aria-pressed="simulation.status.overlay?.[key]"
          @click="simulation.toggleOverlay(key)"
        >
          <span
            v-if="key === 'planned' || key === 'actual'"
            class="legend-line"
            :class="{ dashed: key === 'planned' }"
          ></span
          >{{ label }}
        </button>
      </div>
      <span class="muted">拖动旋转 · 滚轮缩放</span>
    </div>
  </section>
</template>
