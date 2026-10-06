<script setup lang="ts">
import { useSimulationStore } from "../stores/simulation";
import AppIcon from "./AppIcon.vue";
import { executeTask, errorMessage } from "../api/http";
const simulation = useSimulationStore();
const cameraNames: Record<string, string> = {
  cam_global: "全景",
  cam_overview: "总览",
  cam_side: "侧视",
  cam_valve: "阀门",
  cam_bird: "俯视",
  cam_shade: "遮光",
};
async function execute() {
  if (!simulation.task) return;
  try {
    await executeTask(simulation.task.id);
    await simulation.refreshStatus();
  } catch (e) {
    simulation.lastError = errorMessage(e);
  }
}
</script>
<template>
  <section class="panel control-panel">
    <div class="control-actions">
      <button
        v-if="simulation.task?.status === 'planned'"
        class="primary"
        @click="execute"
      >
        <AppIcon name="play" />执行计划</button
      ><button
        :disabled="simulation.task?.status !== 'running'"
        @click="simulation.sendControl('pause')"
      >
        <AppIcon name="pause" />暂停</button
      ><button
        :disabled="simulation.task?.status !== 'paused'"
        @click="simulation.sendControl('resume')"
      >
        <AppIcon name="play" />继续</button
      ><button
        :disabled="!simulation.status.active_task"
        class="danger-quiet"
        @click="simulation.sendControl('stop')"
      >
        <AppIcon name="stop" />停止任务</button
      ><button
        :disabled="!!simulation.status.active_task"
        @click="simulation.sendControl('reset')"
      >
        <AppIcon name="reset" />重置场景
      </button>
    </div>
    <div class="camera-row">
      <AppIcon name="camera" /><span>视角</span
      ><button
        :class="{ selected: simulation.cameraName === 'free' }"
        @click="simulation.sendCamera({ action: 'reset' })"
      >
        自由</button
      ><button
        v-for="camera in simulation.status.available_cameras"
        :key="camera"
        :class="{ selected: simulation.cameraName === camera }"
        @click="simulation.sendCamera({ action: 'set_fixed', camera })"
      >
        {{ cameraNames[camera] || camera }}</button
      ><button
        aria-label="放大画面"
        @click="simulation.sendCamera({ action: 'zoom', amount: -0.4 })"
      >
        ＋</button
      ><button
        aria-label="缩小画面"
        @click="simulation.sendCamera({ action: 'zoom', amount: 0.4 })"
      >
        −
      </button>
    </div>
  </section>
</template>
