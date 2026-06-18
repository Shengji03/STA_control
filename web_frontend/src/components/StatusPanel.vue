<script setup lang="ts">
import { computed } from "vue";

import { useSimulationStore } from "../stores/simulation";

const simulation = useSimulationStore();

const resolutionText = computed(() => {
  const [width, height] = simulation.status.resolution;
  if (!width || !height) return "--";
  return `${width} x ${height}`;
});

const activeTaskText = computed(() => simulation.status.active_task?.id || "--");
const activeTaskStatus = computed(() => simulation.status.active_task?.status || "--");
const activeTaskMessage = computed(() => simulation.status.active_task?.message || "--");

const activeTaskProgress = computed(() => {
  const progress = simulation.status.active_task?.progress;
  if (progress === undefined || progress === null) return "--";
  return `${Math.round(progress * 100)}%`;
});
</script>

<template>
  <section class="status-panel" aria-label="仿真状态">
    <h2>状态</h2>
    <dl>
      <div>
        <dt>连接</dt>
        <dd>{{ simulation.connectionState }}</dd>
      </div>
      <div>
        <dt>分辨率</dt>
        <dd>{{ resolutionText }}</dd>
      </div>
      <div>
        <dt>帧序号</dt>
        <dd>{{ simulation.status.frame_index }}</dd>
      </div>
      <div>
        <dt>相机模式</dt>
        <dd>{{ simulation.status.camera.mode || "--" }}</dd>
      </div>
      <div>
        <dt>当前相机</dt>
        <dd>{{ simulation.cameraName }}</dd>
      </div>
      <div>
        <dt>当前任务</dt>
        <dd>{{ activeTaskText }}</dd>
      </div>
      <div>
        <dt>任务状态</dt>
        <dd>{{ activeTaskStatus }}</dd>
      </div>
      <div>
        <dt>任务信息</dt>
        <dd>{{ activeTaskMessage }}</dd>
      </div>
      <div>
        <dt>任务进度</dt>
        <dd>{{ activeTaskProgress }}</dd>
      </div>
      <div v-if="simulation.lastError">
        <dt>错误</dt>
        <dd>{{ simulation.lastError }}</dd>
      </div>
    </dl>
  </section>
</template>
