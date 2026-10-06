<script setup lang="ts">
import { computed } from "vue";
import { RouterLink } from "vue-router";
import SimulationViewport from "../components/SimulationViewport.vue";
import StatusPanel from "../components/StatusPanel.vue";
import ControlPanel from "../components/ControlPanel.vue";
import TelemetryPanel from "../components/TelemetryPanel.vue";
import PlanPanel from "../components/PlanPanel.vue";
import AppIcon from "../components/AppIcon.vue";
import { useSimulationStore } from "../stores/simulation";
import { statusLabel } from "../types/simulation";
const simulation = useSimulationStore();
const task = computed(() => simulation.task);
</script>
<template>
  <section class="page">
    <div class="page-heading">
      <div>
        <div class="eyebrow">LIVE WORKSPACE</div>
        <h1>仿真监控</h1>
        <p>实时观察双臂协作、运动轨迹与控制响应</p>
      </div>
      <RouterLink class="button primary" to="/tasks/dispatch"
        ><AppIcon name="task" />新建任务<AppIcon name="arrow" :size="16"
      /></RouterLink>
    </div>
    <div class="stat-grid">
      <div class="stat-card">
        <span>任务状态</span
        ><strong
          ><span
            class="live-dot"
            :class="{ online: task?.status === 'running' }"
          ></span
          >{{
            simulation.status.replay ? "历史回放" : statusLabel(task?.status)
          }}</strong
        ><small>{{ task?.id || "等待下发任务" }}</small>
      </div>
      <div class="stat-card">
        <span>仿真时间</span
        ><strong class="mono"
          >{{ simulation.status.sim_time.toFixed(2) }}<em>s</em></strong
        ><small>以仿真时钟记录</small>
      </div>
      <div class="stat-card">
        <span>执行进度</span
        ><strong class="mono"
          >{{ Math.round((task?.progress || 0) * 100) }}<em>%</em></strong
        ><small>{{
          task?.phase_count
            ? `阶段 ${Math.min(task.phase_index + 1, task.phase_count)} / ${task.phase_count}`
            : "按阶段与技能统计"
        }}</small>
      </div>
      <div class="stat-card">
        <span>画面帧率</span
        ><strong class="mono"
          >{{ simulation.measuredFps || 0 }}<em>FPS</em></strong
        ><small>{{ simulation.status.resolution.join(" × ") }} 像素</small>
      </div>
    </div>
    <div class="monitor-grid">
      <div class="monitor-main"><SimulationViewport /><ControlPanel /></div>
      <div class="monitor-rail">
        <StatusPanel /><PlanPanel :detail="simulation.detail" :task="task" />
      </div>
    </div>
    <TelemetryPanel :samples="simulation.samples" :detail="simulation.detail" />
  </section>
</template>
