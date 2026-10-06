<script setup lang="ts">
import { computed } from "vue";
import { useSimulationStore } from "../stores/simulation";
import AppIcon from "./AppIcon.vue";
const simulation = useSimulationStore();
const latest = computed(() => simulation.status.telemetry);
</script>
<template>
  <section class="panel">
    <div class="panel-heading">
      <div>
        <AppIcon name="robot" />
        <h2>机械臂状态</h2>
      </div>
      <span class="subtle-tag">双 UR5e</span>
    </div>
    <div class="arm-card" v-for="arm in ['L', 'R']" :key="arm">
      <div class="arm-title">
        <span class="arm-marker" :class="arm">{{ arm }}</span
        ><strong>{{ arm === "L" ? "左机械臂" : "右机械臂" }}</strong
        ><span class="subtle-tag">{{
          latest?.arms[arm]?.skill || "待命"
        }}</span>
      </div>
      <div class="arm-values">
        <div>
          <span>末端位置 / m</span
          ><b class="mono">{{
            latest?.arms[arm]?.actual_tcp
              .map((v) => v.toFixed(3))
              .join(" / ") || "—"
          }}</b>
        </div>
        <div>
          <span>跟踪偏差</span
          ><b class="mono">{{
            latest
              ? (latest.arms[arm]!.tcp_error_m * 1000).toFixed(2) + " mm"
              : "—"
          }}</b>
        </div>
      </div>
    </div>
    <div class="task-message" role="status">
      {{
        simulation.task?.message || "选择示例或输入自然语言，开始一次协同作业。"
      }}
    </div>
  </section>
</template>
