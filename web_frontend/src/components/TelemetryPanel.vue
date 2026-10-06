<script setup lang="ts">
import { computed, ref } from "vue";
import type { Sample, TaskDetail } from "../types/simulation";
import LineChart from "./LineChart.vue";
import TrajectoryPlot from "./TrajectoryPlot.vue";
import AppIcon from "./AppIcon.vue";
const props = defineProps<{
  samples: Sample[];
  detail?: TaskDetail | null;
  history?: boolean;
}>();
const arm = ref("L"),
  joint = ref(0),
  tab = ref("position");
const times = computed(() => props.samples.map((s) => s.t));
const actual = computed(() =>
  props.samples.map((s) => s.arms[arm.value]!.actual_tcp),
);
const planned = computed(() => props.detail?.preview?.paths[arm.value] || []);
const series = computed(() => {
  const a = props.samples.map((s) => s.arms[arm.value]!);
  if (tab.value === "position")
    return [
      {
        name: "控制参考",
        color: "#3276eb",
        dashed: true,
        values: a.map((s) => s.desired_q[joint.value]!),
      },
      {
        name: "实际关节角",
        color: "#089b88",
        values: a.map((s) => s.actual_q[joint.value]!),
      },
    ];
  if (tab.value === "error")
    return [
      {
        name: "末端跟踪偏差",
        color: "#089b88",
        values: a.map((s) => s.tcp_error_m * 1000),
      },
    ];
  if (tab.value === "force")
    return ["Fx", "Fy", "Fz"].map((name, i) => ({
      name,
      dashed: i === 1,
      color: ["#3276eb", "#089b88", "#a76e0c"][i]!,
      values: a.map((s) => s.force[i]!),
    }));
  return [
    {
      name: "STA 控制力矩",
      color: "#725ae5",
      values: a.map((s) => s.torque[joint.value]!),
    },
  ];
});
const unit = computed(
  () =>
    (
      ({ position: "rad", error: "mm", force: "N", torque: "N·m" }) as Record<
        string,
        string
      >
    )[tab.value]!,
);
</script>
<template>
  <section class="telemetry-section">
    <div class="section-heading">
      <div>
        <h2>轨迹与控制响应</h2>
        <p>规划轨迹为名义运动预览；误差曲线按运行时控制参考与实测数据计算</p>
      </div>
      <div class="segmented" aria-label="选择机械臂">
        <button
          v-for="a in ['L', 'R']"
          :key="a"
          :class="{ selected: arm === a }"
          :aria-pressed="arm === a"
          @click="arm = a"
        >
          {{ a }} 臂
        </button>
      </div>
    </div>
    <div class="chart-grid">
      <section class="panel chart-panel">
        <div class="panel-heading">
          <div>
            <AppIcon name="chart" />
            <h3>末端运动轨迹</h3>
          </div>
          <span class="subtle-tag">3D</span>
        </div>
        <TrajectoryPlot :planned="planned" :actual="actual" />
      </section>
      <section class="panel chart-panel">
        <div class="panel-heading">
          <div>
            <AppIcon name="chart" />
            <h3>{{ history ? "记录控制数据" : "实时控制数据" }}</h3>
          </div>
          <select
            v-if="tab === 'position' || tab === 'torque'"
            v-model.number="joint"
            aria-label="选择关节"
          >
            <option v-for="i in 6" :key="i" :value="i - 1">关节 {{ i }}</option>
          </select>
        </div>
        <div class="chart-tabs">
          <button
            v-for="(label, key) in {
              position: '关节跟踪',
              error: '末端误差',
              force: '接触力',
              torque: '控制力矩',
            }"
            :key="key"
            :class="{ selected: tab === key }"
            @click="tab = key"
          >
            {{ label }}
          </button>
        </div>
        <LineChart :series="series" :times="times" :unit="unit" />
        <p class="fine-print">
          {{
            history
              ? "显示本次任务完整运行记录，时间轴使用仿真时钟。"
              : "实时窗口保留最近 60 秒，完整记录可在任务历史中查看与导出。"
          }}
        </p>
      </section>
    </div>
  </section>
</template>
