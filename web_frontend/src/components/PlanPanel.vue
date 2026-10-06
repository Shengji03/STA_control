<script setup lang="ts">
import { computed } from "vue";
import type { TaskDetail, ActiveTask } from "../types/simulation";
import AppIcon from "./AppIcon.vue";
const props = defineProps<{
  detail?: TaskDetail | null;
  task?: ActiveTask | null;
}>();
const phases = computed(() => props.detail?.plan?.execution_phases || []);
const report = computed(
  () => props.task?.optimization || props.detail?.optimization,
);
const modes: Record<string, string> = {
  single_L: "L 臂操作",
  single_R: "R 臂操作",
  parallel: "双臂并行",
  assist_R_then_L: "R 辅助 → L 操作",
  idle: "底盘导航",
};
</script>
<template>
  <section class="panel plan-panel">
    <div class="panel-heading">
      <div>
        <AppIcon name="task" />
        <h2>任务规划</h2>
      </div>
      <span class="subtle-tag">{{ phases.length }} 阶段</span>
    </div>
    <div v-if="!phases.length" class="empty-state compact">
      <p>
        {{
          task?.status === "planning"
            ? "正在理解任务并校验分工…"
            : "任务计划将在这里展示"
        }}
      </p>
    </div>
    <div v-else class="plan-body">
      <p class="reasoning">
        {{
          detail?.plan?.reasoning || "已完成主臂分工、辅助部署和可执行性校验。"
        }}
      </p>
      <div
        class="phase-item"
        v-for="(phase, i) in phases"
        :key="i"
        :class="{
          current: task?.status === 'running' && task.phase_index === i,
        }"
      >
        <span class="phase-number">{{ i + 1 }}</span>
        <div>
          <strong>{{ modes[phase.mode] || phase.mode }}</strong>
          <p>
            {{
              phase.nav
                ? `导航到 [${phase.nav.target.join(", ")}]`
                : "保持底盘位置"
            }}
            · {{ phase.steps.length + phase.cleanup_R.length }} 个动作
          </p>
          <details>
            <summary>查看技能序列</summary>
            <ol>
              <li
                v-for="(step, n) in [...phase.steps, ...phase.cleanup_R]"
                :key="n"
              >
                <span class="arm-label" :class="step.arm">{{ step.arm }}</span
                >{{ step.skill
                }}<small v-if="step.params.target_pos">{{
                  JSON.stringify(step.params.target_pos)
                }}</small>
              </li>
            </ol>
          </details>
        </div>
      </div>
      <div v-if="report" class="optimization-summary">
        <span
          >联合规划收益
          <b class="mono">{{ report.objective?.toFixed(3) }}</b></span
        ><span
          >求解
          <b>{{
            report.auxiliary_method === "exact" ? "精确枚举" : "覆盖贪心"
          }}</b></span
        ><span
          >规划耗时
          <b class="mono">{{ report.elapsed_ms?.toFixed(0) }} ms</b></span
        >
      </div>
      <p class="fine-print">
        收益为几何与运动代理指标；任务效果由执行后测量验证。
      </p>
    </div>
  </section>
</template>
