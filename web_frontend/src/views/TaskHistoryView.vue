<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, computed } from "vue";
import {
  fetchTaskHistory,
  fetchTaskDetail,
  replayTask,
  dispatchTask,
  errorMessage,
} from "../api/http";
import type { TaskRecord, TaskDetail } from "../types/simulation";
import { statusLabel, verificationLabel } from "../types/simulation";
import { useSimulationStore } from "../stores/simulation";
import { useRouter } from "vue-router";
import AppIcon from "../components/AppIcon.vue";
import TelemetryPanel from "../components/TelemetryPanel.vue";
import PlanPanel from "../components/PlanPanel.vue";
const tasks = ref<TaskRecord[]>([]),
  detail = ref<TaskDetail | null>(null),
  message = ref(""),
  selected = ref(""),
  index = ref(0),
  playing = ref(false),
  compare = ref<string[]>([]);
const simulation = useSimulationStore(),
  router = useRouter();
let timer: number | undefined, refreshTimer: number | undefined;
async function load() {
  try {
    tasks.value = await fetchTaskHistory();
  } catch (e) {
    message.value = errorMessage(e);
  }
}
async function inspect(id: string) {
  stopReplay();
  selected.value = id;
  detail.value = null;
  try {
    const result = await fetchTaskDetail(id);
    if (selected.value !== id) return;
    detail.value = result;
    index.value = 0;
  } catch (e) {
    if (selected.value === id) message.value = errorMessage(e);
  }
}
function stopReplay() {
  playing.value = false;
  if (timer) clearTimeout(timer);
  timer = undefined;
}
async function seek() {
  if (!detail.value?.samples.length) return;
  try {
    await replayTask(selected.value, index.value);
    await simulation.refreshStatus();
  } catch (e) {
    stopReplay();
    message.value = errorMessage(e);
  }
}
async function play() {
  if (playing.value) {
    stopReplay();
    return;
  }
  if (!detail.value?.samples.length) return;
  if (index.value >= detail.value.samples.length - 1) index.value = 0;
  playing.value = true;
  async function tick() {
    if (!playing.value || !detail.value) return;
    await seek();
    if (!playing.value) return;
    if (index.value === detail.value.samples.length - 1) {
      stopReplay();
      return;
    }
    timer = window.setTimeout(() => {
      if (!playing.value || !detail.value) return;
      index.value = Math.min(index.value + 4, detail.value.samples.length - 1);
      void tick();
    }, 200);
  }
  void tick();
}
async function repeat() {
  if (!detail.value?.plan) return;
  try {
    await dispatchTask({
      instruction: detail.value.task?.instruction || "",
      scene: detail.value.scene || "scene5_glare.xml",
      mode: "实时仿真",
      total_time: 180,
      fps: 20,
      record_tcp: true,
      plan: detail.value.plan,
    });
    await simulation.refreshStatus();
    await router.push("/");
  } catch (e) {
    message.value = errorMessage(e);
  }
}
const comparison = computed(() =>
  tasks.value.filter((t) => compare.value.includes(t.id)),
);
const metrics = computed(
  () => detail.value?.metrics || detail.value?.task?.metrics || {},
);
onMounted(() => {
  void load();
  refreshTimer = window.setInterval(() => {
    void load();
  }, 2000);
});
onBeforeUnmount(() => {
  stopReplay();
  if (refreshTimer) clearInterval(refreshTimer);
});
</script>
<template>
  <section class="page">
    <div class="page-heading">
      <div>
        <div class="eyebrow">EXPERIMENT RECORDS</div>
        <h1>任务历史</h1>
        <p>保留配置、规划与原始运行数据，支持结果对比和状态回放</p>
      </div>
      <button @click="load"><AppIcon name="reset" />刷新记录</button>
    </div>
    <div v-if="message" class="alert" role="alert">{{ message }}</div>
    <section class="panel history-panel">
      <div class="panel-heading">
        <h2>
          实验记录 <span class="muted">{{ tasks.length }}</span>
        </h2>
        <span class="fine-print">勾选记录以比较实验指标</span>
      </div>
      <div class="table-scroll">
        <table class="data-table">
          <thead>
            <tr>
              <th>对比</th>
              <th>任务与场景</th>
              <th>状态</th>
              <th>开始时间</th>
              <th>采样</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="task in tasks"
              :key="task.id"
              :class="{ selected: selected === task.id }"
            >
              <td>
                <input
                  type="checkbox"
                  v-model="compare"
                  :value="task.id"
                  :aria-label="'对比任务 ' + task.id"
                />
              </td>
              <td>
                <strong>{{ task.instruction }}</strong
                ><small
                  >{{
                    task.scene === "scene5_glare.xml"
                      ? "遮光作业场景"
                      : "管道作业场景"
                  }}
                  · {{ task.mode }}</small
                >
              </td>
              <td>
                <span class="badge" :data-status="task.status">{{
                  statusLabel(task.status)
                }}</span>
              </td>
              <td class="mono">
                {{
                  new Date(task.created_at).toLocaleString("zh-CN", {
                    hour12: false,
                  })
                }}
              </td>
              <td class="mono">{{ task.sample_count || 0 }}</td>
              <td>
                <button class="text-button" @click="inspect(task.id)">
                  查看结果 <AppIcon name="arrow" :size="14" />
                </button>
              </td>
            </tr>
            <tr v-if="!tasks.length">
              <td colspan="6">
                <div class="empty-state">
                  <AppIcon name="history" :size="32" />
                  <p>完成第一项任务后，实验记录将保存在这里。</p>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
    <section v-if="comparison.length" class="panel comparison-panel">
      <div class="panel-heading">
        <h2>实验指标对比</h2>
        <span class="fine-print">相同场景与任务条件下的结果才适合直接比较</span>
      </div>
      <div class="table-scroll">
        <table class="data-table">
          <thead>
            <tr>
              <th>任务</th>
              <th>规划收益</th>
              <th>L 末端 RMSE / mm</th>
              <th>R 末端 RMSE / mm</th>
              <th>实际效果</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="t in comparison" :key="t.id">
              <td>{{ t.instruction }}</td>
              <td>{{ t.optimization?.objective?.toFixed(3) || "—" }}</td>
              <td>
                {{
                  t.metrics?.L
                    ? (t.metrics.L.tcp_rmse_m * 1000).toFixed(2)
                    : "—"
                }}
              </td>
              <td>
                {{
                  t.metrics?.R
                    ? (t.metrics.R.tcp_rmse_m * 1000).toFixed(2)
                    : "—"
                }}
              </td>
              <td>{{ verificationLabel(t.execution_effects, t.status) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
    <div v-if="detail" class="result-section">
      <div class="section-heading">
        <div>
          <h2>实验详情</h2>
          <p>
            {{ detail.task?.instruction }} ·
            {{ statusLabel(detail.task?.status) }}
          </p>
        </div>
        <div class="control-actions">
          <a class="button" :href="`/api/tasks/${selected}/export`" download
            ><AppIcon name="download" />导出 JSON</a
          ><button
            :disabled="!!simulation.status.active_task || !detail.plan"
            @click="repeat"
          >
            <AppIcon name="play" />复现任务
          </button>
        </div>
      </div>
      <div class="stat-grid">
        <div v-for="arm in ['L', 'R']" :key="arm" class="stat-card">
          <span>{{ arm }} 臂末端 RMSE</span
          ><strong class="mono"
            >{{
              metrics[arm] ? (metrics[arm]!.tcp_rmse_m * 1000).toFixed(2) : "—"
            }}<em>mm</em></strong
          ><small>相对于运行时控制参考</small>
        </div>
        <div class="stat-card">
          <span>执行效果</span
          ><strong>{{
            verificationLabel(detail.execution_effects, detail.task?.status)
          }}</strong
          ><small>根据实际角度、遮光及放回效果</small>
        </div>
        <div class="stat-card">
          <span>数据记录</span
          ><strong class="mono">{{ detail.samples.length }}<em>点</em></strong
          ><small>含仿真状态，可重建画面</small>
        </div>
      </div>
      <section v-if="detail.samples.length" class="panel replay-panel">
        <div>
          <AppIcon name="history" />
          <h3>仿真状态回放</h3>
        </div>
        <button :disabled="!!simulation.status.active_task" @click="play">
          {{ playing ? "暂停回放" : "播放回放" }}</button
        ><label class="replay-slider"
          >时间位置<input
            type="range"
            v-model.number="index"
            :max="detail.samples.length - 1"
            min="0"
            :disabled="!!simulation.status.active_task"
            @input="stopReplay"
            @change="seek" /></label
        ><span class="mono">{{ detail.samples[index]?.t.toFixed(2) }} s</span
        ><button @click="router.push('/')">查看仿真画面</button>
      </section>
      <section v-if="detail.samples.length" class="panel recorded-frame">
        <div class="panel-heading">
          <h3>历史仿真画面</h3>
          <span class="subtle-tag">逐帧状态重建</span>
        </div>
        <img
          v-if="
            simulation.status.replay &&
            simulation.task?.id === selected &&
            simulation.frameUrl
          "
          :src="simulation.frameUrl"
          alt="当前历史任务的 MuJoCo 状态回放画面"
        />
        <div v-else class="empty-state">
          <p>拖动上方时间位置或点击播放，在这里查看历史机械臂动作。</p>
        </div>
      </section>
      <div
        v-for="g in detail.execution_effects?.goals || []"
        :key="g.goal_id"
        class="effect-row"
      >
        <span
          class="badge"
          :data-status="g.verified ? 'completed' : 'failed'"
          >{{ g.verified ? "通过" : "未通过" }}</span
        ><strong>{{ g.object }}</strong
        ><span v-if="g.angle_error_rad !== undefined"
          >转角误差
          {{ ((g.angle_error_rad * 180) / Math.PI).toFixed(2) }}°</span
        ><span v-if="g.shade_fraction != null"
          >遮光覆盖 {{ (g.shade_fraction * 100).toFixed(1) }}%</span
        ><span v-if="g.board_return_error_m != null"
          >放回误差 {{ (g.board_return_error_m * 1000).toFixed(2) }} mm</span
        ><span>{{ g.reason }}</span>
      </div>
      <TelemetryPanel
        :samples="detail.samples"
        :detail="detail"
        history
      /><PlanPanel :detail="detail" />
    </div>
  </section>
</template>
