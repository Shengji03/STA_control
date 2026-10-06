<script setup lang="ts">
import { onMounted, ref, computed } from "vue";
import { fetchSystemLogs, errorMessage } from "../api/http";
import type { LogRecord } from "../types/simulation";
import AppIcon from "../components/AppIcon.vue";
const logs = ref<LogRecord[]>([]),
  filter = ref("ALL"),
  message = ref("");
const visible = computed(() =>
  filter.value === "ALL"
    ? logs.value
    : logs.value.filter((l) => l.level === filter.value),
);
async function load() {
  try {
    logs.value = await fetchSystemLogs();
    message.value = "";
  } catch (e) {
    message.value = errorMessage(e);
  }
}
onMounted(load);
</script>
<template>
  <section class="page">
    <div class="page-heading">
      <div>
        <div class="eyebrow">SYSTEM EVENTS</div>
        <h1>系统日志</h1>
        <p>查看规划校验、任务执行与服务连接事件</p>
      </div>
      <button @click="load"><AppIcon name="reset" />刷新日志</button>
    </div>
    <div v-if="message" class="alert" role="alert">{{ message }}</div>
    <section class="panel">
      <div class="panel-heading">
        <h2>运行事件</h2>
        <div class="segmented">
          <button
            v-for="level in ['ALL', 'INFO', 'WARN', 'ERROR']"
            :key="level"
            :class="{ selected: filter === level }"
            @click="filter = level"
          >
            {{ level === "ALL" ? "全部" : level }}
          </button>
        </div>
      </div>
      <div class="log-list">
        <div v-for="log in visible" :key="log.id" class="log-row">
          <span
            class="badge"
            :data-status="
              log.level === 'ERROR'
                ? 'failed'
                : log.level === 'WARN'
                  ? 'planning'
                  : 'completed'
            "
            >{{ log.level }}</span
          ><time class="mono">{{
            new Date(log.timestamp).toLocaleTimeString("zh-CN", {
              hour12: false,
            })
          }}</time>
          <p>{{ log.message }}</p>
        </div>
        <div v-if="!visible.length" class="empty-state">
          <p>当前没有该类型的日志</p>
        </div>
      </div>
    </section>
  </section>
</template>
