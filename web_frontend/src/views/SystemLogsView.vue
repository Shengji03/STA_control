<script setup lang="ts">
import { onMounted, ref } from "vue";

import { fetchSystemLogs } from "../api/http";
import type { LogRecord } from "../types/simulation";

const logs = ref<LogRecord[]>([]);

async function loadLogs(): Promise<void> {
  logs.value = await fetchSystemLogs();
}

onMounted(() => {
  void loadLogs();
});
</script>

<template>
  <section class="work-page">
    <header class="work-header">
      <div>
        <h1>系统日志</h1>
        <p>STA Control</p>
      </div>
      <button type="button" @click="loadLogs">刷新</button>
    </header>

    <section class="work-section">
      <h2>运行日志</h2>
      <div class="log-list">
        <div v-for="entry in logs" :key="entry.id" class="log-row">
          <span>{{ entry.level }}</span>
          <time>{{ entry.timestamp }}</time>
          <p>{{ entry.message }}</p>
        </div>
        <div v-if="logs.length === 0" class="log-row">
          <span>INFO</span>
          <time>--</time>
          <p>暂无日志</p>
        </div>
      </div>
    </section>
  </section>
</template>
