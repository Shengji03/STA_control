<script setup lang="ts">
import { onMounted, ref } from "vue";

import { fetchTaskHistory } from "../api/http";
import type { TaskRecord } from "../types/simulation";

const tasks = ref<TaskRecord[]>([]);

async function loadHistory(): Promise<void> {
  tasks.value = await fetchTaskHistory();
}

onMounted(() => {
  void loadHistory();
});
</script>

<template>
  <section class="work-page">
    <header class="work-header">
      <div>
        <h1>任务历史</h1>
        <p>STA Control</p>
      </div>
      <button type="button" @click="loadHistory">刷新</button>
    </header>

    <section class="work-section">
      <h2>执行记录</h2>
      <table class="data-table">
        <thead>
          <tr>
            <th>任务编号</th>
            <th>场景</th>
            <th>状态</th>
            <th>开始时间</th>
            <th>说明</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="task in tasks" :key="task.id">
            <td>{{ task.id }}</td>
            <td>{{ task.scene }}</td>
            <td>{{ task.status }}</td>
            <td>{{ task.created_at }}</td>
            <td>{{ task.message }}</td>
          </tr>
          <tr v-if="tasks.length === 0">
            <td colspan="5">暂无任务记录</td>
          </tr>
        </tbody>
      </table>
    </section>
  </section>
</template>

