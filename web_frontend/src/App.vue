<script setup lang="ts">
import { onMounted, onBeforeUnmount } from "vue";
import { RouterLink, RouterView } from "vue-router";
import { navigationItems } from "./router/routes";
import { useSimulationStore } from "./stores/simulation";
import AppIcon from "./components/AppIcon.vue";
const simulation = useSimulationStore();
const icons = ["monitor", "task", "history", "logs"];
onMounted(() => {
  void simulation.refreshStatus();
  simulation.connect();
});
onBeforeUnmount(() => simulation.disconnect());
</script>
<template>
  <a class="skip-link" href="#main-content">跳转到主内容</a>
  <div class="app-layout">
    <aside class="sidebar">
      <RouterLink to="/" class="brand"
        ><span class="brand-symbol"><AppIcon name="robot" :size="24" /></span
        ><span>STA<span class="brand-sub">CONTROL LAB</span></span></RouterLink
      >
      <div class="nav-caption">工作空间</div>
      <nav aria-label="模块导航">
        <RouterLink
          v-for="(item, i) in navigationItems"
          :key="item.path"
          :to="item.path"
          ><AppIcon :name="icons[i] || 'monitor'" /><span>{{ item.label }}</span
          ><span class="nav-dot"></span
        ></RouterLink>
      </nav>
      <div class="sidebar-bottom">
        <span class="live-dot" :class="{ online: simulation.isOnline }"></span
        >{{ simulation.isOnline ? "仿真服务已连接" : "正在连接服务"
        }}<small>MuJoCo · 双 UR5e</small>
      </div>
    </aside>
    <div class="workspace">
      <header class="topbar">
        <span class="breadcrumb">机器人实验室 <span>/</span> 协同作业平台</span>
        <div class="topbar-right">
          <span class="connection-chip"
            ><span
              class="live-dot"
              :class="{ online: simulation.isOnline }"
            ></span
            >{{ simulation.isOnline ? "在线" : "离线" }}</span
          ><span class="avatar">研</span>
        </div>
      </header>
      <main id="main-content">
        <div v-if="simulation.lastError" class="alert" role="alert">
          {{ simulation.lastError
          }}<button @click="simulation.lastError = ''" aria-label="关闭提示">
            ×
          </button>
        </div>
        <RouterView />
      </main>
      <footer class="workspace-footer">
        STA CONTROL <span>移动双臂机器人 · 任务规划与控制仿真</span>
      </footer>
    </div>
  </div>
</template>
