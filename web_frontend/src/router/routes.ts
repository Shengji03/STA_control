import type { RouteRecordRaw } from "vue-router";

import SimulationMonitorView from "../views/SimulationMonitorView.vue";
import SystemLogsView from "../views/SystemLogsView.vue";
import TaskDispatchView from "../views/TaskDispatchView.vue";
import TaskHistoryView from "../views/TaskHistoryView.vue";

export interface NavigationItem {
  label: string;
  path: string;
}

export const navigationItems: NavigationItem[] = [
  { label: "仿真监控", path: "/" },
  { label: "任务下发", path: "/tasks/dispatch" },
  { label: "任务历史", path: "/tasks/history" },
  { label: "系统日志", path: "/logs" },
];

export const routes: RouteRecordRaw[] = [
  {
    path: "/",
    name: "simulation-monitor",
    component: SimulationMonitorView,
  },
  {
    path: "/tasks/dispatch",
    name: "task-dispatch",
    component: TaskDispatchView,
  },
  {
    path: "/tasks/history",
    name: "task-history",
    component: TaskHistoryView,
  },
  {
    path: "/logs",
    name: "system-logs",
    component: SystemLogsView,
  },
];

