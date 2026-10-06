<script setup lang="ts">
defineProps<{ name: string; size?: number }>();
const paths: Record<string, string> = {
  monitor: "M3 4h18v12H3z M8 21h8 M12 16v5",
  task: "M8 4h12v17H4V4h4 M9 2h6v4H9z M8 10h8 M8 14h6",
  history: "M3 11a9 9 0 1 1 2 7 M3 4v7h7 M12 7v5l3 2",
  logs: "M4 3h16v18H4z M8 7h8 M8 11h8 M8 15h5",
  play: "M8 4l12 8-12 8z",
  pause: "M7 4v16 M17 4v16",
  stop: "M5 5h14v14H5z",
  reset: "M4 10a8 8 0 1 1 1 7 M4 3v7h7",
  arrow: "M5 12h14 M14 7l5 5-5 5",
  robot: "M5 19h14 M7 19l2-7 5-4 M9 12l-4-4 3-4 6 4 4-3 3 4-4 4 M5 8h4",
  check: "M5 12l4 4L19 6",
  download: "M12 3v12 M7 10l5 5 5-5 M4 17v4h16v-4",
  camera: "M3 6h5l2-3h4l2 3h5v15H3z M16 13a4 4 0 1 1-8 0 4 4 0 0 1 8 0",
  chart: "M3 3v18h18 M7 15l4-6 4 3 5-7",
  bolt: "M13 2L4 14h7l-1 8 10-13h-7z",
};
</script>
<template>
  <svg
    :width="size || 20"
    :height="size || 20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.6"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <path :d="paths[name] || paths.monitor" />
  </svg>
</template>
