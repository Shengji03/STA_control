<script setup lang="ts">
import { computed } from "vue";

import { useSimulationStore } from "../stores/simulation";

const simulation = useSimulationStore();

interface CameraButton {
  label: string;
  action?: "reset";
  camera?: string;
}

const cameraButtons = computed<CameraButton[]>(() => [
  { label: "自由", action: "reset" },
  ...simulation.status.available_cameras.map((camera) => ({ label: camera, camera })),
]);

function setCamera(item: CameraButton): void {
  if (item.action === "reset") {
    void simulation.sendCamera({ action: "reset" });
    return;
  }
  if (item.camera) {
    void simulation.sendCamera({ action: "set_fixed", camera: item.camera });
  }
}
</script>

<template>
  <aside class="control-column" aria-label="仿真控制">
    <section class="panel-block">
      <h2>运行控制</h2>
      <div class="button-grid">
        <button type="button" @click="simulation.sendControl('pause')">|| 暂停</button>
        <button type="button" @click="simulation.sendControl('resume')">▶ 继续</button>
        <button type="button" @click="simulation.sendControl('reset')">↻ 重置</button>
      </div>
    </section>

    <section class="panel-block">
      <h2>相机</h2>
      <div class="camera-grid">
        <button
          v-for="item in cameraButtons"
          :key="item.camera || item.action"
          type="button"
          :class="{ active: simulation.cameraName === (item.camera || 'free') }"
          @click="setCamera(item)"
        >
          {{ item.label }}
        </button>
      </div>
    </section>
  </aside>
</template>
