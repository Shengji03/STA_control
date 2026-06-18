<script setup lang="ts">
import { onBeforeUnmount, onMounted } from "vue";

import ControlPanel from "../components/ControlPanel.vue";
import SimulationViewport from "../components/SimulationViewport.vue";
import StatusPanel from "../components/StatusPanel.vue";
import { useSimulationStore } from "../stores/simulation";

const simulation = useSimulationStore();

onMounted(() => {
  void simulation.refreshStatus();
  simulation.connect();
});

onBeforeUnmount(() => {
  simulation.disconnect();
});
</script>

<template>
  <section class="simulation-page">
    <SimulationViewport />
    <div class="right-rail">
      <ControlPanel />
      <StatusPanel />
    </div>
  </section>
</template>

