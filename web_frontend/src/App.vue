<script setup lang="ts">
import { RouterLink, RouterView } from "vue-router";

import { navigationItems } from "./router/routes";
</script>

<template>
  <main class="app-layout">
    <nav class="side-nav" aria-label="模块导航">
      <div class="brand-mark">STA</div>
      <RouterLink
        v-for="item in navigationItems"
        :key="item.path"
        :to="item.path"
      >
        {{ item.label }}
      </RouterLink>
    </nav>
    <RouterView />
  </main>
</template>
