<script setup lang="ts">
import { computed } from "vue";
const props = defineProps<{
  series: { name: string; color: string; values: number[]; dashed?: boolean }[];
  times: number[];
  unit: string;
}>();
const bounds = computed(() => {
  const all = props.series.flatMap((s) => s.values).filter(Number.isFinite);
  let min = 0,
    max = 0;
  for (const v of all) {
    min = Math.min(min, v);
    max = Math.max(max, v);
  }
  if (max - min < 1e-7) {
    min -= 0.5;
    max += 0.5;
  }
  const pad = (max - min) * 0.1;
  return { min: min - pad, max: max + pad };
});
const paths = computed(() => {
  const start = props.times[0] ?? 0;
  const span = Math.max((props.times.at(-1) ?? start) - start, 1e-7);
  return props.series.map((s) => ({
    ...s,
    path: s.values
      .map(
        (v, i) =>
          `${i ? "L" : "M"}${(48 + (((props.times[i] ?? start) - start) / span) * 630).toFixed(2)},${(170 - ((v - bounds.value.min) / (bounds.value.max - bounds.value.min)) * 145).toFixed(2)}`,
      )
      .join(" "),
  }));
});
</script>
<template>
  <div class="line-chart">
    <div class="chart-legend">
      <span v-for="s in series" :key="s.name"
        ><i :style="{ borderColor: s.color }" :class="{ dashed: s.dashed }"></i
        >{{ s.name }}</span
      ><small>{{ unit }}</small>
    </div>
    <svg
      v-if="times.length"
      viewBox="0 0 700 200"
      role="img"
      :aria-label="
        series.map((s) => s.name).join('与') + '时间曲线，单位' + unit
      "
    >
      <g v-for="n in 4" :key="n">
        <line
          x1="48"
          x2="678"
          :y1="25 + (n - 1) * 48"
          :y2="25 + (n - 1) * 48"
          stroke="#e5eaf0"
        />
        <text x="40" :y="29 + (n - 1) * 48" text-anchor="end">
          {{
            (bounds.max - ((n - 1) / 3) * (bounds.max - bounds.min)).toFixed(2)
          }}
        </text>
      </g>
      <path
        v-for="s in paths"
        :key="s.name"
        :d="s.path"
        fill="none"
        :stroke="s.color"
        stroke-width="1.8"
        :stroke-dasharray="s.dashed ? '6 4' : undefined"
      />
      <text x="48" y="193">{{ times[0]?.toFixed(1) }} s</text>
      <text x="678" y="193" text-anchor="end">
        {{ times.at(-1)?.toFixed(1) }} s
      </text>
    </svg>
    <div v-else class="empty-chart">等待运行数据</div>
  </div>
  <div v-if="times.length" class="current-values">
    <span v-for="s in series" :key="s.name"
      >{{ s.name }} {{ s.values.at(-1)?.toFixed(3) }} {{ unit }}</span
    >
  </div>
</template>
