<script setup lang="ts">
import { ref, watch, onMounted, onBeforeUnmount } from "vue";
const props = defineProps<{ planned: number[][]; actual: number[][] }>();
const canvas = ref<HTMLCanvasElement | null>(null);
let yaw = -0.65,
  pitch = 0.55,
  drag: { x: number; y: number } | null = null,
  observer: ResizeObserver | null = null;
function draw() {
  const el = canvas.value;
  if (!el) return;
  const w = el.clientWidth,
    h = el.clientHeight,
    dpr = window.devicePixelRatio || 1;
  el.width = w * dpr;
  el.height = h * dpr;
  const ctx = el.getContext("2d");
  if (!ctx) return;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const all = [...props.planned, ...props.actual];
  if (!all.length) return;
  const limits = [0, 1, 2].map((i) => [
    Math.min(...all.map((p) => p[i]!)),
    Math.max(...all.map((p) => p[i]!)),
  ]);
  const center = limits.map((v) => (v[0]! + v[1]!) / 2),
    span = Math.max(0.3, ...limits.map((v) => v[1]! - v[0]!));
  const scale = Math.min(w * 0.68, h * 0.75) / span;
  function project(p: number[]) {
    const x = p[0]! - center[0]!,
      y = p[1]! - center[1]!,
      z = p[2]! - center[2]!;
    const xx = x * Math.cos(yaw) - y * Math.sin(yaw),
      yy = x * Math.sin(yaw) + y * Math.cos(yaw);
    return [
      w / 2 + xx * scale,
      h * 0.53 + (yy * Math.sin(pitch) - z * Math.cos(pitch)) * scale,
    ];
  }
  function line(points: number[][], color: string, dashed = false) {
    ctx!.beginPath();
    points.forEach((p, i) => {
      const [x, y] = project(p);
      i ? ctx!.lineTo(x!, y!) : ctx!.moveTo(x!, y!);
    });
    ctx!.strokeStyle = color;
    ctx!.lineWidth = 2;
    ctx!.setLineDash(dashed ? [6, 5] : []);
    ctx!.stroke();
  }
  for (let n = 0; n <= 6; n++) {
    const a = limits[0]![0]! + (n / 6) * span,
      b = limits[1]![0]! + (n / 6) * span;
    line(
      [
        [a, limits[1]![0]!, limits[2]![0]!],
        [a, limits[1]![0]! + span, limits[2]![0]!],
      ],
      "#e4eaf2",
    );
    line(
      [
        [limits[0]![0]!, b, limits[2]![0]!],
        [limits[0]![0]! + span, b, limits[2]![0]!],
      ],
      "#e4eaf2",
    );
  }
  const origin = limits.map((v) => v[0]!);
  for (const [i, name] of ["X", "Y", "Z"].entries()) {
    const end = [...origin];
    end[i]! += span * 0.4;
    line([origin, end], "#718198");
    const xy = project(end);
    ctx.fillStyle = "#586a80";
    ctx.font = "11px sans-serif";
    ctx.fillText(name + " / m", xy[0]! + 4, xy[1]! - 3);
  }
  line(props.planned, "#3276eb", true);
  line(props.actual, "#089b88");
  ctx.setLineDash([]);
  if (props.actual.length) {
    const [x, y] = project(props.actual.at(-1)!);
    ctx.beginPath();
    ctx.arc(x!, y!, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#089b88";
    ctx.fill();
  }
}
function down(e: PointerEvent) {
  drag = { x: e.clientX, y: e.clientY };
  (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
}
function move(e: PointerEvent) {
  if (!drag) return;
  yaw += (e.clientX - drag.x) * 0.009;
  pitch = Math.max(-1.2, Math.min(1.2, pitch + (e.clientY - drag.y) * 0.008));
  drag = { x: e.clientX, y: e.clientY };
  draw();
}
function up() {
  drag = null;
}
function rotate(delta: number) {
  yaw += delta;
  draw();
}
function reset() {
  yaw = -0.65;
  pitch = 0.55;
  draw();
}
onMounted(() => {
  observer = new ResizeObserver(draw);
  if (canvas.value) observer.observe(canvas.value);
  draw();
});
onBeforeUnmount(() => observer?.disconnect());
watch(() => [props.planned, props.actual], draw);
</script>
<template>
  <div class="trajectory-chart">
    <div class="chart-legend">
      <span><i class="dashed" style="border-color: #3276eb"></i>规划轨迹</span
      ><span><i style="border-color: #089b88"></i>实际轨迹</span
      ><small>抓取点 · 世界坐标</small>
    </div>
    <canvas
      ref="canvas"
      role="img"
      aria-label="规划与实际末端三维轨迹，可拖动旋转"
      @pointerdown="down"
      @pointermove="move"
      @pointerup="up"
      @pointercancel="up"
    ></canvas>
    <div v-if="!planned.length && !actual.length" class="trajectory-empty">
      下发任务后生成三维轨迹
    </div>
    <div class="plot-tools">
      <button @click="rotate(-0.3)" aria-label="向左旋转轨迹图">↶</button
      ><button @click="rotate(0.3)" aria-label="向右旋转轨迹图">↷</button
      ><button @click="reset">复位视角</button>
    </div>
  </div>
</template>
