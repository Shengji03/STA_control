<script setup lang="ts">
import { ref } from "vue";
import { useRouter } from "vue-router";

import { dispatchTask } from "../api/http";

const router = useRouter();
const instruction = ref("遮光并旋拧1号阀门180度");
const scene = ref("scene5_glare.xml");
const mode = ref("实时仿真");
const totalTime = ref(300);
const fps = ref(20);
const recordTcp = ref(true);
const planText = ref("");
const submitting = ref(false);
const message = ref("");

async function submitTask(): Promise<void> {
  submitting.value = true;
  message.value = "";
  try {
    const plan = parsePlan();
    const task = await dispatchTask({
      instruction: instruction.value,
      scene: scene.value,
      mode: mode.value,
      total_time: totalTime.value,
      fps: fps.value,
      record_tcp: recordTcp.value,
      plan,
    });
    message.value = `${task.id} 已下发`;
    await router.push("/");
  } catch (error) {
    const apiError = error as { response?: { data?: { detail?: string } }; message?: string };
    message.value = apiError.response?.data?.detail || apiError.message || "任务下发失败";
  } finally {
    submitting.value = false;
  }
}

function parsePlan(): Record<string, unknown> | null {
  const raw = planText.value.trim();
  if (!raw) return null;
  const parsed = JSON.parse(raw) as Record<string, unknown>;
  if (!parsed || typeof parsed !== "object" || !Array.isArray(parsed.plan)) {
    throw new Error("plan JSON 必须包含 plan 数组");
  }
  return parsed;
}
</script>

<template>
  <section class="work-page">
    <header class="work-header">
      <div>
        <h1>任务下发</h1>
        <p>STA Control</p>
      </div>
      <button type="button" :disabled="submitting" @click="submitTask">
        {{ submitting ? "下发中" : "下发任务" }}
      </button>
    </header>

    <div class="work-grid">
      <section class="work-section wide">
        <h2>任务指令</h2>
        <label class="field">
          <span>任务描述</span>
          <textarea v-model="instruction" rows="8" placeholder="输入机械臂任务指令"></textarea>
        </label>
        <label class="field">
          <span>预置计划 JSON</span>
          <textarea
            v-model="planText"
            rows="8"
            placeholder='可选。留空时使用 DialogPlanner；填写时格式为 {"plan":[...]}'
          ></textarea>
        </label>
        <div class="form-row">
          <label class="field">
            <span>场景</span>
            <select v-model="scene">
              <option>scene5_glare.xml</option>
              <option>scene4_pipeline.xml</option>
              <option>scene3.xml</option>
            </select>
          </label>
          <label class="field">
            <span>执行模式</span>
            <select v-model="mode">
              <option>实时仿真</option>
              <option>仅规划</option>
            </select>
          </label>
        </div>
        <p v-if="message" class="inline-message">{{ message }}</p>
      </section>

      <section class="work-section">
        <h2>运行参数</h2>
        <label class="field">
          <span>仿真时长</span>
          <input v-model.number="totalTime" type="number" min="2" />
        </label>
        <label class="field">
          <span>画面帧率</span>
          <input v-model.number="fps" type="number" min="1" max="60" />
        </label>
        <label class="field inline">
          <input v-model="recordTcp" type="checkbox" />
          <span>记录 TCP 轨迹</span>
        </label>
      </section>
    </div>
  </section>
</template>
