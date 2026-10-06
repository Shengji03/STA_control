<script setup lang="ts">
import { ref, onMounted } from "vue";
import { useRouter } from "vue-router";
import { dispatchTask, fetchScenarios, errorMessage } from "../api/http";
import type { Scenario } from "../types/simulation";
import { useSimulationStore } from "../stores/simulation";
import AppIcon from "../components/AppIcon.vue";
const router = useRouter(),
  simulation = useSimulationStore();
const presets = ref<Scenario[]>([]),
  selected = ref("shade"),
  source = ref("preset"),
  instruction = ref("遮光并旋拧1号阀门180度"),
  scene = ref("scene5_glare.xml"),
  mode = ref("实时仿真"),
  totalTime = ref(120),
  fps = ref(20),
  recordTcp = ref(true),
  planText = ref(""),
  submitting = ref(false),
  message = ref("");
onMounted(async () => {
  try {
    presets.value = await fetchScenarios();
  } catch (e) {
    message.value = errorMessage(e);
  }
});
function choose(p: Scenario) {
  selected.value = p.id;
  instruction.value = p.instruction;
  scene.value = p.scene;
  source.value = "preset";
  planText.value = "";
  message.value = "";
}
async function submit() {
  submitting.value = true;
  message.value = "";
  try {
    let plan: Record<string, unknown> | null = null;
    if (planText.value.trim()) {
      const parsed = JSON.parse(planText.value);
      if (
        !parsed ||
        typeof parsed !== "object" ||
        !(
          Array.isArray(parsed.plan) ||
          (Array.isArray(parsed.goals) && Array.isArray(parsed.stages))
        )
      )
        throw new Error("预置 JSON 需要包含 plan 或 goals/stages 数组");
      plan = parsed;
    } else if (source.value === "preset") {
      const preset = presets.value.find((p) => p.id === selected.value);
      if (!preset) throw new Error("示例尚未加载");
      plan = preset.plan;
    }
    await dispatchTask({
      instruction: instruction.value,
      scene: scene.value,
      mode: mode.value,
      total_time: totalTime.value,
      fps: fps.value,
      record_tcp: recordTcp.value,
      plan,
    });
    await simulation.refreshStatus();
    await router.push("/");
  } catch (e) {
    message.value =
      e instanceof SyntaxError
        ? "计划 JSON 格式错误，请检查引号、逗号与括号。"
        : errorMessage(e);
  } finally {
    submitting.value = false;
  }
}
</script>
<template>
  <section class="page">
    <div class="page-heading">
      <div>
        <div class="eyebrow">TASK PLANNING</div>
        <h1>任务下发</h1>
        <p>从目标指令到双臂协同计划，在同一工作空间完成</p>
      </div>
      <span
        class="badge"
        :data-status="simulation.status.active_task ? 'planning' : 'completed'"
        >{{ simulation.status.active_task ? "机器人忙碌" : "可接受任务" }}</span
      >
    </div>
    <div class="preset-grid">
      <button
        v-for="p in presets"
        :key="p.id"
        class="preset-card"
        :class="{ selected: selected === p.id && source === 'preset' }"
        @click="choose(p)"
      >
        <span class="preset-icon"
          ><AppIcon
            :name="
              p.id === 'shade' ? 'bolt' : p.id === 'valve' ? 'robot' : 'task'
            "
            :size="25" /></span
        ><strong>{{ p.name }}</strong>
        <p>{{ p.description }}</p>
        <span class="preset-action"
          >载入示例 <AppIcon name="arrow" :size="16"
        /></span>
      </button>
    </div>
    <form class="dispatch-grid" @submit.prevent="submit">
      <section class="panel form-panel">
        <div class="panel-heading">
          <div>
            <AppIcon name="task" />
            <h2>定义作业目标</h2>
          </div>
          <span class="subtle-tag">01</span>
        </div>
        <div class="form-body">
          <div class="source-choice">
            <label
              ><input
                type="radio"
                v-model="source"
                value="preset"
              />预置示例<span>离线验证 · 无需模型接口</span></label
            ><label
              ><input
                type="radio"
                v-model="source"
                value="llm"
                @change="planText = ''"
              />自然语言规划<span>LLM 理解 · 次模分工优化</span></label
            >
          </div>
          <label class="field" for="instruction"
            >任务描述<textarea
              id="instruction"
              v-model="instruction"
              rows="4"
              :readonly="source === 'preset' && !planText"
              placeholder="例如：遮光并旋拧1号阀门180度"
              required
            ></textarea
            ><small>{{
              planText.trim()
                ? "使用高级输入中的预置计划，经过完整可行性校验，不调用模型。"
                : source === "preset"
                  ? "使用上方示例的固定语义目标，仍经过完整可行性校验。"
                  : "当前支持阀门旋拧、压力表观测位到达和 R 臂遮光。"
            }}</small></label
          >
          <div class="form-row">
            <label class="field" for="scene"
              >作业场景<select
                id="scene"
                v-model="scene"
                :disabled="source === 'preset' && !planText.trim()"
              >
                <option value="scene5_glare.xml">化工遮光场景</option>
                <option value="scene4_pipeline.xml">管道双阀门场景</option>
              </select></label
            ><label class="field" for="mode"
              >执行方式<select id="mode" v-model="mode">
                <option>实时仿真</option>
                <option>仅规划</option>
              </select></label
            >
          </div>
          <details class="advanced">
            <summary>高级：载入计划 JSON</summary>
            <label class="field" for="json"
              >预置计划 JSON<textarea
                id="json"
                v-model="planText"
                rows="6"
                class="mono"
                placeholder='{"goals":[...],"stages":[...]}'
              ></textarea>
            </label>
          </details>
        </div>
      </section>
      <section class="panel form-panel">
        <div class="panel-heading">
          <div>
            <AppIcon name="monitor" />
            <h2>运行设置</h2>
          </div>
          <span class="subtle-tag">02</span>
        </div>
        <div class="form-body">
          <label class="field" for="duration"
            >仿真超时上限 / s<input
              id="duration"
              v-model.number="totalTime"
              type="number"
              min="2"
              max="1800"
              required
            /><small>超时上限，不代表任务固定执行时长。</small></label
          ><label class="field" for="fps"
            >画面更新频率<select id="fps" v-model.number="fps">
              <option :value="10">10 FPS · 低负载</option>
              <option :value="20">20 FPS · 推荐</option>
              <option :value="30">30 FPS · 高帧率</option>
            </select></label
          ><label class="record-option"
            ><input type="checkbox" v-model="recordTcp" /><span
              >记录轨迹与控制数据<small
                >保存关节、末端、接触力和控制力矩，支持回放与导出。</small
              ></span
            ></label
          >
          <div class="notice">
            <strong>执行过程</strong>
            <p>目标解析 → 分工优化 → 可行性校验 → 仿真执行 → 效果验证</p>
          </div>
          <div v-if="message" class="alert" role="alert">{{ message }}</div>
          <button
            class="primary submit-button"
            type="submit"
            :disabled="
              submitting ||
              !!simulation.status.active_task ||
              !simulation.isOnline
            "
          >
            <AppIcon :name="mode === '仅规划' ? 'task' : 'play'" />{{
              submitting
                ? "正在下发…"
                : mode === "仅规划"
                  ? "生成并预览计划"
                  : "下发并执行"
            }}<AppIcon name="arrow" :size="17" />
          </button>
          <p class="fine-print">
            {{
              mode === "仅规划"
                ? "生成后保持待执行，可在监控页面手动开始。"
                : "任务在后端独立运行，切换页面不会影响执行。"
            }}
          </p>
        </div>
      </section>
    </form>
  </section>
</template>
