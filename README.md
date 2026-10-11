# STA Control

面向化工场景的移动双臂机器人任务规划、技能执行与控制仿真项目。
当前实现以 MuJoCo、UR5e、双臂协商规划、STA 与技能导纳控制为主，
另有 STA + DDPG 长时序装配实验和 Vue + FastAPI 仿真界面。

## 项目结构

```text
src/
  cli/                命令行入口
  config/             资源路径、机器人配置、LLM 环境配置
  perception/         场景对象登记、世界状态和相机快照
  llm_planner/        双臂协商、提示词、可执行性反馈
  task_planning/      完整任务分工、次模辅助覆盖、公共规划服务、实际效果校验
  pipeline/           计划解析、任务协调、底盘导航、单臂状态
  skills/             技能原语、技能调度、实验共享单臂控制器
  controller/         STA、导纳和 DDPG
  robot/              UR5e 运动学和碰撞几何
  motion_planning/    三次关节轨迹，以及遮光实验使用的基础 RRT
  geometry/           连杆、遮光板及凸几何碰撞计算
  utils/              数学辅助函数
  constanst/          现有数学容差常量
  visualization/     TCP 轨迹叠加及绘图
  web_backend/        FastAPI 接口、任务记录与日志
  web_sim/            仿真会话、相机控制和画面编码
  assets/             MuJoCo 场景、机器人及夹爪资源
experiments/          预设任务、训练实验、评价指标和对比报告
web_frontend/         Vue 界面
outputs/              实验结果、模型权重和网页历史数据（Git 忽略）
```

任务分工与辅助部署的公共入口为
[PlanningService](src/task_planning/planning_service.py)，
规划对比与实际场景实验位于 [experiments](experiments/)。

## 环境与依赖

本次验证环境为 Python 3.12，NumPy 1.26.4、MuJoCo 3.3.7、
Robotics Toolbox 1.1.1、SpatialMath 1.1.15。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-web.txt
```

`requirements.txt` 声明机器人运行和实验依赖；`requirements-web.txt`
在此基础上添加网页后端。
仿真与运动学依赖固定为已验证版本；其余服务依赖按清单范围安装，
这些文件尚未完整锁定所有间接依赖。

## 大模型配置

命令行与网页使用同一套环境配置：

```powershell
$env:STA_LLM_API_KEY = "你的密钥"
$env:STA_LLM_BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"
$env:STA_LLM_MODEL = "qwen3.8-flash"
```

也可以将 `.env.example` 复制为项目根目录的 `.env`，填入自己的密钥，
程序会自动读取，与启动进程的工作目录无关。
配置优先级为命令行显式参数、进程环境变量、项目 `.env`、默认值。
通过 `STA_ENV_FILE` 可以指定其他配置文件。

千问/百炼地址读取 `STA_LLM_API_KEY` 或 `DASHSCOPE_API_KEY`，不会回退使用
DeepSeek 或 OpenAI 的密钥；这些平台的密钥仅在对应服务地址下使用。
通用兼容网关仍支持原有环境变量回退，显式设置 `STA_LLM_API_KEY` 最清楚。
默认模型为 `qwen3.8-flash`，当前同步协商和 JSON 输出关闭思考模式。

## 启动任务

在项目根目录运行：

```powershell
python run_dialog.py --task "遮光并旋拧1号阀门180度"
```

加载预置计划时无需密钥，也不会调用大模型：

```powershell
python run_dialog.py --plan "D:\plans\task.json" --no-viewer --no-trajectory
```

预置计划支持语义 `goals` + `stages`，也支持旧 `plan` 数组。
完整阀门动作块可以自动参与分工优化；无法可靠识别的旧技能序列保留并校验。
资源默认路径相对于项目位置解析，
与启动进程的工作目录无关。可通过 `--scene` 显式指定场景。
TCP 轨迹图保存至 `outputs/tcp_trajectory_3d.png`。

LLM 输出目标、操作、必需遮光和导航阶段，代码展开完整技能块。
规划服务枚举完整主臂分工，并为固定分工优化非负加权辅助覆盖收益；
辅助候选不超过 12 个时穷举，较大集合预留必需辅助后使用边际收益贪心。
主任务完整性、R 臂巡检规则、资源互斥、IK 和碰撞采样是执行前的硬校验。
运行后再检查实际阀门角度、遮光期间的实际光路覆盖和遮光板放回。
评分是几何/关节运动代理收益，不能解释为成功率，未套用原论文的近似界。

仅查看规划、保存 JSON 报告，或设置辅助部署预算：

```powershell
python run_dialog.py --task "遮光并旋拧1号阀门180度" --plan-only --report outputs/plan.json
python run_dialog.py --task "旋拧1号阀门180度" --aux-budget 0 --no-viewer --no-trajectory --report outputs/task.json
```

预算不足以满足指令中的必需遮光时会报错并重新规划，不能省略遮光后执行。

## 启动网页

开发时，分别启动后端和前端：

```powershell
python run_enterprise_web.py
```

```powershell
cd web_frontend
npm.cmd ci
npm.cmd run dev
```

前端开发地址为 `http://127.0.0.1:5173`，后端为 `http://127.0.0.1:8000`。
前端构建后，后端可直接提供网页：

```powershell
cd web_frontend
npm.cmd run build
cd ..
python run_enterprise_web.py
```

浏览器打开 `http://127.0.0.1:8000`。在“任务下发”中选择遮光协作、
单阀门或双目标示例即可离线验证；选择自然语言规划时使用项目的 LLM 配置。
“仅规划”会生成可视化预览并保持待执行，在监控页点击“执行计划”才会启动机械臂。

监控页提供 MuJoCo 画面、摄像机控制、双臂规划与实际末端轨迹、
关节跟踪、末端偏差、接触力和 STA 控制力矩。规划轨迹是名义技能轨迹的
正运动学预览；控制误差按运行时参考与实测状态计算，两者含义不同。
仿真由后端单独驱动，切换页面和增加观看者不会增加物理步数。

启用记录后，任务结果保存到 `outputs/web`，可在“任务历史”中对比指标、
导出 JSON、复现计划和回放机械臂状态。回放根据保存的场景与关节状态重建画面。
网页目前支持遮光和管道场景，装配与 DDPG 实验仍通过独立入口运行。

## 独立实验

```powershell
python -m experiments.pipeline_valve_task
python -m experiments.pipeline_glare_task
python -m experiments.long_sequence_task
python -m experiments.submodular_planning
python -m experiments.submodular_planning --simulate
```

## CARLA 训练与交付

CARLA 是连续动作强化学习自动机。本项目使用 Python + MuJoCo 离线学习
STA 的 alpha、beta、lambda_s 三个正倍率，分别缩放技能六关节的基准数组。
导纳保持固定，训练和验收不使用 LLM、DDPG 或渲染器。

训练代码统一位于 `experiments/carla/`：

| 文件 | 职责 |
|---|---|
| `__main__.py` | 统一命令入口与显式安装已验证参数 |
| `training.py` | 可恢复搜索、原子写入、日志对齐、源码指纹 |
| `benchmarks.py` | 真实前序状态、技能交接、完整任务和物体效果 |
| `suite.py` | 多技能、多种子训练、完整组合选择和留出验收 |
| `refinement.py` | 从已有候选重新选择装配、重复旋拧、遮光组合参数 |
| `evaluate.py` | 运行完整任务，保存指标和运动时序，可选查看器 |
| `exports.py` | CSV、科学图、代表或全部工况时序导出 |

算法和运行时参数加载分别位于 `src/controller/carla_tuner.py`、
`control_parameters.py`；验收规则在 `control_metrics.py` 中共享。
当前唯一随代码交付的参数库是 `src/config/sta_skill_parameters.json`。
旧单技能试跑、旋拧专用训练和一次性收尾入口已删除。

### 使用已训练参数

```powershell
python -X utf8 -B run_enterprise_web.py --sta-parameters src/config/sta_skill_parameters.json
python -X utf8 -B run_dialog.py --task "遮光并旋拧1号阀门180度" --sta-parameters src/config/sta_skill_parameters.json
python -X utf8 -B -m experiments.carla evaluate --family repeat --angle-deg 120 --arm-order LR --sta-parameters src/config/sta_skill_parameters.json --output outputs/carla/manual_repeat_new
python -X utf8 -B -m experiments.carla evaluate --family assembly --assembly-order 102 --sta-parameters src/config/sta_skill_parameters.json --output outputs/carla/manual_assembly_new
```

网页和对话入口必须显式传入 `--sta-parameters`；省略时运行固定 STA 基线。
评价支持八类任务、`--arm-order LL/LR/RL/RR`、装配顺序 `012/102/021`；
`--viewer` 打开 MuJoCo 查看器。网页支持移动双臂场景，装配仍使用独立入口。
参数按场景、对象、动作、任务组合、执行阶段、角度、时长、停靠范围匹配，
超出已验证范围使用固定参数。运行记录保存实际参数库和执行器力矩。

### 新训练、恢复与组合重新筛选

```powershell
python -X utf8 -B -u -m experiments.carla train --iterations 200 --combo-iterations 120 --seeds 7 19 43 --workers 3 --case-workers 3 --case-set study_v2 --output outputs/carla/study_v2
# 仅在相同代码、预算、参数基线、工况标签和输出目录下恢复被中断的实验
python -X utf8 -B -u -m experiments.carla train --iterations 200 --combo-iterations 120 --seeds 7 19 43 --workers 3 --case-workers 3 --case-set study_v2 --output outputs/carla/study_v2 --resume
# 已有完整训练结果时，重新选择三类组合；生成候选库，不自动修改运行库
python -X utf8 -B -u -m experiments.carla refine --previous outputs/carla/skills_suite_20261009 --case-set refinement_v2 --output outputs/carla/refinement_v2 --workers 3
```

15 个配置覆盖装配左臂六类技能（GRASP 分抓取/释放），以及移动平台左右臂的
MOVE、ROTATE、GRASP 抓取/释放。`--profiles` 可指定子集，`--families` 指定组合。
导航用固定位置伺服，不参加机械臂 STA 三参数学习。

每技能配置从 5 个真实前序状态训练当前技能及后续 2 秒交接，再用 8 个完整任务
选参、16 个独立留出任务验收。技能选择目标还包含 0.25 倍完整任务代价。
组合候选每次执行完整任务，检查实际物体效果、释放、撤离、返回和切换表现。
选参先冻结，再评估独立留出工况，成功配对用于计算改善率，失败数另行记录。

`--case-set` 登记一套工况标签；默认采用输出目录名。独立新实验应使用新标签，
不要把已用于诊断或选参的留出数据再次称为独立验证。断点保存 PDF、RNG、
历史代价、代码指纹和候选日志；不匹配时拒绝恢复。源码整理前的旧实验不能
直接 `--resume`。`--smoke` 仅进行 4 工况流程检查，其参数不能安装为运行库。

CARLA 使用 401 点固定网格近似连续密度，以历史代价中位数计算奖励并更新
高斯邻域；暖启动混合均匀探索和已验证参数邻域，是明确记录的工程变体，
不等同于论文的自适应等概率网格。每个进程独占 MuJoCo 状态，
`workers × case-workers` 是工况并发上限，内存较少时降低为 1。

组合验收要求：全部留出任务成功、至少 8 个成功配对、平均综合代价降低 ≥2%、
工况 bootstrap 95% 区间下界 >0、最差配对工况退化 ≤10%。代价包括跟踪误差、
接触力、实际力矩、连续物理步力矩变化、实际目标误差和切换力矩跳变。
移动场景还要求阀门角度误差 ≤5°；网页默认效果容差更宽，不能替代训练验收。
工况置信区间不构成稳定性或全局最优证明。

### 记录导出与参数安装

```powershell
python -X utf8 -B -m experiments.carla export outputs/carla/rejected_refinement_20261010 --output outputs/carla/export_copy
python -X utf8 -B -u -m experiments.carla export outputs/carla/rejected_refinement_20261010 --output outputs/carla/export_copy --traces --trace-limit 1 --workers 3
# 完整训练/重新筛选的 best_parameters.json 通过验收后，再显式安装
python -X utf8 -B -m experiments.carla install outputs/carla/refinement_v2/best_parameters.json
```

`--output` 让导出文件与原实验分开，保留原记录和哈希。时序重跑须与原计分
结果一致；旧物理配置的时序以已保存的原始记录为准。省略 `--trace-limit` 导出
所有工况时序；100 Hz NPZ 保存期望/实际关节、实际执行器力矩和世界轴外力。
完整任务控制指标按 500 Hz 物理步采样。安装前检查控制版本、验收证据和
基准指纹，通过后写入指定参数文件，不再自动生成备份目录。
网页服务需重新启动才能加载新参数。

### 已完成结果与数据位置

首次全技能实验 `outputs/carla/skills_suite_20261009` 已完成 15 配置、8 组合、
3 种子，共 59400 次候选工况评价，2592 条留出 CSV、256 份完整任务 NPZ。
单技能库首次 5 项通过，另外五类完整组合在各自 16 工况集通过验收：
valve_L、valve_R、two_valves、inspect、rotate_inspect。

`outputs/carla/rejected_refinement_20261010` 进一步修正三类执行/组合问题，
从已有训练候选用每类 8 个完整任务选参，全部冻结后各用 32 个新工况验收。

| 组合 | 基线/学习成功 | 平均综合代价降低 | 工况 bootstrap 95% 区间 |
|---|---|---:|---|
| 装配 | 32/32、32/32 | 20.17% | 18.69%–21.65% |
| 连续旋拧 | 32/32、32/32 | 3.70% | 3.43%–3.96% |
| 遮光协作 | 32/32、32/32 | 3.50% | 3.08%–3.93% |

收益相对于修正后的固定参数基线，执行修复收益未计入 CARLA。全部配对工况
均改善，但接触力 RMS 略升，不能理解为每项指标或运行时间均降低。

装配仅把插销 A 闭合设置校准为 0.575，物块/插销 B 保持 0.6；重复旋拧按
手轮当前角度对齐四辐夹取姿态；遮光右臂准备/持板保持基线，左臂旋拧与
右臂收尾分别加载参数，阶段为 `assist_setup/main/assist_cleanup`。
装配圆柱/凹槽没有螺纹机构，ROTATE 验证腕部轨迹并记录物体偏航；
真实物体旋拧通过阀门角度验证。巡检当前验证观测位到达，尚无读数识别。

后续三类修正记录含 192 条 CSV、6 份代表时序、源码/产物 manifest。
另外五类运行 profiles 保持不变，当前八类组合分别在登记的 16 或 32 工况集中
通过验收，部分技能仍用基线。数据还包括历史旋拧扩大实验
`outputs/carla/rotation_expanded_20261009`。

测试目录、迁移验证脚本、旧源码备份、诊断中间结果和控制台重复日志已清理。
保留正式训练的候选日志、断点、选参、留出评价、CSV、图表和时序。
代码路径改变后不要修改旧协议哈希冒充原版本。`outputs/` 被 Git 忽略，交付或移机需要单独
备份实验数据；GitHub 上的代码和参数库足以运行，不包含完整训练原始记录。

机械臂关节现在使用直接力矩输入、动力学偏置补偿和固定力矩变化率限制。
运行记录区分控制指令和实际执行器力矩，旧网页记录保留原始输出含义。
旧位置伺服结构下的 DDPG 权重不能作为当前力矩控制结构的已验证模型。

前两项启动预设任务及原生查看器。长时序实验保持原来的默认流程：
运行 STA 基线，训练 5 个 DDPG episode，再评估并绘制对比结果。
运行该实验会产生训练数据和模型权重，不应将它作为普通导入检查。

模型统一保存和加载于 `outputs/models/ddpg_model.npz`。
旧 `wrist` 包和兼容入口已经移除，实验类和函数请从 `experiments` 导入。

`motion_planning/joint_trajectory.py` 提供当前技能使用的三次关节轨迹；
`motion_planning/rrt` 保留遮光板避障实际调用的基础 RRT。
未接入任务的笛卡尔路径、混合路径、五次插值、二维 RRT 和 RRT* 变体
已删除。现存场景为装配 `scene3`、管道 `scene4_pipeline` 和遮光 `scene5_glare`。

## 前端构建检查

```powershell
cd web_frontend
npm.cmd run build
```

自动测试文件及其专用依赖已移除。运行检查可通过网页中的离线示例任务，
以及上述独立实验入口进行。

## 研究功能状态

已经存在：结构化场景状态、双臂协商规划、完整任务分工与阶段级次模辅助优化、
计划反馈、实际效果校验、技能调度、STA、
接触技能导纳、独立 DDPG 装配实验、网页任务下发与监控。

当前方案由 LLM 给出目标与阶段顺序，再由完整动作模板和次模优化生成执行计划，
已放弃任务复杂度判别和 HTN。网页继续采用 Vue + FastAPI，CARLA 已接入
左右臂技能的离线三参数训练、完整组合验收及按场景/任务/阶段加载参数。
15 个技能配置已完成训练，其中 5 个通过单独技能库验收；组合选参覆盖 8 类任务，
完成上述失败修正后均在各自的独立工况集通过验收。部分技能在组合中仍使用固定基准，
不能将组合改善等同于每个单独技能均已改善。
网页已实现控制曲线、记录对比、持久化和状态回放；后续研究包括
DDPG 补偿与主流程集成，以及更完整的实验评估。
目前辅助动作支持 R 臂遮光；压力表支持观测位到达，尚未实现读数识别。

控制接口已按实际关节自由度提取抓取点雅可比；夹爪合外力采用世界轴的
`[Fx, Fy, Fz, Mx, My, Mz]`，力矩与雅可比使用同一抓取点。
跟踪速度误差包含轨迹速度和导纳补偿速度，导纳按技能限制补偿位移和速度。
