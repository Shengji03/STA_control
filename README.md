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
tests/                Python 回归测试
scripts/              独立辅助工具
outputs/              生成的图片、模型权重、测试缓存（Git 忽略）
```

任务分工与辅助部署的公共入口为
[PlanningService](src/task_planning/planning_service.py)，
数学结构审查与实际场景验证位于 [experiments](experiments/)。

## 环境与依赖

本次验证环境为 Python 3.12，NumPy 1.26.4、MuJoCo 3.3.7、
Robotics Toolbox 1.1.1、SpatialMath 1.1.15。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

`requirements.txt` 声明机器人运行和实验依赖；`requirements-web.txt`
在此基础上添加网页后端；`requirements-dev.txt` 添加 Python 测试依赖。
这些文件是依赖清单，尚未作为完整锁定文件。

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

前两项启动预设任务及原生查看器。长时序实验保持原来的默认流程：
运行 STA 基线，训练 5 个 DDPG episode，再评估并绘制对比结果。
运行该实验会产生训练数据和模型权重，不应将它作为普通导入检查。

模型统一保存和加载于 `outputs/models/ddpg_model.npz`。
旧 `wrist` 包和兼容入口已经移除，实验类和函数请从 `experiments` 导入。

`motion_planning/joint_trajectory.py` 提供当前技能使用的三次关节轨迹；
`motion_planning/rrt` 保留遮光板避障实际调用的基础 RRT。
未接入任务的笛卡尔路径、混合路径、五次插值、二维 RRT 和 RRT* 变体
已删除。现存场景为装配 `scene3`、管道 `scene4_pipeline` 和遮光 `scene5_glare`。

## 验证

```powershell
python -m pytest -q
cd web_frontend
npm.cmd test
npm.cmd run build
```

测试包括现有网页和渲染流程，以及配置优先级、导航时序、预置计划、
共享控制器输出、实验指标，以及清理前记录的轨迹、碰撞和 RRT 路径。
Windows 渲染测试可能产生 GLFW 警告；本次重构前的基线也存在此现象。

## 研究功能状态

已经存在：结构化场景状态、双臂协商规划、完整任务分工与阶段级次模辅助优化、
计划反馈、实际效果校验、技能调度、STA、
接触技能导纳、独立 DDPG 装配实验、网页任务下发与监控。

当前方案由 LLM 给出目标与阶段顺序，再由完整动作模板和次模优化生成执行计划，
已放弃任务复杂度判别和 HTN。网页继续采用 Vue + FastAPI，CARLA 暂缓。
网页已实现控制曲线、记录对比、持久化和状态回放；后续研究包括
DDPG 补偿与主流程集成，以及更完整的实验评估。
目前辅助动作支持 R 臂遮光；压力表支持观测位到达，尚未实现读数识别。

控制接口已按实际关节自由度提取抓取点雅可比；夹爪合外力采用世界轴的
`[Fx, Fy, Fz, Mx, My, Mz]`，力矩与雅可比使用同一抓取点。
跟踪速度误差包含轨迹速度和导纳补偿速度，导纳按技能限制补偿位移和速度。
