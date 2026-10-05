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

目录职责及剩余整理项见 [结构说明](docs/architecture.md)。

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
$env:STA_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:STA_LLM_MODEL = "qwen3.5-plus"
```

密钥读取优先级为 `STA_LLM_API_KEY`、`DASHSCOPE_API_KEY`、
`DEEPSEEK_API_KEY`、`OPENAI_API_KEY`。命令行显式参数优先于环境变量。
`.env.example` 是配置示例，项目当前不会自动加载 `.env`。

## 启动任务

在项目根目录运行：

```powershell
python run_dialog.py --task "遮光并旋拧1号阀门180度"
```

加载预置计划时无需密钥，也不会调用大模型：

```powershell
python run_dialog.py --plan "D:\plans\task.json" --no-viewer --no-trajectory
```

预置计划应包含 `plan` 数组。资源默认路径相对于项目位置解析，
与启动进程的工作目录无关。可通过 `--scene` 显式指定场景。
TCP 轨迹图保存至 `outputs/tcp_trajectory_3d.png`。

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

## 独立实验

```powershell
python -m experiments.pipeline_valve_task
python -m experiments.pipeline_glare_task
python -m experiments.long_sequence_task
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

已经存在：结构化场景状态、双臂协商规划、计划反馈、技能调度、STA、
接触技能导纳、独立 DDPG 装配实验、网页任务下发与监控。

后续需要实现：任务复杂度判别、HTN、次模优化、CARLA，以及网页中的
控制曲线、实验对比、持久化和回放。DDPG 尚未接入网页任务主流程。
重构阶段以已有任务行为为基线，算法扩展另行进行。
