"""
DDPG补偿控制器
"""

import numpy as np
import copy
from collections import deque
import random

class Linear:
    """全连接层"""

    def __init__(self, in_features: int, out_features: int):
        # He 初始化
        scale = np.sqrt(2.0 / in_features)
        self.weight = np.random.randn(out_features, in_features).astype(np.float64) * scale
        self.bias = np.zeros(out_features, dtype=np.float64)

        # 梯度缓存
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

        # 前向传播缓存 (用于反向传播)
        self._input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x.copy()
        return x @ self.weight.T + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """反向传播, 返回对输入的梯度"""
        self.grad_weight = grad_output.T @ self._input
        self.grad_bias = np.sum(grad_output, axis=0)
        grad_input = grad_output @ self.weight
        return grad_input

    def parameters(self):
        return [(self.weight, self.grad_weight), (self.bias, self.grad_bias)]


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_backward(x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    grad = grad_output.copy()
    grad[x <= 0] = 0
    return grad


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_backward(output: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    return grad_output * (1.0 - output ** 2)


# ============================================================================
# Actor 网络
# ============================================================================

class ActorNetwork:
    """
    Actor 网络: state -> action (补偿力矩)

    结构: Input -> FC(h1) -> ReLU -> FC(h2) -> ReLU -> FC(h3) -> ReLU -> FC(out) -> tanh * bound
    """

    def __init__(self, state_dim: int, action_dim: int, action_bound: float,
                 hidden_dims: tuple = (256, 256, 128)):
        self.action_bound = action_bound

        self.fc1 = Linear(state_dim, hidden_dims[0])
        self.fc2 = Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = Linear(hidden_dims[1], hidden_dims[2])
        self.fc_out = Linear(hidden_dims[2], action_dim)

        # 输出层使用较小的初始化, 确保初始输出接近零
        self.fc_out.weight *= 0.003
        self.fc_out.bias *= 0.0

        self._layers = [self.fc1, self.fc2, self.fc3, self.fc_out]

        # 前向传播中间结果缓存
        self._cache = {}

    def forward(self, state: np.ndarray) -> np.ndarray:
        """前向传播"""
        if state.ndim == 1:
            state = state.reshape(1, -1)

        x1 = self.fc1.forward(state)
        h1 = relu(x1)

        x2 = self.fc2.forward(h1)
        h2 = relu(x2)

        x3 = self.fc3.forward(h2)
        h3 = relu(x3)

        x_out = self.fc_out.forward(h3)
        out = tanh(x_out)
        action = out * self.action_bound

        # 缓存中间结果用于反向传播
        self._cache = {
            'state': state, 'x1': x1, 'h1': h1,
            'x2': x2, 'h2': h2, 'x3': x3, 'h3': h3,
            'x_out': x_out, 'out': out
        }

        return action

    def backward(self, grad_action: np.ndarray):
        """反向传播: 从 action 的梯度计算各层参数梯度"""
        c = self._cache

        # action = tanh(x_out) * action_bound
        grad_out = grad_action * self.action_bound
        grad_x_out = tanh_backward(c['out'], grad_out)

        grad_h3 = self.fc_out.backward(grad_x_out)
        grad_x3 = relu_backward(c['x3'], grad_h3)

        grad_h2 = self.fc3.backward(grad_x3)
        grad_x2 = relu_backward(c['x2'], grad_h2)

        grad_h1 = self.fc2.backward(grad_x2)
        grad_x1 = relu_backward(c['x1'], grad_h1)

        self.fc1.backward(grad_x1)

    def get_all_parameters(self):
        params = []
        for layer in self._layers:
            params.extend(layer.parameters())
        return params

    def get_weights(self):
        """获取所有权重的副本"""
        return [(w.copy(), b.copy()) for layer in self._layers
                for w, b in [(layer.weight, layer.bias)]]

    def set_weights(self, weights):
        """设置所有权重"""
        idx = 0
        for layer in self._layers:
            layer.weight = weights[idx][0].copy()
            layer.bias = weights[idx][1].copy()
            idx += 1


# ============================================================================
# Critic 网络
# ============================================================================

class CriticNetwork:
    """
    Critic 网络: (state, action) -> Q-value

    结构: [state, action] -> FC(h1) -> ReLU -> FC(h2) -> ReLU -> FC(h3) -> ReLU -> FC(1)
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: tuple = (256, 256, 128)):
        input_dim = state_dim + action_dim

        self.fc1 = Linear(input_dim, hidden_dims[0])
        self.fc2 = Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = Linear(hidden_dims[1], hidden_dims[2])
        self.fc_out = Linear(hidden_dims[2], 1)

        self.fc_out.weight *= 0.003
        self.fc_out.bias *= 0.0

        self._layers = [self.fc1, self.fc2, self.fc3, self.fc_out]
        self._cache = {}

    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """前向传播"""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        if action.ndim == 1:
            action = action.reshape(1, -1)

        sa = np.concatenate([state, action], axis=1)

        x1 = self.fc1.forward(sa)
        h1 = relu(x1)

        x2 = self.fc2.forward(h1)
        h2 = relu(x2)

        x3 = self.fc3.forward(h2)
        h3 = relu(x3)

        q = self.fc_out.forward(h3)

        self._cache = {
            'sa': sa, 'x1': x1, 'h1': h1,
            'x2': x2, 'h2': h2, 'x3': x3, 'h3': h3,
        }
        return q

    def backward(self, grad_q: np.ndarray):
        """反向传播: 返回对 [state, action] 拼接输入的梯度"""
        c = self._cache

        grad_h3 = self.fc_out.backward(grad_q)
        grad_x3 = relu_backward(c['x3'], grad_h3)

        grad_h2 = self.fc3.backward(grad_x3)
        grad_x2 = relu_backward(c['x2'], grad_h2)

        grad_h1 = self.fc2.backward(grad_x2)
        grad_x1 = relu_backward(c['x1'], grad_h1)

        grad_sa = self.fc1.backward(grad_x1)
        return grad_sa

    def get_all_parameters(self):
        params = []
        for layer in self._layers:
            params.extend(layer.parameters())
        return params

    def get_weights(self):
        return [(w.copy(), b.copy()) for layer in self._layers
                for w, b in [(layer.weight, layer.bias)]]

    def set_weights(self, weights):
        idx = 0
        for layer in self._layers:
            layer.weight = weights[idx][0].copy()
            layer.bias = weights[idx][1].copy()
            idx += 1


# ============================================================================
# Adam 优化器
# ============================================================================

class AdamOptimizer:
    """Adam 优化器 (带梯度裁剪)"""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8,
                 max_grad_norm: float = 1.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.t = 0
        self.m = {}  # 一阶矩
        self.v = {}  # 二阶矩

    def step(self, parameters: list):
        """
        更新参数 (梯度裁剪 + Adam)

        Args:
            parameters: [(param, grad), ...] 列表
        """
        self.t += 1

        # ---- 全局梯度裁剪 (clip by global norm) ----
        total_norm = 0.0
        for _, grad in parameters:
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)

        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for _, grad in parameters:
                grad *= clip_coef

        # ---- Adam 更新 ----
        for i, (param, grad) in enumerate(parameters):
            if i not in self.m:
                self.m[i] = np.zeros_like(param)
                self.v[i] = np.zeros_like(param)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ============================================================================
# 经验回放缓冲区
# ============================================================================

class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.ndarray,
             reward: float, next_state: np.ndarray, done: bool):
        """存储一条经验"""
        self.buffer.append((
            state.copy(), action.copy(), reward,
            next_state.copy(), float(done)
        ))

    def sample(self, batch_size: int) -> tuple:
        """随机采样一个 mini-batch"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float64),
            np.array(actions, dtype=np.float64),
            np.array(rewards, dtype=np.float64).reshape(-1, 1),
            np.array(next_states, dtype=np.float64),
            np.array(dones, dtype=np.float64).reshape(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# Ornstein-Uhlenbeck 探索噪声
# ============================================================================

class OUNoise:
    """
    Ornstein-Uhlenbeck 过程噪声

    论文中用于 Actor 的探索策略:
        dx = θ (μ - x) dt + σ dW
    产生时间相关的噪声, 适合物理控制问题
    """

    def __init__(self, action_dim: int, mu: float = 0.0,
                 theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def reset(self):
        """重置噪声状态"""
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        """生成一个噪声样本"""
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state.copy()

    def set_sigma(self, sigma: float):
        """动态调整噪声强度 (用于训练后期衰减)"""
        self.sigma = sigma


# ============================================================================
# DDPG Agent (补偿控制器)
# ============================================================================

class DDPGAgent:
    """
    DDPG 补偿控制器

    与 PID 并联工作:
        u_total = u_PID + u_DDPG

    状态空间: [关节位置误差(6), 关节速度误差(6)] = 12 维
    动作空间: [各关节补偿力矩(6)] = 6 维

    参考论文 Table I 的超参数设置
    """

    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 6,
        action_bound: float = 1.0,
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.95,
        tau: float = 0.005,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        noise_sigma: float = 0.1,
        noise_theta: float = 0.15,
        warmup_steps: int = 500,
        hidden_dims: tuple = (256, 256, 128),
    ):
        """
        初始化 DDPG 补偿控制器

        Args:
            state_dim:       状态维度 (默认 12 = 6误差 + 6误差变化率)
            action_dim:      动作维度 (默认 6 = 6关节补偿力矩)
            action_bound:    补偿力矩上界 (N·m), 限制 DDPG 输出范围
            actor_lr:        Actor 学习率
            critic_lr:       Critic 学习率
            gamma:           折扣因子
            tau:             目标网络软更新系数
            buffer_capacity: 经验回放缓冲区大小
            batch_size:      训练 mini-batch 大小
            noise_sigma:     OU 噪声标准差
            noise_theta:     OU 噪声回归速率
            warmup_steps:    训练前最少需要的经验数量
            hidden_dims:     隐藏层维度 (3层)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps

        # ---------- 网络初始化 ----------
        # 主网络
        self.actor = ActorNetwork(state_dim, action_dim, action_bound, hidden_dims)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dims)

        # 目标网络 (初始权重 = 主网络权重)
        self.actor_target = ActorNetwork(state_dim, action_dim, action_bound, hidden_dims)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_dims)
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)

        # ---------- 优化器 ----------
        self.actor_optimizer = AdamOptimizer(lr=actor_lr)
        self.critic_optimizer = AdamOptimizer(lr=critic_lr)

        # ---------- 经验回放 ----------
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # ---------- 探索噪声 ----------
        self.noise = OUNoise(action_dim, theta=noise_theta, sigma=noise_sigma)

        # ---------- 状态追踪 ----------
        self.total_steps = 0
        self.training_enabled = True

        # ---------- 状态归一化 (运行均值/方差) ----------
        self._state_mean = np.zeros(state_dim)
        self._state_M2 = np.zeros(state_dim)       # Welford M2 累积量
        self._state_var = np.ones(state_dim)        # 归一化用方差 (从 M2 计算)
        self._state_count = 0

        # ---------- 训练统计 ----------
        self.critic_loss_history = []
        self.reward_history = []

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        根据当前状态选择补偿动作

        Args:
            state:     当前状态向量 [误差, 误差变化率]
            add_noise: 是否添加探索噪声 (训练时 True, 评估时 False)

        Returns:
            补偿力矩向量 (action_dim,)
        """
        # 状态归一化
        norm_state = self._normalize_state(state)

        # Actor 前向传播
        action = self.actor.forward(norm_state).flatten()

        # 添加探索噪声
        if add_noise and self.training_enabled:
            noise = self.noise.sample()
            action = action + noise

        # 裁剪到合法范围
        action = np.clip(action, -self.action_bound, self.action_bound)

        return action

    def store_transition(self, state: np.ndarray, action: np.ndarray,
                         reward: float, next_state: np.ndarray, done: bool):
        """
        存储一条状态转移经验

        Args:
            state:      当前状态
            action:     执行的动作
            reward:     获得的奖励
            next_state: 下一个状态
            done:       是否终止
        """
        # 更新状态归一化统计量
        self._update_state_stats(state)

        norm_state = self._normalize_state(state)
        norm_next_state = self._normalize_state(next_state)

        self.replay_buffer.push(norm_state, action, reward, norm_next_state, done)
        self.total_steps += 1

    def train_step(self) -> dict:
        """
        执行一步训练

        Returns:
            训练信息字典, 包含 critic_loss 等
        """
        if not self.training_enabled:
            return {}

        if len(self.replay_buffer) < self.warmup_steps:
            return {'status': 'warmup', 'buffer_size': len(self.replay_buffer)}

        # 采样 mini-batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # -------- 更新 Critic --------
        # 计算目标 Q 值: y = r + γ * Q'(s', μ'(s'))
        next_actions = self.actor_target.forward(next_states)
        target_q = self.critic_target.forward(next_states, next_actions)
        y = rewards + self.gamma * (1 - dones) * target_q

        # 裁剪 target Q 值, 防止 Q 值无限膨胀
        # 指数奖励范围 ~[-0.5, 1.5], 理论最大 Q ≈ r_max / (1 - γ)
        max_q = 2.0 / (1.0 - self.gamma + 1e-6)
        y = np.clip(y, -max_q, max_q)

        # 当前 Q 值
        current_q = self.critic.forward(states, actions)

        # Critic 损失: L = mean((y - Q(s,a))^2)
        critic_error = current_q - y
        critic_loss = np.mean(critic_error ** 2)

        # Critic 反向传播
        grad_q = 2.0 * critic_error / self.batch_size
        self.critic.backward(grad_q)
        self.critic_optimizer.step(self.critic.get_all_parameters())

        # -------- 更新 Actor --------
        # 策略梯度: ∇θ_μ J ≈ ∇_a Q(s, a)|_{a=μ(s)} · ∇θ_μ μ(s)
        pred_actions = self.actor.forward(states)
        self.critic.forward(states, pred_actions)

        # Q 对 action 的梯度
        grad_q_ones = np.ones((self.batch_size, 1)) / self.batch_size
        grad_sa = self.critic.backward(grad_q_ones)
        grad_action = grad_sa[:, self.state_dim:]  # 取 action 部分的梯度

        # 取负号: 我们要最大化 Q, 即最小化 -Q
        self.actor.backward(-grad_action)
        self.actor_optimizer.step(self.actor.get_all_parameters())

        # -------- 软更新目标网络 --------
        self._soft_update(self.actor_target, self.actor, self.tau)
        self._soft_update(self.critic_target, self.critic, self.tau)

        # 记录统计
        self.critic_loss_history.append(critic_loss)

        return {
            'critic_loss': critic_loss,
            'mean_q': np.mean(current_q),
            'mean_reward': np.mean(rewards),
        }

    def compute_reward(self, error: np.ndarray, prev_error: np.ndarray = None,
                       action: np.ndarray = None) -> float:
        """
        计算奖励信号 (改进版)

        改进点:
            1. 指数形式基础奖励 exp(-k*Σ|e_i|), 对小误差变化更敏感
               (原 1/(1+x) 在 x≈0 时几乎为常数, 无法区分好坏动作)
            2. 更强的误差减小奖励, 放大每步改善的信号
            3. 极小的动作惩罚, 不再阻碍 DDPG 输出补偿

        Args:
            error:      当前关节位置误差向量 (6,)
            prev_error: 上一时刻关节位置误差向量 (6,), 可选
            action:     当前补偿动作 (6,), 可选, 用于动作惩罚

        Returns:
            标量奖励值
        """
        abs_error_sum = np.sum(np.abs(error))

        # 基础奖励: 指数形式, 在典型误差 ~0.02 rad 时给出 ~0.55 的奖励
        # 比 1/(1+x) 形式有更大的动态范围, DDPG 更容易学到差异
        reward = np.exp(-30.0 * abs_error_sum)

        # 误差减小奖励项: 放大系数, 让 DDPG 能"感知"到每步的改善
        if prev_error is not None:
            prev_abs_sum = np.sum(np.abs(prev_error))
            error_reduction = prev_abs_sum - abs_error_sum
            reward += 10.0 * error_reduction

        # 动作代价惩罚: 大幅降低 (0.1 -> 0.01), 避免抑制 DDPG 输出
        if action is not None:
            action_cost = np.mean((action / self.action_bound) ** 2)
            reward -= 0.01 * action_cost

        return reward

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def reset_noise(self):
        """重置探索噪声 (每个 episode 开始时调用)"""
        self.noise.reset()

    def set_training(self, enabled: bool):
        """开启/关闭训练模式"""
        self.training_enabled = enabled

    def decay_noise(self, decay_rate: float = 0.995, min_sigma: float = 0.01):
        """衰减探索噪声 (随训练进行逐步减小探索)"""
        new_sigma = max(self.noise.sigma * decay_rate, min_sigma)
        self.noise.set_sigma(new_sigma)

    def get_stats(self) -> dict:
        """获取训练统计信息"""
        stats = {
            'total_steps': self.total_steps,
            'buffer_size': len(self.replay_buffer),
            'noise_sigma': self.noise.sigma,
            'training_enabled': self.training_enabled,
        }
        if self.critic_loss_history:
            stats['recent_critic_loss'] = np.mean(self.critic_loss_history[-100:])
        return stats

    def save(self, filepath: str):
        """保存模型权重"""
        data = {
            'actor_weights': self.actor.get_weights(),
            'critic_weights': self.critic.get_weights(),
            'actor_target_weights': self.actor_target.get_weights(),
            'critic_target_weights': self.critic_target.get_weights(),
            'state_mean': self._state_mean,
            'state_var': self._state_var,
            'state_M2': self._state_M2,
            'state_count': self._state_count,
            'total_steps': self.total_steps,
        }
        np.savez(filepath, **{f'arr_{i}': v for i, v in enumerate(
            [data['actor_weights'], data['critic_weights'],
             data['actor_target_weights'], data['critic_target_weights'],
             data['state_mean'], data['state_var'], data['state_M2'],
             np.array([data['state_count'], data['total_steps']])]
        )})
        print(f"DDPG 模型已保存到 {filepath}")

    def load(self, filepath: str):
        """加载模型权重"""
        data = np.load(filepath, allow_pickle=True)
        self.actor.set_weights(data['arr_0'])
        self.critic.set_weights(data['arr_1'])
        self.actor_target.set_weights(data['arr_2'])
        self.critic_target.set_weights(data['arr_3'])
        self._state_mean = data['arr_4']
        self._state_var = data['arr_5']
        self._state_M2 = data['arr_6']
        meta = data['arr_7']
        self._state_count = int(meta[0])
        self.total_steps = int(meta[1])
        print(f"DDPG 模型已从 {filepath} 加载 (已训练 {self.total_steps} 步)")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """状态归一化 (零均值, 单位方差)"""
        return (state - self._state_mean) / (np.sqrt(self._state_var) + 1e-8)

    def _update_state_stats(self, state: np.ndarray):
        """
        在线更新状态均值和方差 (Welford 算法)

        正确实现: _state_M2 只做累加, _state_var 从 M2 计算而来,
        两者分离以避免反复除法导致的数值漂移。
        """
        self._state_count += 1
        if self._state_count == 1:
            self._state_mean = state.copy()
            self._state_M2 = np.zeros_like(state)
        else:
            old_mean = self._state_mean.copy()
            self._state_mean += (state - self._state_mean) / self._state_count
            # M2 只做累加, 绝不原地除
            self._state_M2 += (state - old_mean) * (state - self._state_mean)

        # 从 M2 计算归一化用方差 (不修改 M2 本身)
        if self._state_count > 1:
            self._state_var = self._state_M2 / (self._state_count - 1)
            self._state_var = np.maximum(self._state_var, 1e-6)
        else:
            self._state_var = np.ones_like(state)

    @staticmethod
    def _soft_update(target_net, source_net, tau: float):
        """目标网络软更新: θ' ← τθ + (1-τ)θ'"""
        target_weights = target_net.get_weights()
        source_weights = source_net.get_weights()

        new_weights = []
        for (tw, tb), (sw, sb) in zip(target_weights, source_weights):
            new_w = tau * sw + (1 - tau) * tw
            new_b = tau * sb + (1 - tau) * tb
            new_weights.append((new_w, new_b))

        target_net.set_weights(new_weights)

    @staticmethod
    def _hard_update(target_net, source_net):
        """目标网络硬更新: θ' ← θ"""
        target_net.set_weights(source_net.get_weights())
