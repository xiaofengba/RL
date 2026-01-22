import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pygame
import sys
import matplotlib.pyplot as plt
from torch.distributions import Categorical

# ==========================================
# 复用你的配置和 GridWorld 环境
# ==========================================
# (这里为了节省篇幅，假设 GridWorld 类和你原来的一样，直接粘贴即可)
CELL_SIZE = 100
GRID_SIZE = 5
WINDOW_SIZE = CELL_SIZE * GRID_SIZE
FPS_TRAIN = 60
FPS_TEST = 2
WHITE, BLACK, RED, GREEN, BLUE, GRAY = (255, 255, 255), (0, 0, 0), (200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 200)

class GridWorld:
    # ... (请直接复用你之前的 GridWorld 类代码，不需要改动) ...
    def __init__(self, size=GRID_SIZE):
        self.MAX_STEP = 100
        self.size = size
        self.state_dim = 13
        self.action_dim = 4
        self.goal_pos = (size-1, size-1)
        self.obstacles = [(1, 1), (2, 2), (3, 1), (1, 3)]
        self.reset()
    def reset(self):
        self.agent_pos = (0, 0); self.steps = 0; return self._get_state()
    def step(self, action):
        old_pos = self.agent_pos # 记录移动前的位置
        x, y = self.agent_pos
        if action == 0: x -= 1
        elif action == 1: x += 1
        elif action == 2: y -= 1
        elif action == 3: y += 1
        x = max(0, min(self.size-1, x)); y = max(0, min(self.size-1, y))
        next_pos = (x, y)
        reward = -0.1; done = False
        if next_pos == self.goal_pos: reward = 10; done = True
        # elif next_pos in self.obstacles: reward = -10; done = True
        # 修改后
        # [逻辑优化] 优先级判定
        elif next_pos in self.obstacles: 
            reward = -5   # [微调] -5 足够让它避开了，不需要 -50
            done = True   # 踩雷结束
        elif next_pos == old_pos: 
            # [新增] 撞墙判定！
            # 如果坐标没变，说明撞到了边界
            reward = -10.0 # 撞墙比走路痛20倍，它就不会贴墙蹭了
            # 撞墙不结束 done = False，让它学会走出来

        
        self.agent_pos = next_pos; self.steps += 1
        if self.steps > self.MAX_STEP: done = True
        return self._get_state(), reward, done
    def _get_state(self): return self.get_state_at(self.agent_pos)
    def get_state_at(self, pos):
        ax, ay = pos; gx, gy = self.goal_pos
        state = [ax/float(self.size), ay/float(self.size), gx/float(self.size), gy/float(self.size)]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = ax + dx, ay + dy
                val = 0.0
                if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size: val = 1.0
                elif (nx, ny) in self.obstacles: val = 1.0
                state.append(val)
        return np.array(state, dtype=np.float32)

# ==========================================
# [核心修改 1] Actor-Critic 网络
# ==========================================
# 你发现的这个现象，其实揭示了强化学习中 “执行（Execution）” 和 “评估（Evaluation）” 两个完全不同的阶段。
# 在走路时依靠直觉（Probs），在反思时依靠理性（Value）。
class ActorCritic(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(ActorCritic, self).__init__()
        # 公共特征提取层
        self.affine = nn.Linear(input_dim, 128)
        
        # --- Actor 头: 输出动作概率 (Policy) ---
        self.action_head = nn.Linear(128, num_actions)
        
        # --- Critic 头: 输出状态价值 V(s) ---
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine(x))
        
        # Actor 决策 输出经过 Softmax，变成概率分布, 这里的状态和为1； 当前状态下各离散动作的概率
        action_probs = F.softmax(self.action_head(x), dim=-1)
        
        # Critic 打分 输出直接是标量，无激活函数
        state_values = self.value_head(x)
        
        return action_probs, state_values

# ==========================================
# [核心修改 2] 可视化: 直接画概率，不需要Softmax了
# ==========================================
def draw_policy_overlay(screen, env, model, device):
    """绘制 Policy 概率场 (红色越粗代表该方向概率越大)"""
    for r in range(env.size):
        for c in range(env.size):
            if (r, c) in env.obstacles or (r, c) == env.goal_pos: continue
            
            fake_state = env.get_state_at((r, c))
            state_tensor = torch.FloatTensor(fake_state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                probs, _ = model(state_tensor) # 只需要 Actor 的输出
                probs = probs.cpu().numpy()[0]
            
            center_x = c * CELL_SIZE + CELL_SIZE // 2
            center_y = r * CELL_SIZE + CELL_SIZE // 2
            max_len = (CELL_SIZE // 2) * 0.8 
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)] # Up, Down, Left, Right
            
            best_action = np.argmax(probs)
            
            for action_idx, (dx, dy) in enumerate(directions):
                # 箭头长度直接对应网络输出的概率
                length = probs[action_idx] * max_len
                if length < 5: continue 
                
                end_x = int(center_x + dx * length)
                end_y = int(center_y + dy * length)
                
                # 概率最大的画红粗线，其他的画细灰线
                if action_idx == best_action:
                    color = RED
                    width = 4
                else:
                    color = (150, 150, 150)
                    width = 2
                
                pygame.draw.line(screen, color, (center_x, center_y), (end_x, end_y), width)
                pygame.draw.circle(screen, color, (end_x, end_y), 3)

# 简单的渲染辅助函数
def render_grid(screen, env, episode_num=None, reward=None):
    screen.fill(WHITE)
    for x in range(0, WINDOW_SIZE, CELL_SIZE): pygame.draw.line(screen, GRAY, (x, 0), (x, WINDOW_SIZE))
    for y in range(0, WINDOW_SIZE, CELL_SIZE): pygame.draw.line(screen, GRAY, (0, y), (WINDOW_SIZE, y))
    gx, gy = env.goal_pos
    pygame.draw.rect(screen, GREEN, (gy*CELL_SIZE, gx*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    for (ox, oy) in env.obstacles:
        pygame.draw.rect(screen, (50, 50, 50), (oy*CELL_SIZE, ox*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    ax, ay = env.agent_pos
    s = pygame.Surface((CELL_SIZE-20, CELL_SIZE-20)); s.set_alpha(128); s.fill(BLUE)
    screen.blit(s, (ay*CELL_SIZE+10, ax*CELL_SIZE+10))
    if episode_num is not None:
        font = pygame.font.SysFont(None, 24)
        img = font.render(f"Ep: {episode_num}, R: {reward:.1f} (Actor-Critic)", True, BLACK)
        screen.blit(img, (10, 10))

# ==========================================
# [核心修改 3] 主循环
# ==========================================
def main():
    LR = 0.003          # AC 通常学习率可以稍微高一点点
    GAMMA = 0.99
    NUM_EPISODES = 1500
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridWorld()
    model = ActorCritic(env.state_dim, env.action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 注意：这里移除了 ReplayBuffer，也不需要 Epsilon
    # 策略梯度自带随机性 (Stochastic Policy)

    reward_history = []
    
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Actor-Critic Training")
    clock = pygame.time.Clock()
    training_active = True

    for i_episode in range(NUM_EPISODES):
        if not training_active: break
        # 网络的输入仍然是 xy+局部8个格子状态
        state = env.reset()
        total_reward = 0
        done = False
        
        # --- AC 算法通常是单步更新 (或一个Episode更新一次) ---
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: training_active = False; done = True

            # 观察状态 ---> 大脑思考 ---> 掷骰子决定 ---> 采取行动
            # 1. 前向传播 = “大脑思考”，这里侧重概率 probs
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            # 这里的 probs 当前状态下执行动作的概率
            # 这里的 value 当前状态的价值
            probs, value = model(state_tensor)
            
            # 2. 动作选择：基于概率分布进行采样 (Sampling) = “掷骰子”
            # 这就是“策略”的核心：输出不是定值，而是分布
            m = Categorical(probs)
            action = m.sample() # 依概率随机选一个动作

            # 3. 执行动作
            next_state, reward, done = env.step(action.item())
            total_reward += reward
            
            # 4. 计算 Loss (关键步骤)
            # 这里的逻辑体现了“策略迭代”
            
            # 4.1 计算 TD Target (Critic 的目标)， 这里重点使用value
            # 评估当前状态 V(s) 应该等于 r + gamma * V(s')
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            with torch.no_grad():
                # 基于AC模型评估执行action后预期状态的价值
                _, next_value = model(next_state_tensor)
                # 它是 “执行动作后的真实反馈”。即：“做了动作后，拿到了奖励加上新局面的价值，这一趟实际值多少分？”
                target_value = reward + (1 - int(done)) * GAMMA * next_value
            
            # 4.2 计算 Advantage (优势函数)
            # 优势 = 实际发生的收益 (TD Target) - Critic 预测的平均收益 (Value)
            # 这就是告诉 Actor："你刚才那一步选得比预期的好(正)还是差(负)？"
            # 两者的差值 (td_error) 才能精确地衡量出 “这个动作到底带来了多少额外的好处”， 为后续的actor_loss作准备
            td_error = target_value - value
            
            # 4.3 Critic Loss: 也就是让 V(s) 逼近 r + gamma * V(s')
            # 相当于值迭代中的 Value Update 步骤
            critic_loss = F.mse_loss(value, target_value)
            
            # 4.4 Actor Loss: -log(prob) * advantage
            # 相当于策略迭代中的 Policy Improvement 步骤
            # 如果 advantage > 0 (比预期好)，loss 会让 log(prob) 变大 -> 增加该动作概率
            # .detach() 很重要：Critic 的更新不要通过 Advantage 传回给 Actor，各司其职
            # 看动作带来的好处正负，然后最大化这个概率
            actor_loss = -m.log_prob(action) * td_error.detach()
            
            loss = actor_loss + critic_loss
            
            # 5. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
        
        reward_history.append(total_reward)
        
        if i_episode % 10 == 0:
            render_grid(screen, env, i_episode, total_reward)
            pygame.display.flip()

    print("Training Finished.")
    
    # --- 演示 ---
    # Matplotlib 绘图部分省略，与DQN一致
    
    state = env.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            probs, _ = model(state_tensor)
            # 演示时，可以选择采样，也可以选择概率最大的 (更稳定)
            action = torch.argmax(probs).item()
            
        state, _, done = env.step(action)
        render_grid(screen, env)
        draw_policy_overlay(screen, env, model, device)
        pygame.display.flip()
        clock.tick(FPS_TEST)
        if done: pygame.time.wait(1000); state = env.reset()

if __name__ == "__main__":
    main()