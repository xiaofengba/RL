import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pygame
import sys
import matplotlib.pyplot as plt

# --- Pygame 配置 ---
CELL_SIZE = 100
GRID_SIZE = 5
WINDOW_SIZE = CELL_SIZE * GRID_SIZE
FPS_TRAIN = 60  # 训练时帧率
FPS_TEST = 2    # 演示时帧率 (慢一点方便看)

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)     # 障碍物 / 最佳策略箭头
GREEN = (50, 200, 50)   # 目标
BLUE = (50, 50, 200)    # Agent
GRAY = (200, 200, 200)  # 网格线 / 普通箭头

# ==========================================
# 1. 环境定义 (GridWorld)
# ==========================================
class GridWorld:
    def __init__(self, size=GRID_SIZE):
        self.MAX_STEP = 50                 # 机器人交互的最大步数
        self.size = size
        # 修改：输入维度变为 13 (2机器人坐标 + 2终点坐标 + 9局部环境)
        self.state_dim = 13                 
        self.action_dim = 4                 # 0:Up, 1:Down, 2:Left, 3:Right
        self.goal_pos = (size-1, size-1)    # 目标位置
        self.obstacles = [(1, 1), (2, 2), (3, 1), (1, 3)] 
        self.reset()

    # 重置机器人状态，机器人的位置
    def reset(self):
        self.agent_pos = (0, 0)     # 机器人位置
        self.steps = 0              # 记录已经行走的步数
        return self._get_state()    # 

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:   x -= 1 # Up
        elif action == 1: x += 1 # Down
        elif action == 2: y -= 1 # Left
        elif action == 3: y += 1 # Right
        
        x = max(0, min(self.size-1, x))
        y = max(0, min(self.size-1, y))
        next_pos = (x, y)
        
        reward = -0.1
        done = False
        
        if next_pos == self.goal_pos:
            reward = 10
            done = True
        elif next_pos in self.obstacles:
            reward = -10
            done = True
        
        self.agent_pos = next_pos
        self.steps += 1
        if self.steps > self.MAX_STEP: done = True
            
        return self._get_state(), reward, done

    def _get_state(self):
        return self.get_state_at(self.agent_pos)

    def get_state_at(self, pos):
        """
        修改：生成指定位置的状态向量 (坐标+局部环境)
        输入: Robot XY (2), Goal XY (2), 3x3 Map (9) = Total 13
        """
        ax, ay = pos
        gx, gy = self.goal_pos
        
        state = []
        
        # 1. 归一化的自身坐标和终点坐标 (范围 0~1)
        state.append(ax / float(self.size))
        state.append(ay / float(self.size))
        state.append(gx / float(self.size))
        state.append(gy / float(self.size))
        
        # 2. 以机器人为中心的 3x3 局部环境
        # 0为无障碍，1为障碍或边界
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = ax + dx, ay + dy
                
                val = 0.0
                # 检查边界 (墙壁视为障碍)
                if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
                    val = 1.0
                # 检查障碍物
                elif (nx, ny) in self.obstacles:
                    val = 1.0
                    
                state.append(val)
                
        return np.array(state, dtype=np.float32)

# ==========================================
# 2. 神经网络 & 3. 经验回放
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()
        # 修改：使用全连接层替代卷积层 (MLP)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x): 
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), action, reward, np.array(next_state), done
    def __len__(self): return len(self.buffer)

# ==========================================
# 4. 可视化工具函数 (箭头 & 曲线)
# ==========================================
def draw_arrow(surface, color, start, end, width=2):
    pygame.draw.line(surface, color, start, end, width)
    pygame.draw.circle(surface, color, end, width + 1)

def draw_q_overlay(screen, env, model, device):
    """绘制 Q 值箭头场 (已修复 int 报错)"""
    font = pygame.font.SysFont(None, 18)
    
    for r in range(env.size):     # Row (x)
        for c in range(env.size): # Col (y)
            if (r, c) in env.obstacles or (r, c) == env.goal_pos:
                continue
            
            fake_state = env.get_state_at((r, c))
            # 修改：Input不再是图像，而是向量
            state_tensor = torch.FloatTensor(fake_state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values = model(state_tensor)[0]
            
            probs = F.softmax(q_values, dim=0).cpu().numpy()
            best_action = torch.argmax(q_values).item()
            
            center_x = c * CELL_SIZE + CELL_SIZE // 2
            center_y = r * CELL_SIZE + CELL_SIZE // 2
            max_len = (CELL_SIZE // 2) * 0.8 
            
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)] # Up, Down, Left, Right
            
            for action_idx, (dx, dy) in enumerate(directions):
                length = probs[action_idx] * max_len
                if length < 5: continue 
                
                # --- 修复点：强制转换为 int ---
                end_x = int(center_x + dx * length)
                end_y = int(center_y + dy * length)
                
                if action_idx == best_action:
                    color = RED
                    width = 4
                else:
                    color = (100, 100, 100)
                    width = 2
                
                draw_arrow(screen, color, (center_x, center_y), (end_x, end_y), width)

def render_grid(screen, env, episode_num=None, reward=None, epsilon=None):
    screen.fill(WHITE)
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, WINDOW_SIZE))
    for y in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (WINDOW_SIZE, y))

    gx, gy = env.goal_pos
    pygame.draw.rect(screen, GREEN, (gy*CELL_SIZE, gx*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    for (ox, oy) in env.obstacles:
        pygame.draw.rect(screen, (50, 50, 50), (oy*CELL_SIZE, ox*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    ax, ay = env.agent_pos
    padding = CELL_SIZE * 0.2
    s = pygame.Surface((CELL_SIZE - 2*padding, CELL_SIZE - 2*padding))  
    s.set_alpha(128)                
    s.fill(BLUE)           
    screen.blit(s, (ay*CELL_SIZE + padding, ax*CELL_SIZE + padding))

    if episode_num is not None:
        font = pygame.font.SysFont(None, 24)
        info_text = f"Ep: {episode_num}, R: {reward:.1f}, Eps: {epsilon:.2f}"
        img = font.render(info_text, True, BLACK)
        screen.blit(img, (10, 10))

def plot_training_results(rewards, losses):
    """绘制 Matplotlib 曲线"""
    def moving_average(data, window_size=10):
        if len(data) < window_size: return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(12, 5))

    # Reward 曲线
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw')
    if len(rewards) > 10:
        plt.plot(moving_average(rewards), color='blue', label='Avg (10)')
    plt.title('Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)

    # Loss 曲线
    plt.subplot(1, 2, 2)
    plt.plot(losses, alpha=0.3, color='orange', label='Raw')
    if len(losses) > 10:
        plt.plot(moving_average(losses), color='orange', label='Avg (10)')
    plt.title('Avg Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    print(">>> Close the plot window to start the visualization demo...")
    plt.show()

# ==========================================
# 5. 主程序
# ==========================================
def main():

    BATCH_SIZE  = 64        # 每次从经验池提取的数据量，用于梯度下降    
    LR          = 0.001     # 学习率 (Learning Rate)
    GAMMA       = 0.9      # 折扣因子 (Discount Factor)：0.99表示非常重视未来长远奖励

    # 控制着探索的概率
    EPSILON_START   = 0.9   # 刚开始以 90% 的概率随机乱走 (探索)
    EPSILON_END     = 0.1  # 最终保留 5% 的概率随机走，防止陷入局部最优
    EPSILON_DECAY   = 0.999 # 衰减速率：每个Episode后 epsilon 乘以 0.995

    TARGET_UPDATE   = 10        # 每隔30个Episode，把 PolicyNet 参数复制给 TargetNet
    MEMORY_CAPACITY = 10000     # 经验回放池的大小
    NUM_EPISODES    = 200       # 总训练回合数

    # 检测是否有GPU，没有则使用CPU
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env     = GridWorld()

    # Policy Net: 实时训练的网络，负责选择动作
    policy_net = DQN(env.state_dim, env.action_dim).to(device)
    # Target Net: 目标网络，参数固定一段时间，用于计算 TD Target，增加训练稳定性
    target_net = DQN(env.state_dim, env.action_dim).to(device)

    # 初始时，将 Target Net 的参数同步为 Policy Net 的参数
    # Target Net 设为评估模式，不需要计算梯度
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # 优化器
    optimizer   = optim.Adam(policy_net.parameters(), lr=LR)
    memory      = ReplayBuffer(MEMORY_CAPACITY)
    epsilon     = EPSILON_START
    loss_fn     = nn.MSELoss()

    # 数据记录
    reward_history = []
    loss_history = []

    # --- Pygame 可视化初始化 ---
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("DQN Training Phase")
    clock = pygame.time.Clock()

    print("Training Started...")
    training_active = True

    # --- 1. 训练主循环 (按回合进行) ---
    for i_episode in range(NUM_EPISODES):
        if not training_active: break
        
        # 重置环境，获得初始状态
        state = env.reset()
        total_reward = 0
        episode_losses = []
        done = False

        # --- 回合内循环 (Steps) ---
        while not done:
            # 处理 Pygame 关闭事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT: training_active = False; done = True

            # --- 动作选择 (Epsilon-Greedy 策略, 是否探索) ---
            if random.random() > epsilon:
                # [利用]: 使用网络预测 Q 值最大的动作
                with torch.no_grad():
                    print(state)
                    # 增加 Batch 维度: (13) -> (1, 13)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    # shape(4, 1).max(1)[1] 返回最大值的索引 (即动作 ID)
                    action = policy_net(state_tensor).max(1)[1].item()
            else:
                # [探索]: 随机选择一个动作
                action = random.randint(0, env.action_dim - 1)
            
            # 与环境交互，获得：下一状态、奖励、是否结束
            next_state, reward, done = env.step(action)

            # --- 存入经验回放池 ---
            memory.push(state, action, reward, next_state, done)

            # 更新状态
            state = next_state
            total_reward += reward
            
            # 开始
            if len(memory) > BATCH_SIZE:
                # 1. 随机采样 Batch 数据 (打破时间相关性)
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                # 转换为 Tensor 并移动到设备
                states_t        = torch.FloatTensor(states).to(device)
                actions_t       = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_t       = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states_t   = torch.FloatTensor(next_states).to(device)
                dones_t         = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                # 2. 计算当前 Q 值 (Q_eval)
                # policy_net 输出所有动作的 Q 值 [Batch, 4]
                # .gather(1, actions_t) 提取实际执行动作对应的那个 Q 值
                q_values = policy_net(states_t).gather(1, actions_t)

                # 3. 计算目标 Q 值 (Q_target)
                with torch.no_grad():
                    # 使用 Target Net 计算下一状态的最大 Q 值: max Q(s', a')
                    next_q_values = target_net(next_states_t).max(1)[0].unsqueeze(1)
                    # 贝尔曼方程: y = r + gamma * max Q(s')
                    # (1 - dones_t) 确保如果是终止状态，未来奖励为 0
                    expected_q_values = rewards_t + (1 - dones_t) * GAMMA * next_q_values
                # 4. 梯度下降
                loss = loss_fn(q_values, expected_q_values)
                optimizer.zero_grad()   # 清空旧梯度
                loss.backward()         # 反向传播
                optimizer.step()        # 更新参数
                episode_losses.append(loss.item())

        # --- 回合结束处理 ---
        reward_history.append(total_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        loss_history.append(avg_loss)

        # 衰减 Epsilon (逐步减少随机探索)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        if i_episode % TARGET_UPDATE == 0: target_net.load_state_dict(policy_net.state_dict())
        
        # 训练时的可视化 (为了不拖慢训练，每10局才画一次)
        if i_episode % 10 == 0:
            render_grid(screen, env, i_episode, total_reward, epsilon)
            pygame.display.flip()

    print("Training Finished.")
    
    # --- 2. 绘制曲线 ---
    if training_active:
        # 为了避免Pygame和Matplotlib冲突，先暂停一下Pygame或者不做任何Pygame操作
        # 注意：plt.show() 会阻塞主线程
        print("Displaying results... Please close the plot window to continue.")
        plot_training_results(reward_history, loss_history)

    # --- 3. 最终可视化演示 ---
    print("Starting Q-Value Visualization Demo...")
    pygame.display.set_caption("DQN Demo (Red Arrow = Best Action)")
    state = env.reset()
    done = False
    
    while training_active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: training_active = False

        # 推理动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = policy_net(state_tensor).max(1)[1].item()
        
        state, _, done = env.step(action)
        
        # 渲染
        render_grid(screen, env) # 基础层
        draw_q_overlay(screen, env, policy_net, device) # 箭头层
        
        pygame.display.flip()
        clock.tick(FPS_TEST)
        
        if done:
            pygame.time.wait(1000)
            state = env.reset()
            done = False

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()