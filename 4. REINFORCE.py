import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pygame
import sys
from torch.distributions import Categorical

# ==========================================
# 配置参数
# ==========================================
CELL_SIZE = 100
GRID_SIZE = 5
WINDOW_SIZE = CELL_SIZE * GRID_SIZE
FPS_TRAIN = 300 # REINFORCE 收敛慢，建议把训练速度调快
FPS_TEST = 2
WHITE, BLACK, RED, GREEN, BLUE, GRAY = (255, 255, 255), (0, 0, 0), (200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 200)

# ==========================================
# 1. 环境 (复用之前的逻辑)
# ==========================================
class GridWorld:
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
        old_pos = self.agent_pos
        x, y = self.agent_pos
        if action == 0: x -= 1
        elif action == 1: x += 1
        elif action == 2: y -= 1
        elif action == 3: y += 1
        x = max(0, min(self.size-1, x)); y = max(0, min(self.size-1, y))
        next_pos = (x, y)
        
        done = False
        # 奖励逻辑 (保持 AC 优化后的版本，这对 REINFORCE 同样重要)
        if next_pos == self.goal_pos: 
            reward = 10.0; done = True
        elif next_pos in self.obstacles: 
            reward = -15.0; done = True
        elif next_pos == old_pos:
            reward = -10.0 # 撞墙惩罚
        else:
            reward = -0.05
            
        self.agent_pos = next_pos
        self.steps += 1
        if self.steps > self.MAX_STEP: done = True
        return self._get_state(), reward, done

    def _get_state(self): return self.get_state_at(self.agent_pos)
    
    def get_state_at(self, pos):
        ax, ay = pos; gx, gy = self.goal_pos
        state = [(ax/self.size)-0.5, (ay/self.size)-0.5, (gx/self.size)-0.5, (gy/self.size)-0.5]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = ax + dx, ay + dy
                val = 0.0
                if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size: val = 1.0
                elif (nx, ny) in self.obstacles: val = 1.0
                state.append(val)
        return np.array(state, dtype=np.float32)

# ==========================================
# [核心修改 1] Policy Network (只有 Actor)
# ==========================================
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(PolicyNetwork, self).__init__()
        self.affine = nn.Linear(input_dim, 128)
        self.action_head = nn.Linear(128, num_actions)
        # 注意：这里删除了 value_head，因为 REINFORCE 不需要 Critic

    def forward(self, x):
        x = F.relu(self.affine(x))
        # 只输出概率分布
        action_probs = F.softmax(self.action_head(x), dim=-1)
        return action_probs

# ==========================================
# 可视化工具 (适配 PolicyNetwork)
# ==========================================
def draw_policy_overlay(screen, env, model, device):
    for r in range(env.size):
        for c in range(env.size):
            if (r, c) in env.obstacles or (r, c) == env.goal_pos: continue
            fake_state = env.get_state_at((r, c))
            state_tensor = torch.FloatTensor(fake_state).unsqueeze(0).to(device)
            with torch.no_grad():
                # 修改：模型只返回 probs
                probs = model(state_tensor)
                probs = probs.cpu().numpy()[0]
            center_x = c * CELL_SIZE + CELL_SIZE // 2
            center_y = r * CELL_SIZE + CELL_SIZE // 2
            max_len = (CELL_SIZE // 2) * 0.8 
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            best_action = np.argmax(probs)
            for action_idx, (dx, dy) in enumerate(directions):
                length = probs[action_idx] * max_len
                if length < 5: continue 
                end_x = int(center_x + dx * length); end_y = int(center_y + dy * length)
                color = RED if action_idx == best_action else (150, 150, 150)
                width = 4 if action_idx == best_action else 2
                pygame.draw.line(screen, color, (center_x, center_y), (end_x, end_y), width)
                pygame.draw.circle(screen, color, (end_x, end_y), 3)

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
        img = font.render(f"Ep: {episode_num}, R: {reward:.2f} (REINFORCE)", True, BLACK)
        screen.blit(img, (10, 10))

# ==========================================
# [核心修改 2] 主循环：回合更新逻辑
# ==========================================
def main():
    LR = 0.001          # REINFORCE 比较不稳定，学习率通常要低一点
    GAMMA = 0.99
    NUM_EPISODES = 3000 # 收敛通常比 AC 慢，需要更多回合
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridWorld()
    model = PolicyNetwork(env.state_dim, env.action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("REINFORCE Training")
    clock = pygame.time.Clock()
    training_active = True

    for i_episode in range(NUM_EPISODES):
        if not training_active: break
        state = env.reset()
        done = False
        
        # --- REINFORCE 特有：存储一整局的数据 ---
        log_probs = []    # 存每一步的 log_probability
        rewards = []      # 存每一步的 reward
        
        # 1. 玩游戏 (采集轨迹)
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: training_active = False; done = True

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs = model(state_tensor)
            
            m = Categorical(probs)
            action = m.sample()
            
            # 关键：只记录 log_prob，不进行反向传播
            log_probs.append(m.log_prob(action))
            
            next_state, reward, done = env.step(action.item())
            rewards.append(reward)
            
            state = next_state
            
            # 限制渲染帧率，加快训练
            if i_episode % 20 == 0: 
                 # 只在特定回合渲染，否则太慢
                 pass 

        # 2. 游戏结束，开始“秋后算账” (计算 Loss 并更新)
        
        # 2.1 计算折扣回报 (Discounted Returns, G_t)
        returns = []
        R = 0
        # 从最后一步倒着往前算：G_t = r_t + gamma * G_{t+1}
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R) # 插入到最前面
        
        returns = torch.tensor(returns).to(device)
        
        # [关键技巧] 回报标准化 (Normalization)
        # REINFORCE 方差极大，不加这行代码很难收敛
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # 2.2 计算 Policy Loss
        # Loss = - sum( log_prob * G_t )
        loss = []
        for log_prob, R in zip(log_probs, returns):
            # 这里的 R 就是蒙特卡洛回报，代替了 AC 中的 td_error
            loss.append(-log_prob * R)
            
        loss = torch.stack(loss).sum()
        
        # 2.3 反向传播更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_reward = sum(rewards)
        
        # 可视化部分
        if i_episode % 20 == 0:
            render_grid(screen, env, i_episode, total_reward)
            pygame.display.flip()
            # clock.tick(FPS_TRAIN) # 可以取消注释来观察训练过程，但会变慢

    print("Training Finished.")
    
    # --- 演示 Loop ---
    state = env.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(state_tensor)
            action = torch.argmax(probs).item()
        
        state, _, done = env.step(action)
        render_grid(screen, env)
        draw_policy_overlay(screen, env, model, device)
        pygame.display.flip()
        clock.tick(FPS_TEST)
        if done: pygame.time.wait(1000); state = env.reset()

if __name__ == "__main__":
    main()