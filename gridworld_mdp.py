import numpy as np
import random

class GridWorld:
    def __init__(self, rows=5, cols=5,
                 A=(0, 0), A_prime=(4, 0), rA=10,
                 B=(0, 1), B_prime=(4, 1), rB=5,
                 C=(0, 2), C_prime=(4, 2), rC=3,
                 step_cost=-3):
        """
        rows, cols: 网格大小
        A, A_prime: 特殊位置 A 及其目标 A'
        B, B_prime: 特殊位置 B 及其目标 B'
        C, C_prime: 特殊位置 C 及其目标 C'
        rA, rB, rC: A->A', B->B', C->C' 的奖励
        step_cost: 普通移动的代价
        """
        self.rows = rows
        self.cols = cols
        
        # 特殊位置
        self.A = A
        self.A_prime = A_prime
        self.rA = rA
        
        self.B = B
        self.B_prime = B_prime
        self.rB = rB
        
        self.C = C
        self.C_prime = C_prime
        self.rC = rC
        
        self.step_cost = step_cost
        
        # 动作集合：上、下、左、右 (索引 0, 1, 2, 3)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
        self.action_symbols = ['^', 'v', '<', '>'] # 可选：用于打印策略
        self.n_actions = len(self.actions)
        
        # 状态集合：每个 (i, j) 都是一个状态
        self.states = [(i, j) for i in range(self.rows) for j in range(self.cols)]
        self.n_states = len(self.states)
    
    def next_state_and_reward(self, state, action):
        """
        给定当前状态和动作，返回下一个状态和即时奖励。
        如果 state 属于 A/B/C，则无论 action 是什么，都跳转到相应 A'/B'/C' 并给奖励。
        否则按照上下左右的动作进行普通移动，若越界则留在原地。
        """
        # 如果是特殊位置，直接跳转
        if state == self.A:
            return self.A_prime, self.rA
        if state == self.B:
            return self.B_prime, self.rB
        if state == self.C:
            return self.C_prime, self.rC
        
        # 普通位置，根据 action 移动
        (i, j) = state
        (di, dj) = action
        new_i = i + di
        new_j = j + dj
        
        # 边界判断
        if new_i < 0 or new_i >= self.rows or new_j < 0 or new_j >= self.cols:
            # 越界则留在原地，但依然付出 step_cost
            return (i, j), self.step_cost
        else:
            # 合法移动
            return (new_i, new_j), self.step_cost

def value_iteration(env, gamma=0.9, theta=1e-5):
    """
    env: GridWorld 环境
    gamma: 折扣因子
    theta: 收敛阈值，用于判断值函数更新前后差异是否足够小
    
    返回：
    V: 收敛后的状态价值函数 (dict: state -> value)
    policy: 导出的贪心策略 (dict: state -> 最优动作索引)
    iterations: 迭代次数
    """
    # 初始化价值函数
    V = {s: 0.0 for s in env.states}
    
    iterations = 0
    while True:
        delta = 0
        # 对所有状态做一次同步更新
        newV = V.copy() # 使用 .copy() 以免影响当前轮次的计算
        for s in env.states:
            # 针对每个动作，计算期望回报
            action_values = []
            for a_idx, a in enumerate(env.actions):
                s_next, r = env.next_state_and_reward(s, a)
                action_values.append(r + gamma * V[s_next])
            
            # 取最大动作价值
            best_value = max(action_values)
            # 计算差值需要在更新 newV 之前
            delta = max(delta, abs(best_value - V[s]))
            newV[s] = best_value # 更新到新价值函数的字典中
        
        # 完成一轮所有状态的更新后，再整体替换 V
        V = newV
        iterations += 1
        
        # 判断是否收敛
        if delta < theta:
            break
            
    # 由价值函数导出策略（对每个状态选使 Q(s,a) 最大的 a）
    policy = {}
    for s in env.states:
        # 对所有动作计算 Q(s,a)
        best_a_idx = 0
        best_value = float('-inf')
        for a_idx, a in enumerate(env.actions):
            s_next, r = env.next_state_and_reward(s, a)
            q_sa = r + gamma * V[s_next]
            # 使用一个小扰动来处理价值相同的情况，避免策略在等价值动作间震荡 (可选)
            # 这里简单地取第一个遇到的最大值
            if q_sa > best_value:
                best_value = q_sa
                best_a_idx = a_idx
        policy[s] = best_a_idx
    
    return V, policy, iterations

def policy_evaluation(env, policy, gamma=0.9, theta=1e-5):
    """
    给定固定策略 policy，对其进行策略评估，返回收敛的 V_pi
    policy: dict, state -> 动作索引
    """
    V = {s: 0.0 for s in env.states}
    
    while True:
        delta = 0
        newV = V.copy() # 使用 .copy()
        for s in env.states:
            # 根据 policy[s] 给出的动作来计算价值
            a_idx = policy[s]
            a = env.actions[a_idx]
            s_next, r = env.next_state_and_reward(s, a)
            # 贝尔曼期望方程 V_pi(s) = R(s, pi(s), s') + gamma * V_pi(s') (因为转移是确定的)
            current_val = r + gamma * V[s_next] # 注意这里用旧的V计算
            delta = max(delta, abs(current_val - V[s]))
            newV[s] = current_val # 更新到 newV
        V = newV # 整体更新 V
        if delta < theta:
            break
    
    return V

def policy_improvement(env, V, policy, gamma=0.9):
    """
    根据给定的 V，进行一次贪心策略改进
    返回改进后的新策略 new_policy 和策略是否稳定 stable
    """
    new_policy = {}
    stable = True # 假设策略稳定
    for s in env.states:
        old_action_idx = policy.get(s) # 获取旧策略的动作（用于比较）
        
        best_a_idx = 0
        best_value = float('-inf')
        for a_idx, a in enumerate(env.actions):
            s_next, r = env.next_state_and_reward(s, a)
            q_sa = r + gamma * V[s_next]
            if q_sa > best_value:
                best_value = q_sa
                best_a_idx = a_idx
        new_policy[s] = best_a_idx
        
        # 检查新策略的动作是否与旧策略不同
        if old_action_idx is not None and best_a_idx != old_action_idx:
            stable = False # 只要有一个状态的动作改变，策略就不稳定
            
    return new_policy, stable

def policy_iteration(env, gamma=0.9):
    """
    策略迭代主函数
    返回：
    policy: 收敛后的最优策略
    V: 对应的状态价值函数
    iterations: 策略改进的次数 (即外层循环次数)
    """
    # 1) 初始化随机策略
    policy = {}
    for s in env.states:
        policy[s] = random.randint(0, env.n_actions - 1)  # 随机动作索引
    
    iterations = 0
    while True:
        iterations += 1 # 每次进入循环代表一次策略改进/评估周期
        # 2) 策略评估: 基于当前 policy 计算 V
        V = policy_evaluation(env, policy, gamma=gamma)
        
        # 3) 策略改进: 基于 V 得到新策略 new_policy，并检查是否稳定
        new_policy, stable = policy_improvement(env, V, policy, gamma=gamma) # 传入旧 policy 以检查稳定
        
        # 更新策略
        policy = new_policy
        
        # 4) 检查策略是否稳定
        if stable:
            break # 如果策略没有改变，则已找到最优策略和价值函数
            
    # 再次评估以确保返回的 V 是最终稳定策略下的 V
    V = policy_evaluation(env, policy, gamma=gamma) 
    
    return policy, V, iterations


# 辅助函数：打印策略
def print_policy(policy, env):
    print("Optimal Policy:")
    policy_grid = [[' ' for _ in range(env.cols)] for _ in range(env.rows)]
    for s, a_idx in policy.items():
        i, j = s
        if s == env.A or s == env.B or s == env.C:
             policy_grid[i][j] = '*' # 特殊状态标记
        else:
             policy_grid[i][j] = env.action_symbols[a_idx]

    for i in range(env.rows):
        print("+---" * env.cols + "+")
        row_str = "|"
        for j in range(env.cols):
            row_str += f" {policy_grid[i][j]} |"
        print(row_str)
    print("+---" * env.cols + "+")

# 辅助函数：打印价值函数
def print_value_function(V, env):
    print("Value Function:")
    value_grid = [[0.0 for _ in range(env.cols)] for _ in range(env.rows)]
    for s, val in V.items():
        i, j = s
        value_grid[i][j] = val

    for i in range(env.rows):
        print("+-------" * env.cols + "+")
        row_str = "|"
        for j in range(env.cols):
            row_str += f" {value_grid[i][j]:6.2f} |"
        print(row_str)
    print("+-------" * env.cols + "+")


if __name__ == "__main__":
    # 创建环境
    env = GridWorld(rows=8, cols=8,
                    A=(0, 1), A_prime=(7, 1), rA=10,
                    B=(0, 3), B_prime=(4, 3), rB=5,
                    C=(0, 5), C_prime=(2, 5), rC=3,
                    step_cost=-1) # 保持默认设置
    
    results = {}

    for gamma in [0.9, 0.6]:
        print(f"===== gamma = {gamma} =====")
        
        # 值迭代
        V_vi, pi_vi, vi_iter = value_iteration(env, gamma=gamma, theta=1e-5)
        print(f"[Value Iteration] Converged in {vi_iter} iterations.")
        # print_value_function(V_vi, env) # 可选：打印价值函数
        # print_policy(pi_vi, env)       # 可选：打印策略
        
        # 策略迭代
        # (The local redefinitions and related comments below are redundant and will be removed)

        # 运行策略迭代 (This will use the global policy_iteration and policy_improvement)
        pi_pi, V_pi, pi_iter = policy_iteration(env, gamma=gamma)
        print(f"[Policy Iteration] Stabilized in {pi_iter} policy improvement steps.")
        # print_value_function(V_pi, env) # 可选：打印价值函数

        results[gamma] = {'VI_iters': vi_iter, 'PI_iters': pi_iter}

    print("===== Summary =====")
    for gamma, res in results.items():
        print(f"Gamma = {gamma}:")
        print(f"  Value Iteration iterations: {res['VI_iters']}")
        print(f"  Policy Iteration iterations: {res['PI_iters']}") 