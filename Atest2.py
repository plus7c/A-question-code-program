import dwavebinarycsp
from dwave.system import LeapHybridSampler

# 输入数据，包括评分卡1、评分卡2、评分卡3的通过率和坏账率
t1 = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
h1 = [0.1, 0.12, 0.15, 0.18, 0.22, 0.27, 0.32, 0.38, 0.44, 0.5]

t2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
h2 = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]

t3 = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
h3 = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]

# 定义CSP问题
csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)

# 定义变量
z1 = ['z1_'+str(i) for i in range(10)]
z2 = ['z2_'+str(i) for i in range(10)]
z3 = ['z3_'+str(i) for i in range(10)]

# 添加约束
for i in range(10):
    csp.add_constraint(lambda z1i, z2i, z3i: z1i + z2i + z3i == 1, [z1[i], z2[i], z3[i]])
    csp.add_constraint(lambda z1i, z2i, z3i: z1i + z2i + z3i >= 1, [z1[i], z2[i], z3[i]])

# 定义能量函数
H = 0
for i in range(10):
    H += -1 * (h1[i] + h2[i] + h3[i]) * (z1[i] + z2[i] + z3[i]) + (t1[i] * z1[i] + t2[i] * z2[i] + t3[i] * z3[i])

# 转化为二进制问题
bqm = dwavebinarycsp.stitch(csp, max_graph_size=10000)

# 使用D-Wave Leap API进行量子求解
sampler = LeapHybridSampler()
response = sampler.sample(bqm, num_reads=100)

# 处理求解结果
best_solution = response.first.sample
best_energy = response.first.energy
for var, value in best_solution.items():
if value:
print(f"信用评分卡{var//10+1}设置为第{var%10+1}项闯值")
print(f"最大总收入为{-(best_energy-bias)/2}")