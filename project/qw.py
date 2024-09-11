import cma

# 定义一个三维目标函数
# 例如，我们试图找到使 x^2 + y^2 + z^2 最小化的 [x, y, z] 值
def objective_function(v):
    x, y, z = v
    return (x-1)**2 + (y-2)**2 + (z+3)**2

# 初始猜测值
x0 = [1.0, 1.0, 1.0]  # 三维空间中的起始点

# 初始步长
sigma0 = 0.5  # 步长决定了搜索的初始“速度”和探索范围

# 运行 CMA-ES 优化
es = cma.CMAEvolutionStrategy(x0, sigma0).optimize(objective_function)

# 输出结果
res = es.result
print('最优解：', res[0])
print('最优解的目标函数值：', res[1])
