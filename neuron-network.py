import numpy as np
import matplotlib.pyplot as plt

# --- 1. 数据和网络定义（和上次一样） ---
np.random.seed(42)
x = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y_true = np.sin(x)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 初始化参数（注意：这里是随机的！这是打破对称的关键）
input_size = 1
hidden_size = 10  # 我们还是用10个神经元
output_size = 1

# 关键：W1 是随机生成的，每个神经元的初始权重都不一样！
W1 = np.random.randn(input_size, hidden_size) 
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# --- 2. 训练过程（简略，和上次一样） ---
learning_rate = 0.2
epochs = 15000

for i in range(epochs):
    # 前向传播
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)  # a1 的形状是 (100, 10)，10列代表10个神经元的输出
    z2 = np.dot(a1, W2) + b2
    y_pred = z2
    
    # 反向传播（省略具体计算，重点看结果）
    loss = np.mean((y_pred - y_true) ** 2)
    delta_y = 2 * (y_pred - y_true) / len(x)
    dW2 = np.dot(a1.T, delta_y)
    db2 = np.sum(delta_y, axis=0, keepdims=True)
    delta_z1 = np.dot(delta_y, W2.T) * (a1 * (1 - a1))
    dW1 = np.dot(x.T, delta_z1)
    db1 = np.sum(delta_z1, axis=0, keepdims=True)
    
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# --- 3. 见证奇迹：分解神经网络 ---

plt.figure(figsize=(14, 6))

# 子图1：最终的拟合结果
plt.subplot(1, 2, 1)
plt.plot(x, y_true, 'r--', label='真实 sin(x)', linewidth=3)
plt.plot(x, y_pred, 'k-', label='神经网络拟合', linewidth=2)
plt.title('最终成果：10个神经元合作的结果')
plt.legend()
plt.grid(True)

# 子图2：把10个神经元单独拉出来示众！
plt.subplot(1, 2, 2)

# 我们要看看每个神经元对最终结果的贡献
# 最终结果 y_pred = a1 * W2 + b2
# 我们可以把每个神经元的输出(a1[:, i])乘以它的权重(W2[i, 0])画出来

for i in range(hidden_size):
    # 计算第 i 个神经元的“贡献曲线”
    neuron_contribution = a1[:, i] * W2[i, 0]
    
    # 画出这条曲线
    plt.plot(x, neuron_contribution, label=f'神经元 {i+1}', linestyle='--')

# 把所有神经元的贡献加起来（再加上偏置），就是最终的黑线
# 为了清晰，我们不画总和，只画零件

plt.title('拆解图：每个神经元在做不同的事')
plt.axhline(0, color='k', lw=0.5)
plt.legend(loc='upper right', fontsize='small')
plt.grid(True)
plt.ylim([-1.5, 1.5])

plt.tight_layout()
plt.show()