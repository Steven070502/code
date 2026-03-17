import numpy as np
import matplotlib.pyplot as plt

# 1. 造一点假数据（y = 2x + 3）
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

# 2. 初始化 w, b
w = 0.0
b = 0.0
lr = 0.01  # 学习率

# 3. 训练：梯度下降
for epoch in range(1000):
    y_hat = w * x + b          # 预测
    cost = np.mean((y_hat - y) ** 2)  # 代价函数

    # 梯度（求导结果直接用）
    dw = np.mean(2 * x * (y_hat - y))
    db = np.mean(2 * (y_hat - y))

    # 更新
    w -= lr * dw
    b -= lr * db

    if epoch % 100 == 0:
        print(f"epoch {epoch}: cost={cost:.4f}, w={w:.2f}, b={b:.2f}")

# 画图
plt.scatter(x, y)
plt.plot(x, w*x + b, color='red')
plt.show()