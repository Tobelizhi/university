import numpy as np
import matplotlib.pyplot as plt

# 定义 sin 函数
def sin(x):
    return np.sin(x)

# 定义线性模型
def linear_model(x, m, b):
    return m * x + b

# 计算损失函数（均方误差）
def compute_loss(x, y, m, b):
    y_pred = linear_model(x, m, b)
    return ((y_pred - y) ** 2).mean()

# 梯度下降算法
def gradient_descent(x, y, learning_rate, num_iterations):
    m = b = 0  # 初始斜率和截距
    for i in range(num_iterations):
        y_pred = linear_model(x, m, b)
        m_grad = -2 * (y_pred - y).mean() * x.mean()
        b_grad = -2 * (y_pred - y).mean()
        m -= learning_rate * m_grad
        b -= learning_rate * b_grad
    return m, b

# 生成数据点
x_data = np.linspace(-np.pi, np.pi, 100)
y_data = sin(x_data)

# 执行梯度下降
learning_rate = 0.01
num_iterations = 1000
m, b = gradient_descent(x_data, y_data, learning_rate, num_iterations)

# 打印拟合直线的参数
print(f"Fitted line: y = {m:.2f}x + {b:.2f}")

# 绘制图像
plt.figure(figsize=(10, 5))
plt.scatter(x_data, y_data, label='Data Points')
plt.plot(x_data, linear_model(x_data, m, b), label='Fitted Line', color='red')
plt.plot(x_data, y_data, label='True sin(x)', linestyle='dashed', color='blue')

# 绘制真实 sin(x) 函数图像
x_true = np.linspace(-np.pi, np.pi, 1000)
y_true = sin(x_true)
plt.plot(x_true, y_true, label='True sin(x)', linestyle='dotted')

plt.title('Linear Fit of sin(x) using Gradient Descent')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.show()

# 计算拟合直线的损失
loss = compute_loss(x_data, y_data, m, b)
print(f"Loss: {loss:.4f}")