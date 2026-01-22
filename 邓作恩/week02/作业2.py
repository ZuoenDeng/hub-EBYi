import torch
import matplotlib.pyplot as plt
import numpy as np

# 1.生成模拟数据
X_numpy = np.linspace(0, 20, 100)
Y_numpy = 50 * np.sin(X_numpy) + np.random.randn(100) * 20 # 生成100个数据点，每个数据点有0.4的噪声
plt.plot(X_numpy, Y_numpy, 'b.')
plt.show()
X = torch.from_numpy(X_numpy).float()
Y = torch.from_numpy(Y_numpy).float()

# 2. 直接创建参数张量 a 和 b
a = torch.randn(1, requires_grad=True, dtype=torch.float) # 其中1是输出的维度，requires_grad=True表示需要计算梯度，dtype=torch.float表示数据类型为浮点数
b = torch.randn(1, requires_grad=True, dtype=torch.float)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss() # 回归任务，使用均方误差损失函数
optimizer = torch.optim.Adam([a, b], lr=0.01) # 优化器，基于 a b 梯度 自动更新

# 4. 训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * np.sin(X) + b
    y_pred = a * np.sin(X) + b

    # 计算损失
    loss = loss_fn(y_pred, Y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
a_learned = a.item()
b_learned = b.item()
print(f"拟合的倍率 a: {a_learned:.4f}")
print(f"拟合的偏置 b: {b_learned:.4f}")
print("---" * 10)

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
    y_predicted = a_learned * np.sin(X) + b_learned

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, Y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label=f'Model: y = {a_learned:.2f}*sin(x) + {b_learned:.2f}', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()