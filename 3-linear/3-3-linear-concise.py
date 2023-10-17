from torch import nn
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2

# 合成的
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size: int, is_train=True) -> data.DataLoader:
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)


net = nn.Sequential(nn.Linear(2, 1))
# net [0] 选择第一个图层

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()

# 小批量随机梯度下降算法
# trainer 获得 net 参数的引用，来训练
trainer: torch.optim.SGD = torch.optim.SGD(net.parameters(), lr=0.03)

# [3.3.7 训练]
num_epochs = 3
for epoch in range(num_epochs):
    for x, y in data_iter:
        # x: labels y: features
        l = loss(net(x), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# [练习还没做]
