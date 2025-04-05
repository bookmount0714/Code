import torch
import torch.nn as nn
import torch.nn.functional as F


class DimensionalityReductionNet(nn.Module):
    def __init__(self):
        super(DimensionalityReductionNet, self).__init__()
        # 定义第一层，从512维降至256维
        self.fc1 = nn.Linear(512, 256)
        # 定义第二层，从256维降至128维
        self.fc2 = nn.Linear(256, 128)
        # 定义第三层，从128维降至64维
        self.fc3 = nn.Linear(128, 64)
        # 定义第四层，从64维降至20维
        self.fc4 = nn.Linear(64, 20)

    def forward(self, x):
        # 通过第一层，应用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 通过第二层，应用ReLU激活函数
        x = F.relu(self.fc2(x))
        # 通过第三层，应用ReLU激活函数
        x = F.relu(self.fc3(x))
        # 通过第四层，此处无需激活函数，直接输出降维后的结果
        x = self.fc4(x)
        return x
