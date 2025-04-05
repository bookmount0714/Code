import torch
import torch.nn as nn
import torch.nn.functional as F
class dimensionincrease(nn.Module):
    def __init__(self):
        super(dimensionincrease, self).__init__()
        # 定义第一层，从20维升至64维
        self.fc1 = nn.Linear(20, 64)
        # 定义第二层，从64维升至128维
        self.fc2 = nn.Linear(64, 128)
        # 定义第三层，从128维升至256维
        self.fc3 = nn.Linear(128, 256)
        # 定义第四层，从256维升至512维
        self.fc4 = nn.Linear(256, 512)

    def forward(self, x):
        # 通过第一层，应用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 通过第二层，应用ReLU激活函数
        x = F.relu(self.fc2(x))
        # 通过第三层，应用ReLU激活函数
        x = F.relu(self.fc3(x))
        # 通过第四层，此处无需激活函数，直接输出升维后的结果
        x = self.fc4(x)
        return x