import torch
import torch.nn as nn
import torch.nn.functional as F
class Imageclassify(nn.Module):
    def __init__(self):
        super(Imageclassify, self).__init__()
        self.fc1 = nn.Linear(384, 836)

    def forward(self, x):
        # 通过第一层，应用ReLU激活函数
        x = F.sigmoid(self.fc1(x))
        return x
