import torch
import json
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer

# 计算余弦相似度
def cosine_similarity_batch(tensor_a, tensor_b):
    # tensor_a: (N, D)
    # tensor_b: (1, D)
    dot_product = torch.matmul(tensor_a, tensor_b.T)  # (N, 1)
    norm_a = torch.norm(tensor_a, dim=1, keepdim=True)  # (N, 1)
    norm_b = torch.norm(tensor_b, dim=1, keepdim=True)  # (1, 1)
    norms = norm_a * norm_b
    similarity = dot_product / norms
    return similarity.squeeze()  # (N,)

# device = "cuda:1"
# sentence_bert = SentenceTransformer("../../model/sentence_bert")
print("done")
# # sentence_bert.to(device)
# iu_report_radgraph = '../../r2gen/data/iu_xray/annotation.json'
# train_data = json.loads(open(iu_report_radgraph,'r').read())['train']
# # ids = [item['id'] for item in train_data]   #样本对应名称
#
# # reports = [item['report'] for item in train_data]   #样本文字报告
#
# # reports_tensor =torch.from_numpy(sentence_bert.encode(reports))    #样本报告转化后张量
#
# reports_tensor_file = '../../r2gen/data/iu_xray/reports_tensor.npy'
# reports_file = '../../r2gen/data/iu_xray/reports.json'
# ids_file = '../../r2gen/data/iu_xray/ids.json'
#
# # np.save(reports_tensor_file,reports_tensor)
# # with open(ids_file, 'w', encoding='utf-8') as file:
# #     json.dump(ids, file, ensure_ascii=False, indent=4)
# # with open(reports_file, 'w', encoding='utf-8') as file:
# #     json.dump(reports, file, ensure_ascii=False, indent=4)
#
# ids = json.loads(open(ids_file,'r').read())
# reports = json.loads(open(reports_file,'r').read())
# reports_tensor = torch.from_numpy(np.load(reports_tensor_file))
#
# test = reports_tensor[0,:].unsqueeze(0)
#
#
# # similarities = sentence_bert.similarity(reports_tensor,test)
#
# #不用bert模型余弦相似度
# similarities_ = cosine_similarity_batch(reports_tensor, test)
#
# # 获取最大的三个值及其对应的下标
#
# # 取出最后三个下标，即最大值对应的下标
# top_3_indices = torch.argsort(similarities_, descending=True)[:3]
#
# # 对应的最大值
# top_3_values = similarities_[top_3_indices]
#
# print(f"前三个最大的值: {top_3_values}")
# print(f"对应的下标: {top_3_indices}")



#视觉转化为384维度的张量，用于与知识库中的报告向量进行相似度比较
class ImageTransform(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim1, hidden_dim2):
        super(ImageTransform, self).__init__()
        # 第一个卷积层
        self.conv1d_1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AdaptiveAvgPool1d(49)

        # 第二个卷积层
        self.conv1d_2 = nn.Conv1d(in_channels=hidden_dim1, out_channels=hidden_dim2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # 输入 x 的形状为 (batch_size, 98, 2048)
        x = x.permute(0, 2, 1)  # 变换维度为 (batch_size, 2048, 98)

        # 第一个卷积层、批量归一化、激活和池化
        x = self.conv1d_1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 第二个卷积层、批量归一化、激活和池化
        x = self.conv1d_2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.squeeze(-1)  # 去掉最后一个维度，形状为 (batch_size, hidden_dim2)
        x = self.fc(x)  # 经过全连接层，输出形状为 (batch_size, output_dim)
        return x

# # 定义输入和输出的维度
# input_dim = 2048
# hidden_dim1 = 1024
# hidden_dim2 = 512
# output_dim = 384
#
# model = ImageTransform(input_dim, output_dim, hidden_dim1, hidden_dim2)

