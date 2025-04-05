import torch
import json
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
import dgl


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



# mimic_report_ann = '/data1/lijunliang/r2gen/data/mimic/assited/newann.json'
# train_data = json.loads(open(mimic_report_ann,'r').read())['train']
# val_data = json.loads(open(mimic_report_ann,'r').read())['val']
# test_data = json.loads(open(mimic_report_ann,'r').read())['test']
#
# path = json.loads(open('/data1/lijunliang/r2gen/data/mimic/assited/image_complete_path.json','r').read())
# radgraph_mimic_data = json.loads(open('/data1/data/physionet.org/files/radgraph/1.0.0/MIMIC-CXR_graphs.json','r').read())
# max_ = 0
# error_list=[]
# none_list=[]
# data_entity = []
# for image_id in path:
#     try:
#         data_entity = radgraph_mimic_data[image_id]['entities']
#     except(KeyError):
#         error_list.append(image_id)
#         continue
#     if not data_entity:
#         none_list.append(image_id)
#         continue
# none_set = set(none_list)
# none_set = {s.split('/')[-1].split('.')[0][1:] for s in none_set}
# filtered_ann = [obj for obj in train_data if str(obj['study_id']) not in none_set]
# data_dict = {'train': filtered_ann, 'val': val_data, 'test': test_data}
# with open('/data1/lijunliang/r2gen/data/mimic/assited/newann_1.json', 'w') as f:
#     json.dump(data_dict, f, indent=4)


# device = "cuda:0"
# sentence_bert = SentenceTransformer("../../model/sentence_bert")
# print("done")
# sentence_bert.to(device)
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
# study_ids = '/data1/lijunliang/r2gen/data/mimic/assited/study_ids.json'
# mimic_report_ann = '/data1/lijunliang/r2gen/data/mimic/assited/newann_1.json'
# train_data = json.loads(open(mimic_report_ann,'r').read())['train']

# rad = '/data1/data/physionet.org/files/radgraph/1.0.0/MIMIC-CXR_graphs.json'
# rad = json.loads(open(rad, 'r').read())
# keylist = list(rad.keys())
# keyset = set(keylist)

# mimic_report_ann = '/data1/liuyuxue/Data/mimic_cxr/annotation.json'
# mimic_report_radgraph = '/data1/liuyuxue/code_scsr/SCSR/radgraph_mimic.json'
# mimic_rad = json.loads(open(mimic_report_radgraph,'r').read())['text']

# train_data = json.loads(open(mimic_report_ann,'r').read())['train']
# val_data = json.loads(open(mimic_report_ann,'r').read())['val']
# test_data = json.loads(open(mimic_report_ann,'r').read())['test']

#
# reports = [item['report'] for item in train_data]
# ids = [item['study_id'] for item in train_data]
# image_path = [item['image_path'] for item in train_data]
# new_array = [f"{path[0].split('/')[0]}/{path[0].split('/')[1]}/{path[0].split('/')[2]}.txt" for path in image_path]
# # #
# dic_1 = {id: new_array[i] for i, id in enumerate(ids)}
# dic_2 = {id: reports[i] for i, id in enumerate(ids)}
# ids_list = list(dic_1.keys())
# path_list = list(dic_1.values())
#
# #newann
# pathset = set(path_list)
#
# result = [p for p in pathset if p not in keyset]
# print(result)
# #
# diff_set = pathset-keyset
# id_set = {s.split('/')[-1].split('.')[0][1:] for s in diff_set}
# # 过滤掉与 id_set 中相同的对象
# filtered_ann = [obj for obj in train_data if str(obj['study_id']) not in id_set]
#
#
# data_dict = {'train': filtered_ann, 'val': val_data, 'test': test_data}
# with open('/data1/lijunliang/r2gen/data/mimic/assited/newann.json', 'w') as f:
#     json.dump(data_dict, f, indent=4)
##newann

# report_list = list(dic_2.values())
# report_tensor  = torch.from_numpy(sentence_bert.encode(report_list))
# id_file = '/data1/lijunliang/r2gen/data/mimic/assited/study_ids.json'
# path = '/data1/lijunliang/r2gen/data/mimic/assited/image_complete_path.json'
# report_file = '/data1/lijunliang/r2gen/data/mimic/assited/reports.json'
# report_tensor_file = '/data1/lijunliang/r2gen/data/mimic/assited/reports_tensor.npy'
# with open(path, 'w') as outfile:
#     json.dump(path_list, outfile)
# with open(id_file, 'w') as outfile:
#     json.dump(ids_list, outfile)
# #
# with open(report_file, 'w') as outfile:
#     json.dump(report_list, outfile)
# np.save(report_tensor_file,report_tensor)

#
#
# reports = json.loads(open('/data1/lijunliang/r2gen/data/mimic/reports.json','r').read())
# study_id = json.loads(open('/data1/lijunliang/r2gen/data/mimic/study_ids.json','r').read())
# reports_tensor =torch.from_numpy(sentence_bert.encode(reports))
# reports_tensor_file = '../../r2gen/data/mimic/reports_tensor.npy'
# np.save(reports_tensor_file,reports_tensor)


# reports_file = '/data1/lijunliang/r2gen/data/mimic/reports.json'
# reports_tensor_file = '/data1/lijunliang/r2gen/data/mimic/reports_tensor_rad.npy'
# study_ids_file = '/data1/lijunliang/r2gen/data/mimic/study_ids.json'
# ids_file = '/data1/lijunliang/r2gen/data/mimic/ids.json'
# mimic_report_radgraph = '/data1/liuyuxue/code_scsr/SCSR/radgraph_mimic.json'
# reports_rad_file = '/data1/lijunliang/r2gen/data/mimic/reports_rad.json'
#
#
#
# device = "cuda:0"
# sentence_bert = SentenceTransformer("../../model/sentence_bert")
# sentence_bert.to(device)
#
# study_ids_rad_file = '/data1/lijunliang/r2gen/data/mimic/study_ids_rad.json'
#
# # ids = json.study_ids = json.loads(open(loads(open(ids_file, 'r').read())
# study_ids_file, 'r').read())
# reports = json.loads(open(reports_file, 'r').read())
# # reports_tensor = torch.from_numpy(np.load(reports_tensor_file))
# report_dict = {id: reports[i] for i, id in enumerate(study_ids)}
# # ids_list = list(report_dict.keys())
# report_list = list(report_dict.values())
#
# with open(reports_rad_file, 'w', encoding='utf-8') as file:
#     json.dump(report_list, file, ensure_ascii=False, indent=4)
#将radgraph中报告及study_id提取出来



# mimic = '/data1/data/physionet.org/files/radgraph/1.0.0/MIMIC-CXR_graphs.json'
# mimic = json.loads(open(mimic, 'r').read())
# findings_list = []
# keys_list = []
# complete_keys_list = []
# for key, value in mimic.items():
#     text = value.get("text", "")
#     findings_start = text.find("FINDINGS :")
#     impression_start = text.find("IMPRESSION :") if "IMPRESSION :" in text else None
#     if findings_start!= -1:
#         if impression_start is not None:
#             findings_text = text[findings_start + len("FINDINGS :"):impression_start].strip()
#             findings_list.append(findings_text)
#             complete_keys_list.append(key)
#             key_parts = key.split('/')
#             if len(key_parts) == 3:
#                 last_part = key_parts[-1]
#                 number_part = last_part.split('.')[0][1:] if last_part.split('.')[0].startswith('s') else last_part.split('.')[0]
#                 keys_list.append(int(number_part))
#         else:
#             findings_text = text[findings_start + len("FINDINGS :"):].strip()
#             findings_list.append(findings_text)
#             complete_keys_list.append(key)
#             key_parts = key.split('/')
#             if len(key_parts) == 3:
#                 last_part = key_parts[-1]
#                 number_part = last_part.split('.')[0][1:] if last_part.split('.')[0].startswith('s') else last_part.split('.')[0]
#                 keys_list.append(int(number_part))
#
# with open('/data1/lijunliang/r2gen/data/mimic/findings.json', 'w') as outfile:
#     json.dump(findings_list, outfile)
#
# with open('/data1/lijunliang/r2gen/data/mimic/keys.json', 'w') as key_outfile:
#     json.dump(keys_list, key_outfile)
# with open('/data1/lijunliang/r2gen/data/mimic/complete_keys.json', 'w') as key_outfile:
#     json.dump(complete_keys_list, key_outfile)
# findings = json.loads(open('/data1/lijunliang/r2gen/data/mimic/findings.json', 'r').read())
# reports_tensor =torch.from_numpy(sentence_bert.encode(findings))
# reports_tensor_file = '../../r2gen/data/mimic/findings.npy'
# np.save(reports_tensor_file,reports_tensor)
# ids = json.loads(open('/data1/lijunliang/r2gen/data/mimic/study_ids_rad.json', 'r').read())
# keys = json.loads(open('/data1/lijunliang/r2gen/data/mimic/keys.json', 'r').read())
# # unique_ids = list(dict.fromkeys(keys))
#
# ids_set = set(ids)
# keys_set = set(keys)
#
# result = ids_set - keys_set
#
# rad = '/data1/data/physionet.org/files/radgraph/1.0.0/MIMIC-CXR_graphs.json'
# rad = json.loads(open(rad, 'r').read())
# for key, value in rad.items():
#     if str(50014382) in key:
#         print(f"Key: {key}, Value: {value}")
#
# data = json.loads(open('/data1/liuyuxue/Data/mimic_cxr/annotation.json', 'r').read())['train']
# target_study_id = 50000103
#
# print(list(result))
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

