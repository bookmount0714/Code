import dgl
import torch as th
from modules.tokenizers import Tokenizer
import argparse
import torch.nn as nn
import json
from heteroGnn import HeteroGraphNN
from dimensionReduction import DimensionalityReductionNet
from dimensionIncrease import dimensionincrease
feature_size = 512
def parse_agrs():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--image_dir', type=str, default='../../r2gen/data/iu_xray/images/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='../../r2gen/data/iu_xray/annotation.json',
                        help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True,
                        help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search',
                        help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/',
                        help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'],
                        help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, default=None,
                        help='whether to resume the training from existing checkpoints.')
    # parser.add_argument('--resume', type=str, default="./results/iu_xray/R2Gen_base_current_checkpoint.pth", help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--log_dir', type=str, default='./records/log/',
                        help='whether to resume the training from existing checkpoints.')
    args = parser.parse_args()
    return args

iu_report_radgraph = '../../r2gen/data/iu_xray/iu_report_radgraph.json'

test_data = {"text": "The heart is normal in size . The mediastinum is stable . There are postsurgical changes of the left breast . The lungs are clear .", "entities": {"1": {"tokens": "heart", "label": "ANAT-DP", "start_ix": 1, "end_ix": 1, "relations": []}, "2": {"tokens": "normal", "label": "OBS-DP", "start_ix": 3, "end_ix": 3, "relations": [["located_at", "1"]]}, "3": {"tokens": "size", "label": "ANAT-DP", "start_ix": 5, "end_ix": 5, "relations": [["modify", "1"]]}, "4": {"tokens": "mediastinum", "label": "ANAT-DP", "start_ix": 8, "end_ix": 8, "relations": []}, "5": {"tokens": "stable", "label": "OBS-DP", "start_ix": 10, "end_ix": 10, "relations": [["located_at", "4"]]}, "6": {"tokens": "postsurgical", "label": "OBS-DP", "start_ix": 14, "end_ix": 14, "relations": [["modify", "7"]]}, "7": {"tokens": "changes", "label": "OBS-DP", "start_ix": 15, "end_ix": 15, "relations": [["located_at", "9"]]}, "8": {"tokens": "left", "label": "ANAT-DP", "start_ix": 18, "end_ix": 18, "relations": [["modify", "9"]]}, "9": {"tokens": "breast", "label": "ANAT-DP", "start_ix": 19, "end_ix": 19, "relations": []}, "10": {"tokens": "lungs", "label": "ANAT-DP", "start_ix": 22, "end_ix": 22, "relations": []}, "11": {"tokens": "clear", "label": "OBS-DP", "start_ix": 24, "end_ix": 24, "relations": [["located_at", "10"]]}}}
entities = test_data["entities"]

# with open(iu_report_radgraph, 'r', encoding='utf-8') as file:
#     data = json.load(file)
# # 用于存储label和relation类型的集合
# labels = set()
# relations = set()
# # 遍历数据项
# for key, value in data.items():
#     entities = value["entities"]
#     for entity_key, entity_value in entities.items():
#         # 添加label类型
#         labels.add(entity_value["label"])
#         # 遍历并添加relation类型
#         for relation in entity_value["relations"]:
#             relations.add(relation[0])
#
# print("Label的类型有：", labels)
# print("Relation的种类有：", relations)


# 节点类型
node_types = {'OBS-DA', 'OBS-U', 'ANAT-DP', 'OBS-DP'}
# 边的种类
edge_types = {'suggestive_of', 'modify', 'located_at'}


def generate_edge_type(src_type, edge_type, dst_type):
    return f"{src_type}_{edge_type}_{dst_type}"

# 生成所有可能的边类型和编号
edge_type_to_id = {}
for src in node_types:
    for etype in edge_types:
        for dst in node_types:
            edge_type_str = generate_edge_type(src, etype, dst)
            edge_type_to_id[edge_type_str] = len(edge_type_to_id)



# 初始化每种类型的节点ID映射表
node_id_maps = {}
# 初始化entities_data字典
edge_dict = {}
for entity_id, entity_info in entities.items():
    entity_label = entity_info["label"]
    # 如果当前节点类型还未添加到映射表，初始化映射
    if entity_label not in node_id_maps:
        node_id_maps[entity_label] = {}

    # 为当前节点分配新的ID，并更新映射表
    if entity_id not in node_id_maps[entity_label]:
        node_id_maps[entity_label][entity_id] = len(node_id_maps[entity_label])

    for relation_info in entity_info["relations"]:
        relation_type = relation_info[0]
        target_id = relation_info[1]
        target_label = entities[target_id]["label"]
        # 如果目标节点类型还未添加到映射表，初始化映射
        if target_label not in node_id_maps:
            node_id_maps[target_label] = {}

        # 为目标节点分配新的ID，并更新映射表
        if target_id not in node_id_maps[target_label]:
            node_id_maps[target_label][target_id] = len(node_id_maps[target_label])

        # 根据关系类型初始化边数据结构
        edge_key = (entity_label, relation_type, target_label)
        if edge_key not in edge_dict:
            edge_dict[edge_key] = ([], [])

        # 添加边信息到边数据结构，使用映射后的ID
        src_mapped_id = node_id_maps[entity_label][entity_id]
        dst_mapped_id = node_id_maps[target_label][target_id]
        edge_dict[edge_key][0].append(src_mapped_id)
        edge_dict[edge_key][1].append(dst_mapped_id)

# 将边列表转换为张量
for edge_type, (src_list, dst_list) in edge_dict.items():
    edge_dict[edge_type] = (th.tensor(src_list), th.tensor(dst_list))
g = dgl.heterograph(edge_dict)


# 创建token到ID的映射字典
token_to_id = {}
# 用于给节点编号的计数器，分别为每种类型的节点计数
node_counter = {}
for entity_id, entity_info in entities.items():
    # 获取实体的类型
    entity_type = entity_info["label"]

    # 如果实体类型是第一次遇见，则初始化计数器
    if entity_type not in node_counter:
        node_counter[entity_type] = 0

    # 获取实体的token
    token = entity_info["tokens"]

    # 建立token到(类型, ID)的映射
    # 注意：ID是从0开始的整数，因此使用计数器的当前值
    token_to_id[token] = (entity_type, node_counter[entity_type])

    # 更新当前类型的节点计数器
    node_counter[entity_type] += 1


# 类型特定的线性变换字典
type_transforms = {node_type: nn.Linear(feature_size, feature_size) for node_type in node_types}


"""
节点赋值
"""
def get_features_for_token(token):
    # 这里随机生成一个特征向量，实际中应该用已知的特征值替换
    return th.randn((feature_size,))

features_by_type = {}
for token, (node_type, node_id) in token_to_id.items():
    # 获取当前token的特征向量
    feature_vector = get_features_for_token(token)

    # 应用类型特定的线性变换
    transformed_feature_vector = type_transforms[node_type](feature_vector)

    # 如果当前类型的节点还没有特征张量，则创建一个
    if node_type not in features_by_type:
        # 计算当前节点类型的节点数量
        num_nodes = max([id for ntype, id in token_to_id.values() if ntype == node_type]) + 1
        # 为当前节点类型初始化特征张量
        features_by_type[node_type] = th.zeros((num_nodes, feature_size))

# 为对应节点赋予特征向量
    features_by_type[node_type][node_id] = transformed_feature_vector + feature_vector

for node_type, features in features_by_type.items():
    g.nodes[node_type].data['feat'] = features


"""
构建meta边表
"""
edge_type_to_id = {
('OBS-DA', 'suggestive_of', 'OBS-DA'): 0,
('OBS-DA', 'suggestive_of', 'OBS-U'): 1,
('OBS-DA', 'suggestive_of', 'ANAT-DP'): 2,
('OBS-DA', 'suggestive_of', 'OBS-DP'): 3,
('OBS-DA', 'modify', 'OBS-DA'): 4,
('OBS-DA', 'modify', 'OBS-U'): 5,
('OBS-DA', 'modify', 'ANAT-DP'): 6,
('OBS-DA', 'modify', 'OBS-DP'): 7,
('OBS-DA', 'located_at', 'OBS-DA'): 8,
('OBS-DA', 'located_at', 'OBS-U'): 9,
('OBS-DA', 'located_at', 'ANAT-DP'): 10,
('OBS-DA', 'located_at', 'OBS-DP'): 11,
('OBS-U', 'suggestive_of', 'OBS-DA'): 12,
('OBS-U', 'suggestive_of', 'OBS-U'): 13,
('OBS-U', 'suggestive_of', 'ANAT-DP'): 14,
('OBS-U', 'suggestive_of', 'OBS-DP'): 15,
('OBS-U', 'modify', 'OBS-DA'): 16,
('OBS-U', 'modify', 'OBS-U'): 17,
('OBS-U', 'modify', 'ANAT-DP'): 18,
('OBS-U', 'modify', 'OBS-DP'): 19,
('OBS-U', 'located_at', 'OBS-DA'): 20,
('OBS-U', 'located_at', 'OBS-U'): 21,
('OBS-U', 'located_at', 'ANAT-DP'): 22,
('OBS-U', 'located_at', 'OBS-DP'): 23,
('ANAT-DP', 'suggestive_of', 'OBS-DA'): 24,
('ANAT-DP', 'suggestive_of', 'OBS-U'): 25,
('ANAT-DP', 'suggestive_of', 'ANAT-DP'): 26,
('ANAT-DP', 'suggestive_of', 'OBS-DP'): 27,
('ANAT-DP', 'modify', 'OBS-DA'): 28,
('ANAT-DP', 'modify', 'OBS-U'): 29,
('ANAT-DP', 'modify', 'ANAT-DP'): 30,
('ANAT-DP', 'modify', 'OBS-DP'): 31,
('ANAT-DP', 'located_at', 'OBS-DA'): 32,
('ANAT-DP', 'located_at', 'OBS-U'): 33,
('ANAT-DP', 'located_at', 'ANAT-DP'): 34,
('ANAT-DP', 'located_at', 'OBS-DP'): 35,
('OBS-DP', 'suggestive_of', 'OBS-DA'): 36,
('OBS-DP', 'suggestive_of', 'OBS-U'): 37,
('OBS-DP', 'suggestive_of', 'ANAT-DP'): 38,
('OBS-DP', 'suggestive_of', 'OBS-DP'): 39,
('OBS-DP', 'modify', 'OBS-DA'): 40,
('OBS-DP', 'modify', 'OBS-U'): 41,
('OBS-DP', 'modify', 'ANAT-DP'): 42,
('OBS-DP', 'modify', 'OBS-DP'): 43,
('OBS-DP', 'located_at', 'OBS-DA'): 44,
('OBS-DP', 'located_at', 'OBS-U'): 45,
('OBS-DP', 'located_at', 'ANAT-DP'): 46,
('OBS-DP', 'located_at', 'OBS-DP'): 47
}

# 假设嵌入的维度为embedding_dim
embedding_dim = 512
num_edge_types = len(edge_type_to_id)

# 创建嵌入层
edge_embeddings = nn.Embedding(num_edge_types, embedding_dim)
# 初始化权重（可选步骤，根据需要选择是否使用）
nn.init.xavier_uniform_(edge_embeddings.weight)
# 为边赋予特征
for canonical_etype in g.canonical_etypes:
    src_type, etype, dst_type = canonical_etype

    # 获取所有边的ID
    edge_ids = g.edges(etype=canonical_etype, form='eid')

    # 获取边类型对应的编号
    # 注意这里需要使用规范化的边类型作为键来查找ID
    edge_type_id = edge_type_to_id[(src_type, etype, dst_type)]

    # 获取嵌入向量
    edge_feat = edge_embeddings(th.LongTensor([edge_type_id]))

    # 为边赋予特征
    g.edges[canonical_etype].data['feat'] = edge_feat.repeat(len(edge_ids), 1)



# 获取节点特征字典
node_features = {ntype: g.nodes[ntype].data['feat'] for ntype in g.ntypes}

# 获取边特征字典
edge_features = {etype: g.edges[etype].data['feat'] for etype in g.canonical_etypes}
rel_names = ['suggestive_of', 'modify', 'located_at']
ntypes = {'OBS-DA', 'OBS-U', 'ANAT-DP', 'OBS-DP'}


model = HeteroGraphNN(20, 20,20 , g.etypes)

h_dict = model(g,node_features)

print(h_dict)


# # 打印图的基本信息
# print(g)
#
# # 获取图的节点类型
# node_types = g.ntypes
# print('Node types:', node_types)
#
# # 获取图的边类型
# edge_types = g.etypes
# print('Edge types:', edge_types)
#
# # 获取图的规范边类型（即源节点类型、边类型、目标节点类型）
# canonical_edge_types = g.canonical_etypes
# print('Canonical edge types:', canonical_edge_types)
#
# # 遍历所有节点类型，并打印节点和其特征
# for ntype in node_types:
#     print(f'Number of "{ntype}" nodes:', g.number_of_nodes(ntype))
#     # 如果节点有特征，打印特征
#     if g.nodes[ntype].data:
#         print(f'Features of "{ntype}" nodes:', g.nodes[ntype].data)
#
# # 遍历所有规范边类型，并打印边和其特征
# for etype in canonical_edge_types:
#     src_type, edge_type, dst_type = etype
#     print(f'Number of "{edge_type}" edges:', g.number_of_edges(etype))
#     # 如果边有特征，打印特征
#     if g.edges[etype].data:
#         print(f'Features of "{edge_type}" edges:', g.edges[etype].data)


# print(g)
# args = parse_agrs()
# tokenizer = Tokenizer(args)
# print(tokenizer.token2idx['heart'])





