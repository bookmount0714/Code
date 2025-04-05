from collections import OrderedDict
import json
import numpy
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
import sys
from segment_anything import sam_model_registry,SamAutomaticMaskGenerator,SamPredictor
import matplotlib.pyplot as plt
from modules import globals_ele
from graph.heteroGnn import HeteroGraphNN
import dgl
import torch.nn.functional as F
from bert.similarityCulculate import ImageTransform
from torch.nn.functional import cosine_similarity
from bert.similarityCulculate import cosine_similarity_batch
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import hdbscan
from modules.imageclassify import Imageclassify


device = "cuda:0"
node_types = {'OBS-DA', 'OBS-U', 'ANAT-DP', 'OBS-DP'}
edge_types = {'suggestive_of', 'modify', 'located_at','self-loop'}
feature_size = 512
class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        # self.sentence_bert = SentenceTransformer("../../model/sentence_bert")
        self.sentence_bert = SentenceTransformer("/data1/lijunliang/model/sentence_bert")
        self.reducer = umap.UMAP(n_components=2, random_state=42)
        #节点类别特征
        self.type_transforms = nn.Sequential(OrderedDict([
            ('OBS-DA',nn.Linear(feature_size, feature_size)),
            ('OBS-U', nn.Linear(feature_size, feature_size)),
            ('ANAT-DP', nn.Linear(feature_size, feature_size)),
            ('OBS-DP', nn.Linear(feature_size, feature_size)),
        ]))
        #图卷积
        self.graph_conv = HeteroGraphNN(20, 20,20 , edge_types)
        #图像转化384维,该数据用于与报告对应的384维的向量进行比较，选取前三相似的报告用于生成图
        self.imagetransform = ImageTransform(2048,384,1024,512)
        #接收384维图像张量，将其转化为x维的张量，x取决于报告分类的种类数量
        self.Imageclassify = Imageclassify()
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images,targets=None,batch_report_labels=None,images_id=(),mode='train'):
        #通过visual_extractor提取视觉特征
        att_feats_sam_0, fc_feats_sam_0 = self.visual_extractor(images[:,2])
        att_feats_sam_1, fc_feats_sam_1 = self.visual_extractor(images[:,3])
        fc_feats_sam = torch.cat((fc_feats_sam_0, fc_feats_sam_1), dim=1)   #fc_feats_sam 16,4096
        att_feats_sam = torch.cat((att_feats_sam_0, att_feats_sam_1), dim=1)    #attfeats_sam 16,98,2048


        att_feats_0, fc_feats_0 = self.visual_extractor(images[:,0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:,1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        # 将图片数据转化为384维(16,384)
        att_feats_384 = self.imagetransform(att_feats)
        #寻找到批次中图像最相近的3个报告
        ids = globals_ele.ids
        reports = globals_ele.reports
        reports_tensor = globals_ele.reports_tensor.to(device)
        # reports = json.loads(open(reports_file,'r').read())
        # reports_tensor = torch.from_numpy(np.load(reports_tensor_file)).to(device)

        nums = len(images_id)
        #初始化下记录下标张量,一个批次16张图片,获取每张图片在报告知识库reports_tensor中最相似的前3个(16,3),每张图片对应3个下标
        result = torch.zeros((nums, 3), dtype=torch.long, device=device)
        top_3_values = torch.zeros((nums, 3), dtype=torch.float, device=device)
        for i in range(att_feats_384.shape[0]):
            similarities = cosine_similarity_batch(reports_tensor,att_feats_384[i,:].unsqueeze(0))
            top_3_indices = torch.argsort(similarities,descending=True)[:3]
            top_3_value = similarities[top_3_indices]
            result[i] = top_3_indices
            top_3_values[i] = top_3_value

        #获取对应报告(16,3)转化过来的图
        titles = [[ids[idx] for idx in image_result] for image_result in result.cpu().numpy()]
        prefix = '/data1/lijunliang/r2gen/data/iu_xray/report_txt/'
        suffix = '.txt'
        features_list = []
        original_lengths_list = []  # 用来保存每个样本的原始长度

        for title_group in titles:
            group_features_list = []
            original_lengths_group = []  # 用来保存当前样本的3个特征原始长度
            for image_id in title_group:
                data_index = prefix + image_id + suffix
                test_ = globals_ele.radgraph_iu_data[data_index]

                data_entity = globals_ele.radgraph_iu_data[data_index]['entities']
                g = self.graph_create(data_index, data_entity)
                g = self.graph_node_feature(g, data_entity)
                g = g.to(device)
                node_features = {ntype: g.nodes[ntype].data['feat'] for ntype in
                                 g.ntypes}  # node_features中保存的是原始节点token信息加上种类信息
                # 将字典中tensor提取并拼接
                node_features_concated = torch.cat(list(node_features.values()), dim=0)
                for key in node_features:
                    node_features[key] = node_features[key].to(device)
                h_dict = self.graph_conv(g, node_features)
                h_dict_concated = torch.cat(list(h_dict.values()), dim=0)
                graph_features = node_features_concated + h_dict_concated
                mean_feature = torch.mean(graph_features, dim=0).unsqueeze(0)
                graph_features = torch.cat((graph_features, mean_feature), dim=0)

                # 保存原始长度
                original_length = graph_features.size(0)
                original_lengths_group.append(original_length)
                # 填充到60
                if graph_features.size(0) < 66:
                    # 计算需要填充的数量
                    padding_needed = 66 - graph_features.size(0)
                    graph_features = F.pad(graph_features, (0, 0, 0, padding_needed), 'constant', value=0)
                group_features_list.append(graph_features)

            group_features = torch.stack(group_features_list, dim=0)
            features_list.append(group_features)

            # 保存每个group的长度信息
            original_lengths_list.append(original_lengths_group)

        # 转换为张量 (16, 3)
        original_lengths_tensor = torch.tensor(original_lengths_list,device=device)

        graph_feature = torch.stack(features_list, dim=0)   #获取的16张图片对应的3个图信息(16,3,66,512)

        if mode == 'train':
            prefix = '/data1/lijunliang/r2gen/data/iu_xray/report_txt/'
            suffix = '.txt'
            # 初始化一个空列表来收集特征向量
            features_list = []
            original_lengths_ground = []  # 用于存储每个样本的原始长度
            for image_id in images_id:
                # test = 'CXR212_IM-0746-1001'
                # data_index = prefix+test+suffix
                data_index = prefix + image_id + suffix
                test_ = globals_ele.radgraph_iu_data[data_index]

                data_entity = globals_ele.radgraph_iu_data[data_index]['entities']
                g = self.graph_create(data_index, data_entity)
                g = self.graph_node_feature(g, data_entity)
                g = g.to(device)
                node_features = {ntype: g.nodes[ntype].data['feat'] for ntype in
                                 g.ntypes}  # node_features中保存的是原始节点token信息加上种类信息
                # 将字典中tensor提取并拼接
                node_features_concated = torch.cat(list(node_features.values()), dim=0)
                for key in node_features:
                    node_features[key] = node_features[key].to(device)
                h_dict = self.graph_conv(g, node_features)
                h_dict_concated = torch.cat(list(h_dict.values()), dim=0)
                graph_features = node_features_concated + h_dict_concated
                mean_feature = torch.mean(graph_features, dim=0).unsqueeze(0)
                graph_features = torch.cat((graph_features, mean_feature), dim=0)

                # 记录原始长度
                original_length_ground = graph_features.size(0)
                original_lengths_ground.append(original_length_ground)
                # 填充到60
                if graph_features.size(0) < 66:
                    # 计算需要填充的数量
                    padding_needed = 66 - graph_features.size(0)
                    graph_features = F.pad(graph_features, (0, 0, 0, padding_needed), 'constant', value=0)
                features_list.append(graph_features)
            # 经过填充后的图特征，但是这样的做法是否有损效果有待商榷
            graph_feature_ground = torch.stack(features_list, dim=0)
            # 转换为张量
            original_lengths_tensor_ground = torch.tensor(original_lengths_ground, device=device)  # 形状为 (N,)

            #384维的图像数据转化为41维的标签分类数据
            att_feats_41 = self.Imageclassify(att_feats_384)

            output = self.encoder_decoder(fc_feats,fc_feats_sam, att_feats, att_feats_sam , targets,graph_feature,graph_feature_ground,images_id ,result,top_3_values,self.tokenizer,att_feats_384,att_feats_41,batch_report_labels,original_lengths_tensor,original_lengths_tensor_ground,mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats,fc_feats_sam, att_feats, att_feats_sam,graph_feature,result,top_3_values,self.tokenizer,original_lengths_tensor,mode='sample')
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def graph_create(self,data_index,entities):
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

        for edge_type, (src_list, dst_list) in edge_dict.items():
            edge_dict[edge_type] = (torch.tensor(src_list), torch.tensor(dst_list))

        for node_type, id_map in node_id_maps.items():
            # 获取所有节点的映射编号列表
            mapped_ids = list(id_map.values())
            # 创建自环关系的键
            self_loop_key = (node_type, 'self-loop', node_type)
            # 将源点和终点列表转换为tensor
            src_tensor = torch.tensor(mapped_ids)
            dst_tensor = torch.tensor(mapped_ids)
            edge_dict[self_loop_key] = (src_tensor,dst_tensor)
        try:
            g = dgl.heterograph(edge_dict)
        except:
            print(data_index)
        return g

    def graph_node_feature(self,g,entities):#这里根据图节点的单词赋予特征（特征来自decoder中使用的embedding）然后每种节点（OBS-DA等）拥有对应不同的线性变化层，特征被输入到对应的线性层进行处理
        # 创建token到ID的映射字典
        token_type_to_id = {}
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

            # 建立(token, 类型)到(类型, ID)的映射
            # 注意：ID是从0开始的整数，因此使用计数器的当前值
            token_type_key = (token, entity_type)
            token_type_to_id[token_type_key] = (entity_type, node_counter[entity_type])

            # 更新当前类型的节点计数器
            node_counter[entity_type] += 1

        features_by_type = {}
        for (token, node_type), (node_type, node_id) in token_type_to_id.items():
            # 获取当前token的特征向量
            feature_vector = self.get_feature_for_token(token).to(device)

            specified_layer = getattr(self.type_transforms,node_type)

            # 应用类型特定的线性变换
            transformed_feature_vector = feature_vector + specified_layer(feature_vector)#残差 保存原来的节点信息
            # 如果当前类型的节点还没有特征张量，则创建一个
            if node_type not in features_by_type:
                # 计算当前节点类型的节点数量
                num_nodes = max([id for (_, ntype), (_, id) in token_type_to_id.items() if ntype == node_type]) + 1
                # 为当前节点类型初始化特征张量
                features_by_type[node_type] = torch.zeros((num_nodes, feature_size))

            features_by_type[node_type][node_id] = transformed_feature_vector + feature_vector

        for node_type, features in features_by_type.items():
            g.nodes[node_type].data['feat'] = features
        return g


    def get_feature_for_token(self,token):
        try:
            token_id = self.tokenizer.token2idx[token.lower()]
            feature_embedding = self.encoder_decoder.model.tgt_embed[0]
            token_feature = feature_embedding(torch.tensor(token_id).to(device))
            return token_feature
        except:
            return torch.zeros(512)

    def dimension_reduce_report(self,x):
        embedding = self.reducer.fit_transform(x)