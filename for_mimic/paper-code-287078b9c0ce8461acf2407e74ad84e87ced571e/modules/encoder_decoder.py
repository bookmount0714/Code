from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json


from . import globals_ele
from .att_model import pack_wrapper, AttModel
from mia.MIA import MIA



iu_report_radgraph = '../r2gen/data/iu_xray/iu_report_radgraph.json'
device='cuda:1'

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

#iu
# def generate_mask(lengths, max_length=66):
#     mask = torch.arange(max_length,device=device).expand(len(lengths), max_length) < lengths.unsqueeze(1)
#     return mask  # 返回形状 (16, 66) 的掩码

#mimic
def generate_mask(lengths, max_length=215):
    mask = torch.arange(max_length,device=device).expand(len(lengths), max_length) < lengths.unsqueeze(1)
    return mask  # 返回形状 (16, 66) 的掩码

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Memory_encoder(nn.Module):
    def __init__(self,layer,N):
        super(Memory_encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        layer_1 = self.layers[0]
        layer_2 = self.layers[1]
        x_1 = self.norm(layer_1(x,mask))
        x_2 = layer_2(x_1,mask)
        return x_1,self.norm(x_2)

loss_func = F.mse_loss
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, rm , MIA , MemoryEncoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm
        #引入mia模块
        self.mia = MIA
        #引入记忆encoder 用于记忆分割后图像对一场区域的注意力
        self.memory_encoder = MemoryEncoder
        #引入可训练权重参数
        self.weight_src_avg = nn.Parameter(torch.tensor(1.0,requires_grad=True))
        self.weight_src = nn.Parameter(torch.tensor(1.0,requires_grad=True))
        self.weight_src_sam = nn.Parameter(torch.tensor(1.0,requires_grad=True))
        self.weight_src_new = nn.Parameter(torch.tensor(1.0,requires_grad=True))

        # self.data = self.get_json_data(iu_report_radgraph)


    # def forward(self, src, src_sam ,tgt, src_mask, graph_feature,tgt_mask):
    #
    #     #对src_sam 和 tgt 进行 mia处理
    #     src_sam_mia , _ = self.mia(src_sam,self.tgt_embed(tgt),graph_feature) #src_sam 收集到报告信息的图像信息（未经过encoder处理） src_sam_mia是src_sam经过mia接收到报告信息后，也就是说可以注意到某些重点区域的图像特征
    #
    #     src_sam_memory = self.memory_encoder(src_sam,src_mask)#由于后续测试集无法接触到报告信息 故考虑用encoder（自注意力机制）来记住需要关注的异常区域
    #     src_encoder = self.encode(src, src_mask) #16,98,512
    #     src_sam_encoder = self.encode(src_sam_mia, src_mask)
    #     # src_sam_encoder = self.encode(src_sam_memory, src_mask) #这里应该用经过memory_encoder处理的不含文字信息的图像参与报告生成
    #
    #
    #     #记忆模块输出和sam经过mia输出的损失
    #     globals_ele.loss_memory_sam = loss_func(src_sam_memory,src_sam_mia,reduction='mean')
    #
    #     src_combined = ((src_encoder + src_sam_encoder) / 2)*self.weight_src_avg.data.item() + src_encoder * self.weight_src.data.item() + src_sam_encoder * self.weight_src_sam.data.item()
    #     return self.decode(src_combined, src_mask, tgt, tgt_mask)
    #     # return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    # def forward(self, src, src_sam, tgt, src_mask, graph_feature,images_id,tgt_mask):
    #     # 对src_sam 和 tgt 进行 mia处理
    #     src_sam_mia_1, src_sam_mia_2 = self.mia(src_sam, self.tgt_embed(tgt),
    #                                             graph_feature)  # src_sam 收集到报告信息的图像信息（未经过encoder处理） src_sam_mia是src_sam经过mia接收到报告信息后，也就是说可以注意到某些重点区域的图像特征
    #
    #     src_sam_memory_1, src_sam_memory_2 = self.memory_encoder(src_sam,
    #                                                              src_mask)  # 由于后续测试集无法接触到报告信息 故考虑用encoder（自注意力机制）来记住需要关注的异常区域
    #     src_encoder = self.encode(src, src_mask)  # 16,98,512
    #     src_sam_encoder = self.encode(src_sam_mia_2, src_mask)
    #     # src_sam_encoder = self.encode(src_sam_memory, src_mask) #这里应该用经过memory_encoder处理的不含文字信息的图像参与报告生成
    #
    #     # 记忆模块输出和sam经过mia输出的损失
    #     globals_ele.loss_memory_sam = loss_func(src_sam_memory_1, src_sam_mia_1, reduction='mean') + loss_func(src_sam_memory_2, src_sam_mia_2, reduction='mean')
    #
    #     src_combined = ((src_encoder + src_sam_encoder) / 2) * self.weight_src_avg.data.item() + src_encoder * self.weight_src.data.item() + src_sam_encoder * self.weight_src_sam.data.item()
    #     return self.decode(src_combined, src_mask, tgt, tgt_mask)

    def forward(self,src, src_sam, tgt, src_mask, graph_feature,graph_feature_ground,images_id,result,top_3_values,tgt_mask,tokenizer,att_feats_384,att_feats_41,batch_report_labels,original_lengths_tensor,original_lengths_tensor_ground):
        #(16,3,66,512)的图张量，一个样本搜索到三个报告对应转化的图信息
        graph_feature_1 = graph_feature[:, 0, :, :]
        graph_feature_2 = graph_feature[:, 1, :, :]
        graph_feature_3 = graph_feature[:, 2, :, :]
        # 分别获取每个图特征的原始长度
        original_lengths_1 = original_lengths_tensor[:, 0]  # (16,)
        original_lengths_2 = original_lengths_tensor[:, 1]  # (16,)
        original_lengths_3 = original_lengths_tensor[:, 2]  # (16,)
        # 为 graph_feature_1, graph_feature_2, graph_feature_3 生成掩码
        mask_1 = generate_mask(original_lengths_1).unsqueeze(1)  # (16, 66)
        mask_2 = generate_mask(original_lengths_2).unsqueeze(1)  # (16, 66)
        mask_3 = generate_mask(original_lengths_3).unsqueeze(1)  # (16, 66)
        mask_ground = generate_mask(original_lengths_tensor_ground).unsqueeze(1)
        #获取三个报告对应的下标，从tokenizer下手

        # 初始化三个列表来存储每个位置的报告
        first_reports = []
        second_reports = []
        third_reports = []
        #iu
        # 遍历每个样本
        for i in range(result.shape[0]):
            report_indices = result[i]  # 获取第 i 个样本的报告索引
            first_reports.append(globals_ele.reports_ann[report_indices[0]])  # 第一个下标对应的报告
            second_reports.append(globals_ele.reports_ann[report_indices[1]])  # 第二个下标对应的报告
            third_reports.append(globals_ele.reports_ann[report_indices[2]])  # 第三个下标对应的报告
        tokenized_first_reports = [tokenizer(text)[:60] for text in first_reports]
        tokenized_second_reports = [tokenizer(text)[:60] for text in second_reports]
        tokenized_third_reports = [tokenizer(text)[:60] for text in third_reports]
        tokenized_first_reports = torch.tensor([inner_list + [0] * (60 - len(inner_list)) for inner_list in tokenized_first_reports]).to(device)[:, :-1]
        tokenized_second_reports = torch.tensor([inner_list + [0] * (60 - len(inner_list)) for inner_list in tokenized_second_reports]).to(device)[:, :-1]
        tokenized_third_reports = torch.tensor([inner_list + [0] * (60 - len(inner_list)) for inner_list in tokenized_third_reports]).to(device)[:, :-1]


        #加入memory模块
        src_sam_mia_ground_1, src_sam_mia_ground_2 = self.mia(src_sam, self.tgt_embed(tgt),
                                                graph_feature_ground,mask_ground)  # src_sam 收集到报告信息的图像信息（未经过encoder处理） src_sam_mia是src_sam经过mia接收到报告信息后，也就是说可以注意到某些重点区域的图像特征
        src_sam_memory_1, src_sam_memory_2 = self.memory_encoder(src_sam,
                                                                 src_mask)  # 由于后续测试集无法接触到报告信息 故考虑用encoder（自注意力机制）来记住需要关注的异常区域
        src_sam_encoder = self.encode(src_sam_mia_ground_2, src_mask)
        globals_ele.loss_memory_sam = loss_func(src_sam_memory_1, src_sam_mia_ground_1, reduction='mean') + loss_func(
            src_sam_memory_2, src_sam_mia_ground_2, reduction='mean')


        #经过三个图信息强化过的三个sam图像
        _,src_sam_mia_1 = self.mia(src_sam,self.tgt_embed(tokenized_first_reports),graph_feature_1,mask_1)
        _,src_sam_mia_2 = self.mia(src_sam,self.tgt_embed(tokenized_second_reports),graph_feature_2,mask_2)
        _,src_sam_mia_3 = self.mia(src_sam,self.tgt_embed(tokenized_third_reports),graph_feature_3,mask_3)

        src_sam_mia_1_encoder = self.encode(src_sam_mia_1, src_mask)
        src_sam_mia_2_encoder = self.encode(src_sam_mia_2, src_mask)
        src_sam_mia_3_encoder = self.encode(src_sam_mia_3, src_mask)
        src_encoder = self.encode(src, src_mask)  # 16,98,512

        top_3_values_ = top_3_values
        # 将top3_values扩展为 (16, 3, 1, 1) 以便进行广播运算
        top_3_values = top_3_values.unsqueeze(-1).unsqueeze(-1)
        # 将张量合并为 (16, 3, 98, 512) 的形状
        sam_all = torch.stack((src_sam_mia_1_encoder, src_sam_mia_2_encoder, src_sam_mia_3_encoder), dim=1)
        # 使用top3_values进行加权组合
        sam_new = torch.sum(sam_all * top_3_values, dim=1)  # 形状为 (16, 98, 512)

        # src_combined = src_encoder*self.weight_src+sam_new
        src_combined = ((src_encoder + src_sam_encoder + sam_new) / 3) * self.weight_src_avg + src_encoder * self.weight_src + src_sam_encoder * self.weight_src_sam + sam_new * self.weight_src_new
        #以下损失是每张图片择取的三个报告bert处理后张量与图片对应基准报告384张量之间的损失
        losses = 0
        for i, image_id in enumerate(images_id):
            # 从 report_dict 中获取对应 image_id 的基准张量
            ground_truth_tensor = globals_ele.report_dict[image_id]
            index = result[i,:]
            predicted_tensor_1 = globals_ele.reports_tensor_ann[index[0]]
            predicted_tensor_2 = globals_ele.reports_tensor_ann[index[1]]
            predicted_tensor_3 = globals_ele.reports_tensor_ann[index[2]]
            loss_1 = loss_func(predicted_tensor_1,ground_truth_tensor)
            loss_2 = loss_func(predicted_tensor_2,ground_truth_tensor)
            loss_3 = loss_func(predicted_tensor_3,ground_truth_tensor)
            values = top_3_values_[i]
            losses = losses + (values[0] * loss_1 + values[1] * loss_2 + values[2] * loss_3)
        globals_ele.loss_selected_report = losses/ len(images_id)

        #图片化为384维后与对应报告之间的损失
        tensor_list = []
        for i,image_id in enumerate(images_id):
            tensor_list.append(globals_ele.report_dict[image_id])
        tensor_data = torch.stack(tensor_list).to(device)#报告对应384维张量
        globals_ele.loss_image_report_384 = loss_func(att_feats_384,tensor_data)

        #对文本分类以及图像分类进行损失计算 att_feats_41是图像的分类 batch_report_labels是报告对应标签
        loss_func_bce = nn.CrossEntropyLoss()
        globals_ele.loss_classify = loss_func_bce(att_feats_41,batch_report_labels)

        return self.decode(src_combined, src_mask, tgt, tgt_mask)



    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def memory_encode(self,src,mask):#获取src_sam 经过记忆模块的输出
        return self.memory_encoder(src,mask)
    def decode(self, hidden_states, src_mask, tgt, tgt_mask):
        # memory = self.rm.init_memory(hidden_states.size(0)).to(hidden_states)
        # memory = self.rm(self.tgt_embed(tgt), memory)
        memory = None
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, memory)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask, memory)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots, rm_d_model):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # self.sublayer = clones(ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        m = hidden_states
        # x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory)
        # x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # return self.sublayer[2](x, self.feed_forward, memory)
        return self.sublayer[2](x, self.feed_forward)


# class ConditionalSublayerConnection(nn.Module):
#     def __init__(self, d_model, dropout, rm_num_slots, rm_d_model):
#         super(ConditionalSublayerConnection, self).__init__()
#         self.norm = ConditionalLayerNorm(d_model, rm_num_slots, rm_d_model)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, sublayer, memory):
#         return x + self.dropout(sublayer(self.norm(x, memory)))
#
#
# class ConditionalLayerNorm(nn.Module):
#     def __init__(self, d_model, rm_num_slots, rm_d_model, eps=1e-6):
#         super(ConditionalLayerNorm, self).__init__()
#         self.gamma = nn.Parameter(torch.ones(d_model))
#         self.beta = nn.Parameter(torch.zeros(d_model))
#         self.rm_d_model = rm_d_model
#         self.rm_num_slots = rm_num_slots
#         self.eps = eps
#
#         self.mlp_gamma = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
#                                        nn.ReLU(inplace=True),
#                                        nn.Linear(rm_d_model, rm_d_model))
#
#         self.mlp_beta = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
#                                       nn.ReLU(inplace=True),
#                                       nn.Linear(d_model, d_model))
#
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0.1)
#
#     def forward(self, x, memory):
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         delta_gamma = self.mlp_gamma(memory)
#         delta_beta = self.mlp_beta(memory)
#         gamma_hat = self.gamma.clone()
#         beta_hat = self.beta.clone()
#         gamma_hat = torch.stack([gamma_hat] * x.size(0), dim=0)
#         gamma_hat = torch.stack([gamma_hat] * x.size(1), dim=1)
#         beta_hat = torch.stack([beta_hat] * x.size(0), dim=0)
#         beta_hat = torch.stack([beta_hat] * x.size(1), dim=1)
#         gamma_hat += delta_gamma
#         beta_hat += delta_beta
#         return gamma_hat * (x - mean) / (std + self.eps) + beta_hat


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

#
# class RelationalMemory(nn.Module):
#
#     def __init__(self, num_slots, d_model, num_heads=1):
#         super(RelationalMemory, self).__init__()
#         self.num_slots = num_slots
#         self.num_heads = num_heads
#         self.d_model = d_model
#
#         self.attn = MultiHeadedAttention(num_heads, d_model)
#         self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
#                                  nn.ReLU(),
#                                  nn.Linear(self.d_model, self.d_model),
#                                  nn.ReLU())
#
#         self.W = nn.Linear(self.d_model, self.d_model * 2)
#         self.U = nn.Linear(self.d_model, self.d_model * 2)
#
#     def init_memory(self, batch_size):
#         memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
#         if self.d_model > self.num_slots:
#             diff = self.d_model - self.num_slots
#             pad = torch.zeros((batch_size, self.num_slots, diff))
#             memory = torch.cat([memory, pad], -1)
#         elif self.d_model < self.num_slots:
#             memory = memory[:, :, :self.d_model]
#
#         return memory
#
#     def forward_step(self, input, memory):
#         memory = memory.reshape(-1, self.num_slots, self.d_model)
#         q = memory
#         k = torch.cat([memory, input.unsqueeze(1)], 1)
#         v = torch.cat([memory, input.unsqueeze(1)], 1)
#         next_memory = memory + self.attn(q, k, v)
#         next_memory = next_memory + self.mlp(next_memory)
#         next_memory = next_memory + self.mlp(next_memory)
#
#         gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
#         gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
#         input_gate, forget_gate = gates
#         input_gate = torch.sigmoid(input_gate)
#         forget_gate = torch.sigmoid(forget_gate)
#
#         next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
#         next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)
#
#         return next_memory
#
#     def forward(self, inputs, memory):
#         outputs = []
#         for i in range(inputs.shape[1]):
#             memory = self.forward_step(inputs[:, i], memory)
#             outputs.append(memory)
#         outputs = torch.stack(outputs, dim=1)
#         return outputs


class EncoderDecoder(AttModel):

    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = None
        # rm = RelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(
                DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots, self.rm_d_model),
                self.num_layers),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            rm,
            MIA(),
            Memory_encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), 2))
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.rm_num_slots = args.rm_num_slots
        self.rm_num_heads = args.rm_num_heads
        self.rm_d_model = args.rm_d_model

        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, fc_feats_sam ,att_feats, att_feats_sam, att_masks,graph_feature,result,top_3_values,tokenizer,original_lengths_tensor):

        att_masks_sam = att_masks
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        att_feats_sam,_,_,_ = self._prepare_feature_forward(att_feats_sam, att_masks_sam)

        graph_feature_1 = graph_feature[:, 0, :, :]
        graph_feature_2 = graph_feature[:, 1, :, :]
        graph_feature_3 = graph_feature[:, 2, :, :]

        # 分别获取每个图特征的原始长度
        original_lengths_1 = original_lengths_tensor[:, 0]  # (16,)
        original_lengths_2 = original_lengths_tensor[:, 1]  # (16,)
        original_lengths_3 = original_lengths_tensor[:, 2]  # (16,)
        # 为 graph_feature_1, graph_feature_2, graph_feature_3 生成掩码
        mask_1 = generate_mask(original_lengths_1).unsqueeze(1)  # (16, 66)
        mask_2 = generate_mask(original_lengths_2).unsqueeze(1)  # (16, 66)
        mask_3 = generate_mask(original_lengths_3).unsqueeze(1)  # (16, 66)

        # 初始化三个列表来存储每个位置的报告
        first_reports = []
        second_reports = []
        third_reports = []
        # 遍历每个样本
        for i in range(result.shape[0]):
            report_indices = result[i]  # 获取第 i 个样本的报告索引
            first_reports.append(globals_ele.reports_ann[report_indices[0]])  # 第一个下标对应的报告
            second_reports.append(globals_ele.reports_ann[report_indices[1]])  # 第二个下标对应的报告
            third_reports.append(globals_ele.reports_ann[report_indices[2]])  # 第三个下标对应的报告
        tokenized_first_reports = [tokenizer(text)[:60] for text in first_reports]
        tokenized_second_reports = [tokenizer(text)[:60] for text in second_reports]
        tokenized_third_reports = [tokenizer(text)[:60] for text in third_reports]
        tokenized_first_reports = torch.tensor([inner_list + [0] * (60 - len(inner_list)) for inner_list in tokenized_first_reports]).to(device)[:, :-1]
        tokenized_second_reports = torch.tensor([inner_list + [0] * (60 - len(inner_list)) for inner_list in tokenized_second_reports]).to(device)[:, :-1]
        tokenized_third_reports = torch.tensor([inner_list + [0] * (60 - len(inner_list)) for inner_list in tokenized_third_reports]).to(device)[:, :-1]


        # 经过三个图信息强化过的三个sam图像
        _,src_sam_mia_1 = self.model.mia(att_feats_sam, self.model.tgt_embed(tokenized_first_reports), graph_feature_1,mask_1)
        _,src_sam_mia_2 = self.model.mia(att_feats_sam, self.model.tgt_embed(tokenized_second_reports), graph_feature_2,mask_2)
        _,src_sam_mia_3 = self.model.mia(att_feats_sam, self.model.tgt_embed(tokenized_third_reports), graph_feature_3,mask_3)

        src_sam_mia_1_encoder = self.model.encode(src_sam_mia_1, att_masks)
        src_sam_mia_2_encoder = self.model.encode(src_sam_mia_2, att_masks)
        src_sam_mia_3_encoder = self.model.encode(src_sam_mia_3, att_masks)

        memory = self.model.encode(att_feats, att_masks)

        _,memory_encoded = self.model.memory_encoder(att_feats_sam,att_masks)
        memory_encoded_encoded = self.model.encode(memory_encoded,att_masks)

        # src_encoder = self.model.encode(att_feats, att_masks)  # 16,98,512

        # 将top3_values扩展为 (16, 3, 1, 1) 以便进行广播运算
        top_3_values = top_3_values.unsqueeze(-1).unsqueeze(-1)
        # 将张量合并为 (16, 3, 98, 512) 的形状
        sam_all = torch.stack((src_sam_mia_1_encoder, src_sam_mia_2_encoder, src_sam_mia_3_encoder), dim=1)
        # 使用top3_values进行加权组合
        sam_new = torch.sum(sam_all * top_3_values, dim=1)  # 形状为 (16, 98, 512)

        # combined = memory*self.model.weight_src+sam_new
        combined = ((memory + memory_encoded_encoded + sam_new) / 3) * self.model.weight_src_avg.data.item() + memory * self.model.weight_src.data.item() + memory_encoded_encoded * self.model.weight_src_sam.data.item() + sam_new * self.model.weight_src_new.data.item()


        # combined = self.model.weight_src_avg.data.item() * ((memory+memory_encoded_encoded)/2) + self.model.weight_src.data.item() * memory + self.model.weight_src_sam.data.item() * memory_encoded_encoded


        # return fc_feats[..., :1], att_feats[..., :1], memory, att_masks
        return fc_feats[..., :1], att_feats[..., :1], combined, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, fc_feats_sam, att_feats,att_feats_sam ,seq,graph_feature,graph_feature_ground,images_id,result,top_3_values,tokenizer,att_feats_384,att_feats_41,batch_report_labels,original_lengths_tensor,original_lengths_tensor_ground,att_masks=None):
        #处理att_feats_sam
        att_feats_sam, _, _, _ = self._prepare_feature_forward(att_feats_sam, att_masks, seq)

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        out = self.model(att_feats, att_feats_sam ,seq, att_masks,graph_feature,graph_feature_ground,images_id,result,top_3_values,seq_mask,tokenizer,att_feats_384,att_feats_41,batch_report_labels,original_lengths_tensor,original_lengths_tensor_ground)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):

        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0].to(memory.device), it.unsqueeze(1).to(memory.device)], dim=1)
        out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]
