'''
Author: mzcai
Date: 2022-02-09 20:48:26
Description: 
FilePath: /mzcai/cross_model_DT/cross_model_DT/models.py
'''

# import os
# import sys

# import torch.nn as nn
# from transformers import BertModel
# from loguru import logger
# import config
# import torch


# base_dir = os.path.dirname(__file__)
# sys.path.append(base_dir)





# class BaseModel(nn.Module):
#     def __init__(self, _config):

#         super(BaseModel, self).__init__()

#         if config.bert_dir:
#             bert_dir = os.path.join(base_dir, config.bert_dir)
#             self.bert_module = BertModel.from_pretrained(bert_dir)
#         else:
#             self.bert_module = BertModel.from_pretrained(config.bert_name)
#         self.bert_config = self.bert_module.config

#     @staticmethod
#     def _init_weights(blocks, **kwargs):
#         for block in blocks:
#             for module in block.modules():
#                 if isinstance(module, nn.Linear):
#                     nn.init.zeros_(module.bias)
#                 elif isinstance(module, nn.Embedding):
#                     nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
#                 elif isinstance(module, nn.LayerNorm):
#                     nn.init.zeros_(module.bias)
#                     nn.init.ones_(module.weight)

# class cross_Model(BaseModel):
#     def __init__(self,
#                  _config, ):

#         super(cross_Model, self).__init__(_config)

#         # 扩充词表故需要重定义
#         self.bert_module.resize_token_embeddings(_config.len_of_tokenizer)
#         out_dims = self.bert_config.hidden_size
#         mid_linear_dims = _config.mid_linear_dims

#         # 下游任务模型结构构建
#         self.mid_linear = nn.Sequential(
#             nn.Linear(out_dims, mid_linear_dims),
#             nn.ReLU(),
#             nn.Dropout(_config.dropout_prob)
#         )
#         self.classifier = nn.Linear(mid_linear_dims, 1)
#         self.activation = nn.Sigmoid()

#         # 模型初始化
#         init_blocks = [self.mid_linear, self.classifier]
#         self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

#     def forward(self,
#                 input_ids,
#                 attention_mask,
#                 position_ids):
#         out = self.bert_module(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids = position_ids, 
#             return_dict=False
#         )
#         #print("out.last_hidden_state:",out[0].shape,out[1].shape)
#         # out = out.last_hidden_state[:, 0, :]  # 取cls对应的embedding
#         # out = self.mid_linear(out)
#         # out = self.activation(self.classifier(out))
#         '''
#         out[0]:torch.Size([32, 256, 768]) 
#         out[1]:torch.Size([32, 768])
#         '''
#         # print("out",out)
#         # print("out[0]",out[0],out[0].shape)
#         # print("out[1]",out[1],out[1].shape)
#         return out[0]




import os
import sys

import torch.nn as nn
from transformers import BertModel
from loguru import logger
import config
import torch
import math

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)




class BaseModel(nn.Module):
    def __init__(self, _config):

        super(BaseModel, self).__init__()
        
        if _config.bert_dir:
            bert_dir = os.path.join(base_dir, config.bert_dir)
            self.bert = BertModel.from_pretrained(bert_dir)
        else:
            self.bert = BertModel.from_pretrained(config.bert_name)
        self.bert_config = self.bert.config

    @staticmethod
    def _init_weights(blocks, **kwargs):
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.zeros_(module.bias)
                    nn.init.ones_(module.weight)

class cross_Model(BaseModel):
    def __init__(self,
                 _config,
                 num_tasks):

        super(cross_Model, self).__init__(_config)

        # 扩充词表故需要重定义
        self.bert.resize_token_embeddings(_config.len_of_tokenizer)
        out_dims = self.bert_config.hidden_size #768
        #mid_linear_dims = _config.mid_linear_dims #128

        # 下游任务模型结构构建
        # self.mid_linear = nn.Sequential(  #[768,128]
        #     nn.Linear(out_dims, mid_linear_dims),
        #     nn.ReLU(),
        #     nn.Dropout(_config.dropout_prob)
        # )
        self.classifier = nn.Linear(out_dims, num_tasks)
        
        
        
        self.criterion = nn.BCELoss()


        # 模型初始化
        init_blocks = [ self.classifier]
        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self,
                input_ids,
                attention_mask
                ):
        #import pdb; pdb.set_trace()
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        out = out.last_hidden_state[:, 0, :]  # 取cls对应的embedding  [batch_size,out_dims]
        #out = self.mid_linear(out)  #[batch_size,128]
        #out = self.activation(self.classifier(out))
        out = self.classifier(out) #[batch_size, num_tasks] (未激活)

        # if self.criterion:
        #     loss = self.criterion(out, labels)

        #     return out, loss

        return out

#----------------------------------------模型尝试------------------------------------

# class Adjacency_embedding(nn.Module):
#     def __init__(self, input_dim, model_dim, bias=True):
#         super(Adjacency_embedding, self).__init__()

#         self.weight_h = nn.Parameter(torch.Tensor(input_dim, model_dim))
#         self.weight_a = nn.Parameter(torch.Tensor(input_dim))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(model_dim))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight_h.size(1))
#         stdv2 = 1. /math.sqrt(self.weight_a.size(0))
#         self.weight_h.data.uniform_(-stdv, stdv)
#         self.weight_a.data.uniform_(-stdv2, stdv2)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, input_mat):
#         a_w = torch.matmul(input_mat, self.weight_h)
#         out = torch.matmul(a_w.transpose(1,2), self.weight_a)

#         if self.bias is not None:
#             out += self.bias
#         # out.shape : [batch_size, embed_size]
#         return out


# class Smiles_embedding(nn.Module):
#     def __init__(self, vocab_size, embed_size, max_len, adj=False):
#         super().__init__()
#         self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0)
#         self.position = nn.Embedding(max_len, embed_size)
#         self.max_len = max_len
#         self.embed_size = embed_size
        
#         if adj:
#                 self.adj = Adjacency_embedding(max_len, embed_size)

#     def forward(self, sequence, pos_num, adj_mat=None):
#         x = self.token(sequence) + self.position(pos_num)
#         if adj_mat is not None:
#             # additional embedding matrix. need to modify
#             x += self.adj(adj_mat).repeat(1, self.max_len).reshape(-1,self.max_len, self.embed_size)
        
#         # x.shape : [batch_size, max_len, embed_size]
#         return x

# class C_Smiles_BERT(nn.Module):
#     def __init__(self, vocab_size, max_len=256, feature_dim=768, nhead=16, feedforward_dim=1024, nlayers=12, dropout_rate=0, adj=False, num_tasks=None):
#         super(C_Smiles_BERT, self).__init__()
#         self.embedding = Smiles_embedding(vocab_size, feature_dim, max_len, adj=adj)
        
#         trans_layer = nn.TransformerEncoderLayer(feature_dim, nhead, feedforward_dim, activation='gelu', dropout=dropout_rate)
#         self.transformer_encoder = nn.TransformerEncoder(trans_layer, nlayers)
        
#         #self.classifier = nn.Linear(feature_dim, num_tasks)

#     def forward(self, src, pos_num, adj_mat=None):
#         mask = (src == 0)
#         mask = mask.type(torch.bool)

#         #import pdb;pdb.set_trace()
#         x = self.embedding(src, pos_num, adj_mat)
#         x = self.transformer_encoder(x.transpose(1,0), src_key_padding_mask=mask)
#         x = x.transpose(1,0)
#         #x = x[:, 0, :] #[batch_size, num_tasks]
        
#         #x = self.classifier(x) #[batch_size, num_tasks]
        
#         return x

# class BERT_base(nn.Module):
#     def __init__(self, model, output_layer):
#         super().__init__()
#         self.bert = model
#         self.linear = output_layer
#     def forward(self,src, pos_num, adj_mat=None):
#         x = self.bert(src, pos_num, adj_mat)
#         x = self.linear(x)
#         return x
