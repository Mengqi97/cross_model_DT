'''
Author: mzcai
Date: 2022-02-08 16:14:43
Description: 配置文件
FilePath: \cross_model_DT\cross_model_DT\config.py
'''

# 训练参数
max_seq_len = 256
mlm_prob = 0.3
drug_name_replace_prob = 0.6
mid_linear_dims = 128
dropout_prob = 0.1
lr = 5e-5
other_lr = 5e-5
weight_decay = 0
other_weight_decay = 0
adam_epsilon = 1e-8
warmup_proportion = 0
embed_size = 768


#目录
bert_dir = 'models/PubMedBERT_abstract'
data_dir = 'data'
out_model_dir = 'models'
spe_file = 'SPE_ChEMBL.txt'
spe_voc_file = 'spe_voc.txt'
saved_model = './pretrained_model/model_4epoch_only_single_0.6_fix.pt'
gpu_ids = '0'

smi_token_id = 28895
len_of_tokenizer = 28895 + 3117 + 1

# 类型选择

bert_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'

