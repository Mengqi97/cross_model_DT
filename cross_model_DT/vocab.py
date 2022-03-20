'''
Author: mzcai
Date: 2022-02-09 15:06:20
Description: 创建词表，为方便调用，从SMILES_PE源码中(data_utils)抽取出来
FilePath: \cross_model_DT\cross_model_DT\vocab.py
'''
import config
import os
import collections

base_dir = os.path.dirname(__file__)

'''
description: @function load_vocab copy from cross_model_DT\data_utils.py
param {*} vocab_file
return {*} vocab_dict
'''
def load_vocab(vocab_file):
    """
    读取静态存储的spe切分字典，该字典的token的id为了与PubMedBERT的字典合并，整体进行了偏移。并且Special Token的ID采用和PubMedBERT相同
    :param vocab_file: 字典的路径
    :return:
      vocab (:obj:'dict', )({token:id, ...}):
    """
    vocab = collections.OrderedDict()  # OrderedDict是为了兼容py3.6之前的python，这时的python字典是无序的。
    with open(vocab_file, "r", encoding="utf-8") as reader:
        lines = reader.readlines()
    for line in lines:
        token, index = line.split(' ')
        vocab[token] = int(index.rstrip("\n"))
    return vocab


'''
[PAD] 0
[UNK] 1
[CLS] 2
[SEP] 3
[MASK] 4
'''
class SPE_vocab():
    PAD_TAG = "[PAD]"
    UNK_TAG = "[UNK]"
    CLS_TAG = "[CLS]"
    SEP_TAG = "[SEP]"
    MASK_TAG = "[MASK]"
    
    PAD = 0
    UNK = 1
    CLS = 2
    SEP = 3
    MASK = 4
    

    def __init__(self):
        self.vocab = load_vocab(
            os.path.join(
                base_dir,
                config.data_dir,
                config.spe_voc_file
            )
        )
