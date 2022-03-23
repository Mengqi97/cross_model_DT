'''
Author: mzcai
Date: 2022-02-07 10:46:23
Description: cross_model下游任务训练
FilePath: /mzcai/cross_model_DT/cross_model_DT/molenet_finetune.py
'''

import numpy as np 
import torch
import glob
import pandas as pd
import random
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import re
from tqdm import tqdm
from rdkit.Chem import AllChem, Lipinski
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit import Chem
from sklearn.metrics import roc_auc_score
from models import cross_Model
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader, SequentialSampler


from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict


from data_utils import SMILES_SPE_Tokenizer
import config
from vocab import SPE_vocab
from transformers import BertTokenizer
import os
from loguru import logger
from data_utils import load_model_and_parallel
###### from models import C_Smiles_BERT, BERT_base


base_dir = os.path.dirname(__file__)

def generate_scaffold(smiles, include_chirality=False):
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    return scaffold

def scaffold_split(smiles):
    all_scaffolds = {}
    count = 0
    for i in range(len(smiles)):
        if Chem.MolFromSmiles(smiles[i]) != None:
            scaffold = generate_scaffold(smiles[i], include_chirality=True)
            if scaffold not in all_scaffolds:
                all_scaffolds[scaffold] = [i]
            else:
                all_scaffolds[scaffold].append(i)
        else:
            count += 1
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in sorted(all_scaffolds.items(), 
                        key=lambda x: (len(x[1]), x[1][0]), reverse=True)]
    train_cutoff = 0.8 * (len(smiles)-count)
    valid_cutoff = 0.9 * (len(smiles)-count)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)
    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    return train_idx, valid_idx, test_idx


def _init_seed_fix(manualSeed):
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class FinetuningDataset(Dataset):
    def __init__(self, datapath, data_name, SPE_vocab, seq_len):
        #加载预训练好的tokenizer
        self.tokenizer_smi = SMILES_SPE_Tokenizer(
                vocab_file=os.path.join(base_dir, config.data_dir, config.spe_voc_file),
                spe_file=os.path.join(base_dir, config.data_dir, config.spe_file))
        
        self.smiles_dataset = []
        self.adj_dataset = []

        self.seq_len = seq_len

        smiles_data = glob.glob(datapath + "/*" + data_name + ".csv")
        
        text = pd.read_csv(smiles_data[0], sep=',')
        
        #读取数据中smiles、label信息
        if data_name == 'tox21':
            tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
           'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
            smiles_list = np.asarray(text['smiles'])
            label_list = text[tasks]
            label_list = label_list.replace(0,-1)
            label_list = label_list.fillna(0)
        elif data_name == 'bace':
            smiles_list = np.asarray(text['smiles'])
            label_list = text['Class']
            label_list = label_list.replace(0, -1)
        elif data_name == 'bbbp':
            smiles_list = np.asarray(text['smiles'])
            label_list = text['p_np']
            label_list = label_list.replace(0, -1)
        elif data_name == 'clintox':
            smiles_list = np.asarray(text['smiles'])
            tasks = ['FDA_APPROVED', 'CT_TOX']
            label_list = text[tasks]
            label_list = label_list.replace(0, -1)
        elif data_name == 'sider':
            smiles_list = np.asarray(text['smiles'])
            tasks = ['Hepatobiliary disorders',
                       'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
                    'Investigations', 'Musculoskeletal and connective tissue disorders',
                    'Gastrointestinal disorders', 'Social circumstances',
                    'Immune system disorders', 'Reproductive system and breast disorders',
                    'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                    'General disorders and administration site conditions',
                    'Endocrine disorders', 'Surgical and medical procedures',
                    'Vascular disorders', 'Blood and lymphatic system disorders',
                    'Skin and subcutaneous tissue disorders',
                    'Congenital, familial and genetic disorders',
                    'Infections and infestations',
                    'Respiratory, thoracic and mediastinal disorders',
                    'Psychiatric disorders', 'Renal and urinary disorders',
                    'Pregnancy, puerperium and perinatal conditions',
                    'Ear and labyrinth disorders', 'Cardiac disorders',
                    'Nervous system disorders',
                    'Injury, poisoning and procedural complications']
            label_list = text[tasks]
            label_list = label_list.replace(0, -1)
        elif data_name == 'toxcast':
            smiles_list = np.asarray(text['smiles'])
            tasks = list(text.columns[1:])
            label_list = text[tasks]
            label_list = label_list.replace(0, -1)
            label_list = label_list.fillna(0)
        elif data_name == 'muv':
            smiles_list = np.asarray(text['smiles'])
            tasks = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
                    'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
                    'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
            label_list = text[tasks]
            label_list = label_list.replace(0, -1)
            label_list = label_list.fillna(0)
        elif data_name == 'hiv':
            smiles_list = np.asarray(text['smiles'])
            label_list = text['HIV_active']
            label_list = label_list.replace(0, -1)

        self.label = np.asarray(label_list)
        
        for i in smiles_list:
            self.adj_dataset.append(i)
            single_smi_token = self.tokenizer_smi.tokenize(i)
            self.smiles_dataset.append(single_smi_token)
        
        #print("adj_dataset:",self.adj_dataset)
        #print("smiles_dataset:",self.smiles_dataset)
        #print("label_list:",label_list.shape)

    def __len__(self):
        return len(self.smiles_dataset)

    def __getitem__(self, idx):
        
        item = self.adj_dataset[idx]
        label = self.label[idx]

        inputs = self.tokenizer_smi(item, padding='max_length', truncation=True, max_length=config.max_seq_len,
                                    return_attention_mask=True)


        ##### mol = Chem.MolFromSmiles(self.adj_dataset[idx])
        ##### 
        ##### if mol != None:
        #####     adj_mat = GetAdjacencyMatrix(mol)
        #####     smiles_bert_adjmat = self.zero_padding(adj_mat, (self.seq_len, self.seq_len))
        ##### else:
        #####     smiles_bert_adjmat = np.zeros((self.seq_len, self.seq_len), dtype=np.float32)

        output = {	'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask'],
                    'labels': label,
                    ##### 'smiles_bert_adjmat': smiles_bert_adjmat
        }
        #print('input_ids:', output['input_ids'],output['labels'].shape ,end = "\n" )
        #print('attention_mask:', output['attention_mask'],output['labels'].shape ,end = "\n")
        #print('labels:', output['labels'],output['labels'].shape)
        return {key:torch.tensor(value) for key, value in output.items()}

    def zero_padding(self, array, shape):
        if array.shape[0] > shape[0]:
            array = array[:shape[0],:shape[1]]
        padded = np.zeros(shape, dtype=np.float32)
        padded[:array.shape[0], :array.shape[1]] = array
        return padded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="dataset path", type=str, default = None)
    parser.add_argument('--dataset', help="name of dataset", type=str)
    parser.add_argument('--batch', help="batch size", type=int, default=16)
    parser.add_argument('--epoch', help="epoch", type=int, default=100)
    parser.add_argument('--seq', help="sequence length", type=int, default=256)
    parser.add_argument('--lr', help="learning rate", type=float, default=3e-5)
    parser.add_argument('--adjacency', help="use adjacency matrix", type=bool, default=True)
    #parser.add_argument('--embed_size', help="embedding vector size", type=int, default=768)
    parser.add_argument('--model_dim', help="dim of transformer", type=int, default=1024)
    parser.add_argument('--layers', help="number of layers", type=int, default=12)
    parser.add_argument('--nhead', help="number of head", type=int, default=12)
    parser.add_argument('--drop_rate', help="ratio of dropout", type=float, default=0)
    parser.add_argument('--num_workers', help="number of workers", type=int, default=0)
    #parser.add_argument('--saved_model', help="dir of pre-trained model", type=str)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument('--split', help="type of dataset", type=str, default='scaffold')
    arg = parser.parse_args()

    _init_seed_fix(arg.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    print("learning_rate:", arg.lr)

    #----------------------------------------------1、数据准备---------------------------------------------------------------------
    if arg.dataset == "tox21":
        num_tasks = 12
    elif arg.dataset == "bace":
        num_tasks = 1
    elif arg.dataset == 'bbbp':
        num_tasks = 1
    elif arg.dataset == 'clintox':
        num_tasks = 2
    elif arg.dataset == 'sider':
        num_tasks = 27
    elif arg.dataset == 'toxcast':
        num_tasks = 617
    elif arg.dataset == 'muv':
        num_tasks = 17
    elif arg.dataset == 'hiv':
        num_tasks = 1

    Smiles_vocab = SPE_vocab()
    
    #read data
    dataset = FinetuningDataset(arg.path, arg.dataset, Smiles_vocab, seq_len=arg.seq)

    logger.info("Dataset loaded")
    
    if arg.split == 'scaffold':
        smiles_csv = pd.read_csv(arg.path+"/"+arg.dataset+".csv", sep=',')
        smiles_list = smiles_csv['smiles'].tolist()
        
        train_idx, valid_idx, test_idx = scaffold_split(smiles_list)
    else:
        indices = list(range(len(dataset)))
        split1, split2 = int(np.floor(0.1 * len(dataset))), int(np.floor(0.2 * len(dataset)))
        #np.random.seed(arg.seed)
        np.random.shuffle(indices)
        train_idx, valid_idx, test_idx = indices[split2:], indices[split1:split2], indices[:split1]


    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # preprocessing - dataloader(train, valid, test)
    train_dataloader = DataLoader(dataset, batch_size=arg.batch, sampler=train_sampler,num_workers=arg.num_workers, pin_memory=True)####为了方便调试修改num_workers=arg.num_workers
    valid_dataloader = DataLoader(dataset, batch_size=arg.batch, sampler=valid_sampler,num_workers=arg.num_workers, pin_memory=True)
    test_dataloader = DataLoader(dataset, batch_size=arg.batch, sampler=test_sampler, num_workers=arg.num_workers, pin_memory=True)

    #------------------------------------------------------2、模型加载-----------------------------------------------------------
    logger.info("loading model...")
    model = cross_Model(_config=config, num_tasks=num_tasks)
    
    ##### model = C_Smiles_BERT(config.len_of_tokenizer, max_len=arg.seq, nhead=arg.nhead, feature_dim=config.embed_size, feedforward_dim=arg.model_dim, nlayers=arg.layers, dropout_rate=arg.drop_rate, num_tasks=num_tasks, adj=arg.adjacency)
    model.load_state_dict(torch.load(config.chem_bert_model), strict = False)
    ##### classifier = nn.Linear(config.embed_size, num_tasks)
    ##### model = BERT_base(model, classifier)

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optim = Adam(model.parameters(), lr=arg.lr, weight_decay=0)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    logger.info("Start fine-tuning with seed:{}".format(arg.seed))
    min_valid_loss = 100000
    counter = 0

    #------------------------------------------------------3、模型训练----------------------------------------------------------------
    logger.info("Start training...")
    for epoch in range(arg.epoch):
        avg_loss = 0
        valid_avg_loss = 0

        #********************************3.1、模型train集训练*****************************************
        data_iter = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        
        model.train()
        
        for i, data in data_iter:
            data = {key:value.to(device) for key, value in data.items()}
            
            output = model.forward(data["input_ids"], data["attention_mask"])
            
            ##### position_num = torch.arange(arg.seq).repeat(data["input_ids"].size(0), 1).to(device)
            #####
            ##### if arg.adjacency is True:
            #####     output = model.forward(data["input_ids"], position_num, adj_mat=data["smiles_bert_adjmat"])
            ##### else:
            #####     output = model.forward(data["input_ids"], position_num)
            #####
            ##### output = output[:, 0, :] #output:[batch_size,task_num]

            data["labels"] = data["labels"].view(output.shape).to(torch.float64)

            is_valid = data["labels"] ** 2 > 0

            loss = criterion(output.double(), (data["labels"]+1)/2)
            loss = torch.where(is_valid, loss, torch.zeros(loss.shape).to(loss.device).to(loss.dtype))
            loss = torch.sum(loss) / torch.sum(is_valid)
            loss.backward()

            optim.step()
            optim.zero_grad()

            avg_loss += loss.item()
            data_iter.set_description("epoch: {} iter: {} avg_loss: {:.6f} loss: {:.6f}".format(epoch, i+1, (avg_loss / (i+1) ), loss.item()) )

        logger.info("Epoch: {} Training finished.".format(epoch))
        logger.info("Epoch: {} epoch average loss: {}".format(epoch, avg_loss/len(data_iter)) )

        #********************************3.1、模型valid集验证*****************************************
        model.eval()
        valid_iter = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader))
        predicted_list = []
        target_list = []
        
        with torch.no_grad():
            for i, data in valid_iter:
                data = {key:value.to(device) for key, value in data.items()}
                
                output = model.forward(data["input_ids"],data["attention_mask"])				

                ##### position_num = torch.arange(arg.seq).repeat(data["input_ids"].size(0), 1).to(device)
                ##### 
                ##### if arg.adjacency is True:
                #####     output = model.forward(data["input_ids"], position_num, adj_mat=data["smiles_bert_adjmat"])
                ##### else:
                #####     output = model.forward(data["input_ids"], position_num)
                #####
                ##### output = output[:, 0, :] #output:[batch_size,task_num]

                data["labels"] = data["labels"].view(output.shape).to(torch.float64)

                is_valid = data["labels"] ** 2 > 0

                valid_loss = criterion(output.double(), (data["labels"]+1)/2)
                valid_loss = torch.where(is_valid, valid_loss, torch.zeros(valid_loss.shape).to(valid_loss.device).to(valid_loss.dtype))
                valid_loss = torch.sum(valid_loss) / torch.sum(is_valid)

                valid_avg_loss += valid_loss.item()
                predicted = torch.sigmoid(output)
                predicted_list.append(predicted)
                target_list.append(data["labels"])
        
        predicted_list = torch.cat(predicted_list, dim=0).cpu().numpy()
        target_list = torch.cat(target_list, dim=0).cpu().numpy()


        roc_list = []
        for i in range(target_list.shape[1]):
            if np.sum(target_list[:,i] == 1) > 0 and np.sum(target_list[:,i] == -1) > 0:
                is_valid = target_list[:,i] ** 2 > 0
                roc_list.append(roc_auc_score((target_list[is_valid,i]+1)/2, predicted_list[is_valid,i]))
                
        logger.info("AUCROC: {}".format(sum(roc_list)/len(roc_list)) )
        
        if valid_avg_loss < min_valid_loss:
            save_path = "../finetuned_model/" + str(arg.dataset) + "_epoch_" + str(epoch) + "_val_loss_" + str(round(valid_avg_loss/len(valid_dataloader),5))
            torch.save(model.state_dict(), save_path+'.pt')
            model.to(device)
            min_valid_loss = valid_avg_loss
            counter = 0
        counter += 1

        
        if counter > 5:
            break

    # eval
    print("Finished. Start evaluation.")

    predicted_list = []
    target_list = []


    #-----------------------------------------------------------4、模型test集测试-------------------------------------------------
    logger.info("Start testing...")
    logger.info("Evaluate on min valid loss model")
    

    predicted_list = []
    target_list = []
    
    print(save_path)
    model.load_state_dict(torch.load(save_path+'.pt'))
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            data = {key:value.to(device) for key, value in data.items()}

            output = model.forward(data["input_ids"],data["attention_mask"])
            
            ##### position_num = torch.arange(arg.seq).repeat(data["input_ids"].size(0), 1).to(device)
            ##### 
            ##### if arg.adjacency is True:
            #####     output = model.forward(data["input_ids"], position_num, adj_mat=data["smiles_bert_adjmat"])
            ##### else:
            #####     output = model.forward(data["input_ids"], position_num)
            #####
            ##### output = output[:, 0, :] #output:[batch_size,task_num]
            
            data["labels"] = data["labels"].view(output.shape).to(torch.float64)
            predicted = torch.sigmoid(output)
            predicted_list.append(predicted)
            target_list.append(data["labels"])

        predicted_list = torch.cat(predicted_list, dim=0).cpu().numpy()
        target_list = torch.cat(target_list, dim=0).cpu().numpy()
        roc_list = []
        for i in range(target_list.shape[1]):
            if np.sum(target_list[:,i] == 1) > 0 and np.sum(target_list[:,i] == -1) > 0:
                is_valid = target_list[:,i] ** 2 > 0
                roc_list.append(roc_auc_score((target_list[is_valid,i]+1)/2, predicted_list[is_valid,i]))

        
        logger.info("Finally AUCROC:{} ".format(sum(roc_list)/len(roc_list)) )

if __name__ == "__main__":
    main()
