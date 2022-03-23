# cross_model_DT
cross_model在MoleculeNet上运行的下游任务代码

./finetuned_model 用来存储微调过程中的候选最佳模型
./cross_model_DT/cross_model_DT/config.py 配置文件  
./cross_model_DT/cross_model_DT/data_utils.py 创建smiles字符串词级tokenizer  
./cross_model_DT/cross_model_DT/vocab.py 创建词表，为方便调用，从SMILES_PE源码中(data_utils.py)抽取出来  
./cross_model_DT/cross_model_DT/models.py 模型结构  
./cross_model_DT/cross_model_DT/molenet_finetun.py 微调代码   
./cross_model_DT/cross_model_DT/pretrained_model 用来存储预训练模型  
./cross_model_DT/cross_model_DT/common_scripts 用来存储运行所有数据集的脚本  
./cross_model_DT/cross_model_DT/  
./cross_model_DT/cross_model_DT/  
