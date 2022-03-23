# cross_model_DT
cross_model在MoleculeNet上运行的下游任务代码

运行说明：  
运行指令可参照./cross_model_DT/cross_model_DT/common_scripts中八个数据集各自对应的运行脚本。

例如，如果要运行的是bace数据集，运行指令如下:  
`cd ./cross_model_DT/cross_model_DT`    
`python ./molenet_finetune.py --path ./data/DT_dataset --dataset bace `    
其他数据集同理，将'bace'换成其他数据集即可(数据集名称字母需要全小写)。  
  
另外在运行时还可传入其他参数进行调参操作：    
`python ./molenet_finetune.py` <br>
&emsp;&emsp;&emsp;`[--dataset] , help="name of dataset", type=str` <br>
&emsp;&emsp;&emsp;`[--batch], help="batch size", type=int, default=16 ` <br>
&emsp;&emsp;&emsp;`[--lr] , help="learning rate", type=float, default=3e-5 `<br> 
&emsp;&emsp;&emsp;`[--seed] , help="set up random seeds ", type=int, default=7 `<br>
&emsp;&emsp;&emsp;`[--split] , help="type of dataset", type=str, 可传入参数：{'scaffold','random'}` <br>
          

文件结构及功能说明：    
./finetuned_model 用来存储微调过程中的候选最佳模型  
./cross_model_DT/cross_model_DT/config.py 配置文件  
./cross_model_DT/cross_model_DT/data_utils.py 创建smiles字符串词级tokenizer  
./cross_model_DT/cross_model_DT/vocab.py 创建词表，为方便调用，从SMILES_PE源码中(data_utils.py)抽取出来  
./cross_model_DT/cross_model_DT/models.py 模型结构  
./cross_model_DT/cross_model_DT/molenet_finetun.py 微调代码   
./cross_model_DT/cross_model_DT/pretrained_model 用来存储预训练模型,模型从[此处](https://drive.google.com/file/d/124jL0RUQ2zRcX7Gaj9ySs6_fxHJROUXz/view?usp=sharing)下载  
./cross_model_DT/cross_model_DT/common_scripts 用来存储运行所有数据集的脚本  
./cross_model_DT/cross_model_DT/data SPE_ChEMBL.txt和spe_voc.txt是用来初始化的文件  
./cross_model_DT/cross_model_DT/data/DT_dataset 用来存储八个全部下游任务数据集
