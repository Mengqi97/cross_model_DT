#!/bin/bash
#SBATCH -J hpc_HIV_cross_model                              # 作业名
#SBATCH -o hpc_HIV_cross_model.out                           # 屏幕上的输出文件重定向到xxx.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                       # 作业申请 x 个节点
#SBATCH -t 48:00:00                                # 任务运行的最长时间为 x 小时
#SBATCH -w gpu24 --gres=gpu:tesla_v100-pcie-32gb:1            # 指定运行作业的节点是 gpuxx，若不填写系统自动分配节点


source ~/.bashrc

# 设置运行环境
conda activate py36

cd ./cross_model_DT/cross_model_DT

# 输入要执行的命令，例如 ./hello 或 python test.py 等
python -V
       # 执行命令
python ./molenet_finetune.py --path ./data/DT_dataset --dataset hiv  