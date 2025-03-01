#!/bin/bash

#SBATCH --partition=g078t1
### 指定队列为g078t2

#SBATCH --nodes=1
### 指定该作业需要1个节点数

#SBATCH --ntasks-per-node=8
### 每个节点所运行的进程数为8，最大为64

#SBATCH --time=8:00:00
### 作业的最大运行时间，超过时间后作业资源会被SLURM回收;该时间不能超过分区的最大运行时间

#SBATCH --gres=gpu:1
###（声明需要的GPU数量）【单节点最大申请8个GPU】

#SBATCH --comment=efficientllm
### 指定从哪个项目扣费（即导师所在的项目名称，可以在平台上查看，或者咨询导师）

### 程序的执行命令
source ~/.bashrc  ### 初始化环境变量
source /opt/app/anaconda3/bin/activate mezo
cd large_models
export HF_ENDPOINT=https://hf-mirror.com


#MODEL=/home/u2024140809/share/opt-1.3B TASK=SST2 MODE=lora LR=5e-5 EPS=1e-2  BF=fp16  bash mezo.sh 
# MODEL=/home/u2024140809/share/Qwen:Qwen2.5-0.5B TASK=SST2 MODE=lora LR=5e-5 EPS=1e-2  BF=fp16  bash mezo.sh 
#MODEL=/home/u2024140809/share/phi-1 TASK=SST2 MODE=lora LR=5e-5 EPS=1e-2  BF=fp16  bash mezo.sh 
MODEL=/data/pc/models/Llama-3.2-1B TASK=SST2 MODE=lora LR=5e-5 EPS=1e-2  BF=fp16  bash mezo.sh 
