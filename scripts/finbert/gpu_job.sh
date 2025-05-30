#!/bin/bash
#SBATCH --job-name=finbert_large
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # 增加CPU核心数以支持数据加载
#SBATCH --account=macs30113
#SBATCH --time=24:00:00           # 延长运行时间（大数据集需要）
#SBATCH --mem=100G                # 增加内存（处理大数据集需要）
#SBATCH --output=finbert-%j.log   # 添加日志文件

module purge
module load python/anaconda-2022.05
module load cuda/11.7

# 创建并激活专用环境
conda create -n finbert python=3.9 -y
conda activate finbert

# 安装必要库（确保安装GPU版本的PyTorch）
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install transformers pandas tqdm scikit-learn

# 运行脚本（使用更大批处理大小）
python sentiment_analysis_local.py \
  --input "/home/baihuiw/nasdaq_exteral_data.csv" \
  --output_dir "./gpu_results" \
  --model_path "/home/baihuiw/finbert_model" \
  --batch_size 256

