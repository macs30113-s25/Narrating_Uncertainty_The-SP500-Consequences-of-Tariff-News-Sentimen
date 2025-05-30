#!/bin/bash

# 安装必要的库
sudo yum install -y python3-pip
sudo pip3 install torch==1.11.0 --no-cache-dir
sudo pip3 install transformers==4.18.0 --no-cache-dir
sudo pip3 install pandas --no-cache-dir
sudo pip3 install boto3 --no-cache-dir

# 从 S3 下载模型文件
aws s3 sync s3://your-bucket/finbert/model/ /home/hadoop/finbert_model/