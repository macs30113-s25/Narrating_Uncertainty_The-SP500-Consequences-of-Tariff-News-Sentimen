import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# 1. 加载NASDAQ股票数据 - 合并两个JSON文件
# 定义要加载的文件
nasdaq_files = [
    'sp500_all_companies_2019-05-01_to_2019-09-30.json',
    'sp500_all_companies_2020-01-01_to_2020-03-31.json'
]

# 合并两个文件的数据
all_nasdaq_data = []
for file in nasdaq_files:
    try:
        with open(file) as f:
            print(f"加载文件: {file}")
            data = json.load(f)
            all_nasdaq_data.extend(data)
            print(f"成功加载 {len(data)} 条记录")
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {file}")
        continue
    except json.JSONDecodeError:
        print(f"错误: JSON格式无效 - {file}")
        continue

if not all_nasdaq_data:
    print("错误: 无法加载任何NASDAQ股票数据")
    exit(1)

print(f"总共加载 {len(all_nasdaq_data)} 条NASDAQ股票记录")

# 创建DataFrame
df_base = pd.json_normalize(all_nasdaq_data)
df_base.rename(columns={'symbol': 'ticker'}, inplace=True)



# 2. 处理日期范围
# 确保日期列是datetime类型
df_base['date'] = pd.to_datetime(df_base['date'], errors='coerce')

# 删除无效日期
invalid_dates = df_base[df_base['date'].isna()]
if not invalid_dates.empty:
    print(f"警告: 发现 {len(invalid_dates)} 个无效日期记录，已移除")
    df_base = df_base.dropna(subset=['date'])

# 筛选日期范围 (根据您的需求调整)
start_date = '2020-01-01'
end_date = '2020-03-31'

# 筛选日期范围
date_filter = (df_base['date'] >= start_date) & (df_base['date'] <= end_date)
filtered_df = df_base[date_filter].sort_values(['ticker', 'date'])

print(f"筛选后日期范围: {filtered_df['date'].min().date()} 到 {filtered_df['date'].max().date()}")
print(f"筛选后记录数: {len(filtered_df)}")

# 3. 加载并处理NASDAQ情感数据
try:
    # 根据您的截图数据，情感数据可能是CSV格式
    # 如果实际是JSON格式，可以改为: pd.read_json('nasdaq_sentiment_data.json')
    df_sentiment = pd.read_csv('/Users/wangbaihui/final-project-bc/results/nasdaq_exteral_data_Article_title_results.csv')
    
    # 清理列名：去除空格
    df_sentiment.columns = df_sentiment.columns.str.strip()
    
    # 打印列名以确认
    print("\n情感数据列名:", df_sentiment.columns.tolist())
    
    # 检查必要列是否存在
    required_columns = ['negative', 'neutral', 'positive', 'date']
    missing_cols = [col for col in required_columns if col not in df_sentiment.columns]
    
    if missing_cols:
        print(f"错误: 情感数据文件缺少必要的列: {missing_cols}")
        exit(1)
    
    # 转换日期列
    df_sentiment['date'] = pd.to_datetime(df_sentiment['date'], errors='coerce')
    
    # 移除无效日期
    df_sentiment = df_sentiment.dropna(subset=['date'])
    
    # 4. 整合三个情感分数为一个综合分数
    # 方法1: 正面 - 负面 (范围在-1到1之间)
    df_sentiment['score'] = df_sentiment['positive'] - df_sentiment['negative']
    
    # 方法2: 加权分数 (可选)
    # df_sentiment['score'] = (df_sentiment['positive'] * 1 + 
    #                          df_sentiment['neutral'] * 0 + 
    #                          df_sentiment['negative'] * -1)
    
    # 按日期聚合情感分数 (计算每日平均情感分数)
    daily_sentiment = df_sentiment.groupby('date')['score'].mean().reset_index()
    
    print(f"情感数据日期范围: {daily_sentiment['date'].min().date()} 到 {daily_sentiment['date'].max().date()}")
    print(f"情感记录数: {len(daily_sentiment)}")
    
except FileNotFoundError:
    print("错误: 未找到情感数据文件 'nasdaq_sentiment_data.csv'")
    exit(1)
except Exception as e:
    print(f"处理情感数据时出错: {str(e)}")
    exit(1)

# 5. 合并股价与情感数据
print("\n合并股价与情感数据...")
merged_df = pd.merge(
    filtered_df,
    daily_sentiment[['date', 'score']],
    on='date',
    how='left'
)

# 6. 填充缺失值并重命名列
# 计算缺失值数量
missing_before = merged_df['score'].isna().sum()

# 填充缺失值
merged_df['score'] = merged_df['score'].fillna(0)

# 重命名列
merged_df.rename(columns={'score': 'sentiment_score'}, inplace=True)

# 报告缺失值处理情况
if missing_before > 0:
    print(f"警告: {missing_before} 行数据缺失情感分数，已使用0填充")

# 7. 保存最终结果
output_file = 'nasdaq_with_sentiment_score.csv'
merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print("\n数据整合完成!")
print(f"保存至: {output_file}")
print(f"总记录数: {len(merged_df)}")
print(f"日期范围: {merged_df['date'].min().date()} 到 {merged_df['date'].max().date()}")

# 数据质量检查
print("\n数据质量报告:")
print(f"唯一公司数量: {merged_df['ticker'].nunique()}")
print(f"情感分数分布:")
print(merged_df['sentiment_score'].describe())

# 可选: 保存情感分数计算细节
sentiment_details_file = 'sentiment_score_details.csv'
df_sentiment.to_csv(sentiment_details_file, index=False, encoding='utf-8-sig')
print(f"\n情感分数计算细节已保存至: {sentiment_details_file}")