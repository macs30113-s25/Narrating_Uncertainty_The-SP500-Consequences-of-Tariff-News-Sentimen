import pandas as pd

def merge_data(sentiment_file, json_files):
    """
    合并情感分数数据和股价数据
    
    参数:
    sentiment_file: 情感分数CSV文件路径
    json_files: 股价JSON文件路径列表
    """
    # 读取情感分数数据
    sentiment_df = pd.read_csv(sentiment_file)
    print("情感数据列名:", sentiment_df.columns.tolist())
    
    # 读取所有股价JSON文件
    sp_frames = []
    for file in json_files:
        df = pd.read_json(file)
        print(f"{file} 列名:", df.columns.tolist())
        sp_frames.append(df)
    
    # 合并股价数据
    sp_data = pd.concat(sp_frames)
    
    # 标准化列名 - 全部转为小写并去除空格
    sentiment_df.columns = sentiment_df.columns.str.strip().str.lower()
    sp_data.columns = sp_data.columns.str.strip().str.lower()
    
    # 重命名股票代码列以匹配
    # 情感数据中可能是 'symbol' 或 'stock_symbol'
    # 股价数据中可能是 'ticker', 'symbol' 或 'stock_symbol'
    if 'symbol' in sentiment_df.columns:
        sentiment_df.rename(columns={'symbol': 'ticker'}, inplace=True)
    elif 'stock_symbol' in sentiment_df.columns:
        sentiment_df.rename(columns={'stock_symbol': 'ticker'}, inplace=True)
    
    if 'symbol' in sp_data.columns:
        sp_data.rename(columns={'symbol': 'ticker'}, inplace=True)
    elif 'stock_symbol' in sp_data.columns:
        sp_data.rename(columns={'stock_symbol': 'ticker'}, inplace=True)
    
    print("\n合并前情感数据列名:", sentiment_df.columns.tolist())
    print("合并前股价数据列名:", sp_data.columns.tolist())
    
    # 转换日期格式
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    sp_data['date'] = pd.to_datetime(sp_data['date'])
    
    # 合并数据 - 使用重命名后的列
    merged_df = pd.merge(
        sentiment_df,
        sp_data[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']],
        on=['date', 'ticker'],
        how='left'
    )
    
    # 处理缺失值
    na_count = merged_df['close'].isna().sum()
    if na_count > 0:
        print(f"警告: 发现 {na_count} 条记录缺少股价数据，已删除")
        merged_df = merged_df.dropna(subset=['close'])
    
    # 保存合并后的数据
    merged_df.to_csv('merged_stock_sentiment.csv', index=False)
    print(f"\n数据合并完成，保存到 merged_stock_sentiment.csv")
    print(f"总记录数: {len(merged_df)}")
    
    # 显示合并后的列名和前几行数据
    print("\n合并后的列名:", merged_df.columns.tolist())
    print("\n合并后的数据示例:")
    print(merged_df.head())
    
    return merged_df

if __name__ == "__main__":
    # 配置文件路径
    sentiment_file = 'sentiment_score_details.csv'
    json_files = json_files =['sp500_all_companies_2019-05-01_to_2019-09-30.json', 'sp500_all_companies_2020-01-01_to_2020-03-31.json']  # 替换为实际文件路径
    
    # 执行合并
    merged_data = merge_data(sentiment_file, json_files)


