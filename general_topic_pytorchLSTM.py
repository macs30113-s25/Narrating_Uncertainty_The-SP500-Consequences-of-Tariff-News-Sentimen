import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Attention机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),  
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        # 初始化双向LSTM的隐藏状态
        num_directions = 2
        batch_size = x.size(0)
        
        h0 = torch.zeros(self.num_layers * num_directions, 
                         batch_size, 
                         self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * num_directions, 
                         batch_size, 
                         self.hidden_size).to(x.device)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Attention机制
        attn_scores = self.attention(lstm_out)
        attn_scores = attn_scores.squeeze(2)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # 上下文向量（加权和）
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out)
        context = context.squeeze(1)
        
        # 最终输出
        out = self.fc(context)
        return out

def load_model_and_features(model_path, feature_path):
    """加载模型和特征名称"""
    # 加载特征名称
    with open(feature_path, 'r') as f:
        feature_names = json.load(f)
    
    # 初始化模型
    model = LSTMModel(
        input_size=len(feature_names),
        hidden_size=128,
        num_layers=2,
        output_size=1
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, feature_names

def preprocess_data_for_eval(merged_file, look_back, forecast_horizon, ticker, feature_names):
    """数据预处理（与训练时保持一致）"""
    # 加载数据
    if not os.path.exists(merged_file):
        raise FileNotFoundError(f"文件 {merged_file} 不存在")
    
    data = pd.read_csv(merged_file)
    
    # 按股票代码筛选
    if ticker:
        data = data[data['ticker'] == ticker]
        if data.empty:
            raise ValueError(f"没有找到股票代码 {ticker} 的数据")
    
    # 提取日期信息
    if 'date' in data.columns:
        dates = data['date'].values
    else:
        # 如果没有日期列，创建基于索引的日期
        start_date = datetime(2020, 1, 1)
        dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                 for i in range(len(data))]
    
    # 选择特征列
    data = data[feature_names].values.astype(float)
    
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 创建时间序列数据集
    X, y, date_seq = [], [], []
    for i in range(look_back, len(scaled_data) - forecast_horizon + 1):
        X.append(scaled_data[i-look_back:i])
        # 预测未来第forecast_horizon天的收盘价
        y.append(scaled_data[i+forecast_horizon-1, feature_names.index('close')])
        date_seq.append(dates[i+forecast_horizon-1])
    
    # 转换为numpy数组
    X = np.array(X)
    y = np.array(y)
    
    # 使用全部数据进行评估
    return X, y, scaler, date_seq

def evaluate_model_performance(model, X, y, scaler, feature_names, date_seq, batch_size=64):
    """评估模型性能并生成报告"""
    # 转换为PyTorch张量
    X_tensor = torch.tensor(X).float().to(device)
    
    # 预测
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    
    # 反归一化
    target_idx = feature_names.index('close')
    
    # 反归一化预测值
    dummy_pred = np.zeros((len(predictions), len(feature_names)))
    dummy_pred[:, target_idx] = predictions.flatten()
    predictions = scaler.inverse_transform(dummy_pred)[:, target_idx]
    
    # 反归一化实际值
    dummy_actual = np.zeros((len(y), len(feature_names)))
    dummy_actual[:, target_idx] = y.flatten()
    actual = scaler.inverse_transform(dummy_actual)[:, target_idx]
    
    # 计算评估指标
    mse = mean_squared_error(actual, predictions)
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predictions)
    
    # 计算绝对百分比误差
    ape = np.abs((actual - predictions) / actual) * 100
    mape = np.mean(ape)
    
    # 计算残差
    residuals = actual - predictions
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'Date': date_seq,
        'Actual': actual,
        'Predicted': predictions,
        'Residual': residuals,
        'Absolute_Error': np.abs(actual - predictions),
        'Percentage_Error': ape
    })
    
    # 转换日期列为datetime类型
    results_df['Date'] = pd.to_datetime(results_df['Date'])
    
    # 生成时间戳用于文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 1. 实际vs预测折线图 (单独保存)
    plt.figure(figsize=(14, 6))
    plt.plot(results_df['Date'], actual, label='Actual Price', color='blue')
    plt.plot(results_df['Date'], predictions, label='Predicted Price', color='red', alpha=0.7)
    plt.title(f'Stock Price Prediction\nRMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()
    plt.savefig(f'prediction_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 误差分布直方图 (单独保存)
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['Absolute_Error'], kde=True)
    plt.title('Distribution of Absolute Errors')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.savefig(f'absolute_error_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 百分比误差分布 (单独保存)
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['Percentage_Error'], kde=True)
    plt.title(f'Distribution of Percentage Errors (MAPE: {mape:.2f}%)')
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.savefig(f'percentage_error_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 残差随时间变化图 (新增加)
    plt.figure(figsize=(14, 6))
    
    # 绘制残差
    plt.plot(results_df['Date'], residuals, 'o-', color='blue', alpha=0.7, label='Residuals')
    
    # 添加零参考线
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.7)
    
    # 添加趋势线
    sns.regplot(
        x=mdates.date2num(results_df['Date']), 
        y=residuals, 
        scatter=False, 
        color='green',
        line_kws={'linewidth': 2, 'linestyle': '--'},
        label='Trend Line'
    )
    
    # 标注最大正负残差
    max_residual_idx = np.argmax(residuals)
    min_residual_idx = np.argmin(residuals)
    
    plt.annotate(f'Max: {residuals[max_residual_idx]:.2f}', 
                 xy=(results_df['Date'].iloc[max_residual_idx], residuals[max_residual_idx]),
                 xytext=(results_df['Date'].iloc[max_residual_idx] + timedelta(days=10), 
                         residuals[max_residual_idx]),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate(f'Min: {residuals[min_residual_idx]:.2f}', 
                 xy=(results_df['Date'].iloc[min_residual_idx], residuals[min_residual_idx]),
                 xytext=(results_df['Date'].iloc[min_residual_idx] + timedelta(days=10), 
                         residuals[min_residual_idx] - np.ptp(residuals)*0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    # 设置图表属性
    plt.title('Residuals Over Time')
    plt.xlabel('Date')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    # 添加残差统计信息
    residual_stats = f"Mean Residual: {np.mean(residuals):.4f}\nStd Residual: {np.std(residuals):.4f}"
    plt.figtext(0.15, 0.02, residual_stats, backgroundcolor='white', fontsize=10)
    
    plt.savefig(f'residuals_over_time_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 残差分布图 (新增加)
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.axvline(x=np.mean(residuals), color='red', linestyle='--', label='Mean')
    plt.legend()
    plt.savefig(f'residuals_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印评估报告
    print("\n" + "="*50)
    print("模型评估报告")
    print("="*50)
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
    print(f"决定系数 (R²): {r2:.4f}")
    print(f"残差均值: {np.mean(residuals):.4f}")
    print(f"残差标准差: {np.std(residuals):.4f}")
    print(f"最大绝对误差: {results_df['Absolute_Error'].max():.4f}")
    print(f"最小绝对误差: {results_df['Absolute_Error'].min():.4f}")
    print(f"误差大于1%的样本比例: {len(results_df[results_df['Percentage_Error'] > 1])/len(results_df)*100:.2f}%")
    
    # 返回评估结果
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'results_df': results_df,
        'timestamp': timestamp
    }

def perform_regression_testing(model, X, y, scaler, feature_names, date_seq, n_tests=5):
    """执行回归测试：多次评估模型稳定性"""
    print("\n执行回归测试...")
    test_results = []
    timestamps = []
    
    for i in range(n_tests):
        print(f"\n回归测试 {i+1}/{n_tests}")
        results = evaluate_model_performance(model, X, y, scaler, feature_names, date_seq)
        test_results.append({
            'test_id': i+1,
            'rmse': results['rmse'],
            'mape': results['mape'],
            'r2': results['r2'],
            'mean_residual': results['results_df']['Residual'].mean(),
            'std_residual': results['results_df']['Residual'].std()
        })
        timestamps.append(results['timestamp'])
    
    # 创建回归测试结果DataFrame
    regression_df = pd.DataFrame(test_results)
    
    # 计算统计指标
    stats = {
        'mean_rmse': regression_df['rmse'].mean(),
        'std_rmse': regression_df['rmse'].std(),
        'mean_mape': regression_df['mape'].mean(),
        'std_mape': regression_df['mape'].std(),
        'mean_r2': regression_df['r2'].mean(),
        'std_r2': regression_df['r2'].std(),
        'mean_residual': regression_df['mean_residual'].mean(),
        'std_residual': regression_df['std_residual'].mean()
    }
    
    # 打印回归测试报告
    print("\n" + "="*50)
    print("回归测试报告")
    print("="*50)
    print(regression_df)
    print("\n统计指标:")
    print(f"平均RMSE: {stats['mean_rmse']:.4f} ± {stats['std_rmse']:.4f}")
    print(f"平均MAPE: {stats['mean_mape']:.2f}% ± {stats['std_mape']:.2f}%")
    print(f"平均R²: {stats['mean_r2']:.4f} ± {stats['std_r2']:.4f}")
    print(f"平均残差均值: {stats['mean_residual']:.4f}")
    print(f"平均残差标准差: {stats['std_residual']:.4f}")
    
    # 生成当前时间戳用于文件名
    regression_timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 1. RMSE稳定性图表 (单独保存)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=regression_df, x='test_id', y='rmse', marker='o')
    plt.title('RMSE Stability Over Tests')
    plt.xlabel('Test ID')
    plt.ylabel('RMSE')
    plt.ylim(0, regression_df['rmse'].max() * 1.2)
    plt.savefig(f'rmse_stability_{regression_timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. MAPE稳定性图表 (单独保存)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=regression_df, x='test_id', y='mape', marker='o')
    plt.title('MAPE Stability Over Tests (%)')
    plt.xlabel('Test ID')
    plt.ylabel('MAPE')
    plt.ylim(0, regression_df['mape'].max() * 1.2)
    plt.savefig(f'mape_stability_{regression_timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. R²稳定性图表 (单独保存)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=regression_df, x='test_id', y='r2', marker='o')
    plt.title('R² Stability Over Tests')
    plt.xlabel('Test ID')
    plt.ylabel('R²')
    plt.ylim(regression_df['r2'].min() - 0.1, 1.0)
    plt.savefig(f'r2_stability_{regression_timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 残差稳定性图表 (新增加)
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    sns.lineplot(data=regression_df, x='test_id', y='mean_residual', marker='o', label='Mean Residual')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.title('Mean Residual Stability Over Tests')
    plt.xlabel('Test ID')
    plt.ylabel('Mean Residual')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    sns.lineplot(data=regression_df, x='test_id', y='std_residual', marker='o', color='green', label='Residual Std Dev')
    plt.title('Residual Standard Deviation Over Tests')
    plt.xlabel('Test ID')
    plt.ylabel('Residual Std Dev')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'residual_stability_{regression_timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return regression_df, stats

if __name__ == "__main__":
    # 配置参数（与训练脚本一致）
    merged_file = 'merged_stock_sentiment.csv'
    model_path = 'INTC_model.pth'  # 修改为你的模型路径
    feature_path = 'feature_names.json'  # 修改为你的特征文件路径
    look_back = 90
    forecast_horizon = 1
    ticker = 'INTC'
    
    try:
        # 1. 加载模型和特征
        print("加载模型和特征...")
        model, feature_names = load_model_and_features(model_path, feature_path)
        print(f"成功加载模型: {model_path}")
        print(f"特征列表: {feature_names}")
        
        # 2. 预处理数据
        print("\n预处理评估数据...")
        X, y, scaler, date_seq = preprocess_data_for_eval(
            merged_file=merged_file,
            look_back=look_back,
            forecast_horizon=forecast_horizon,
            ticker=ticker,
            feature_names=feature_names
        )
        print(f"评估数据集形状: X={X.shape}, y={y.shape}")
        print(f"日期序列长度: {len(date_seq)}")
        
        # 3. 评估模型性能
        print("\n评估模型性能...")
        eval_results = evaluate_model_performance(model, X, y, scaler, feature_names, date_seq)
        
        # 4. 执行回归测试
        print("\n执行回归测试...")
        regression_df, stats = perform_regression_testing(model, X, y, scaler, feature_names, date_seq, n_tests=10)
        
        # 5. 保存评估结果
        eval_results['results_df'].to_csv('detailed_predictions.csv', index=False)
        regression_df.to_csv('regression_test_results.csv', index=False)
        
        print("\n评估完成! 结果已保存到文件")
        print(f"- 预测对比图: prediction_comparison_*.png")
        print(f"- 绝对误差分布图: absolute_error_distribution_*.png")
        print(f"- 百分比误差分布图: percentage_error_distribution_*.png")
        print(f"- 残差随时间变化图: residuals_over_time_*.png")
        print(f"- 残差分布图: residuals_distribution_*.png")
        print(f"- RMSE稳定性图: rmse_stability_*.png")
        print(f"- MAPE稳定性图: mape_stability_*.png")
        print(f"- R²稳定性图: r2_stability_*.png")
        print(f"- 残差稳定性图: residual_stability_*.png")
        print(f"- 详细预测结果: detailed_predictions.csv")
        print(f"- 回归测试结果: regression_test_results.csv")
        
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


