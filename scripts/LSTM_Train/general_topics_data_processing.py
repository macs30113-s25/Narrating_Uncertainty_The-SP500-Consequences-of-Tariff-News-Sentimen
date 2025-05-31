import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(merged_file, look_back=60, forecast_horizon=1, ticker=None):
    # 初始化返回变量
    X_train = np.array([])
    X_test = np.array([])
    y_train = np.array([])
    y_test = np.array([])
    scaler = None
    feature_columns = []
    
    try:
        # 读取合并后的数据
        data = pd.read_csv(merged_file)
        
        print(f"读取数据: {merged_file}")
        print(f"初始数据形状: {data.shape}")
        print(f"初始数据列名: {data.columns.tolist()}")
        
        # 如果指定了股票代码，则过滤数据
        if ticker:
            # 统一股票代码格式
            ticker = ticker.strip().upper()
            print(f"过滤股票代码: {ticker}")
            
            # 尝试不同的股票代码列
            found = False
            if 'ticker' in data.columns:
                filtered_data = data[data['ticker'].str.strip().str.upper() == ticker]
                if not filtered_data.empty:
                    data = filtered_data
                    found = True
            if not found and 'symbol' in data.columns:
                filtered_data = data[data['symbol'].str.strip().str.upper() == ticker]
                if not filtered_data.empty:
                    data = filtered_data
                    found = True
            if not found and 'stock_symbol' in data.columns:
                filtered_data = data[data['stock_symbol'].str.strip().str.upper() == ticker]
                if not filtered_data.empty:
                    data = filtered_data
                    found = True
            
            if not found:
                print(f"警告: 未找到股票代码 '{ticker}' 的记录")
                return X_train, X_test, y_train, y_test, scaler, feature_columns
        
        # 确保日期格式正确
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').set_index('date')
        
        # 打印过滤后的数据信息
        print(f"过滤后数据形状: {data.shape}")
        if len(data) > 0:
            print(f"数据日期范围: {data.index.min()} 到 {data.index.max()}")
        
        # 检查数据量是否足够
        min_data_points = 20
        if len(data) < min_data_points:
            print(f"错误: 数据量不足 ({len(data)}行)，需要至少{min_data_points}行数据")
            return X_train, X_test, y_train, y_test, scaler, feature_columns
        
        # 技术指标特征工程 - 安全版本
        print(f"开始特征工程，数据量: {len(data)}行")
        
        # 1. 添加滞后特征
        for i in range(1, 6):
            data[f'close_lag_{i}'] = data['close'].shift(i)
        
        # 2. 添加基本特征
        if 'close' in data.columns:
            data['daily_return'] = data['close'].pct_change()
            data['ma_10'] = data['close'].rolling(10).mean()
            data['volatility'] = data['close'].rolling(10).std()
        
        # 3. 添加技术指标 - 使用安全方式
        if 'close' in data.columns and len(data) >= 14:
            try:
                from ta import momentum
                data['momentum_rsi'] = momentum.RSIIndicator(data['close'], window=14).rsi()
            except Exception as e:
                print(f"RSI计算失败: {str(e)}")
        
        if 'close' in data.columns and len(data) >= 26:
            try:
                from ta import trend
                data['trend_macd'] = trend.MACD(data['close']).macd()
            except Exception as e:
                print(f"MACD计算失败: {str(e)}")
        
        # 处理缺失值
        initial_count = len(data)
        data = data.dropna()
        print(f"删除缺失值: {initial_count - len(data)}行，剩余{len(data)}行")
        
        # 再次检查数据量
        if len(data) < look_back + forecast_horizon:
            print(f"错误: 清洗后数据量不足 ({len(data)}行)，需要至少{look_back + forecast_horizon}行")
            return X_train, X_test, y_train, y_test, scaler, feature_columns
        
        # 选择特征列
        possible_features = [
            'close', 'volume', 'sentiment_score', 
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_4', 'close_lag_5',
            'daily_return', 'volatility', 'ma_10',
            'trend_macd', 'momentum_rsi'
        ]
        
        feature_columns = [col for col in possible_features if col in data.columns]
        print(f"使用的特征 ({len(feature_columns)}个): {feature_columns}")
        
        # 目标列
        target_column = 'close'
        
        # 划分数据集
        split_idx = int(len(data) * 0.8)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        # 分别归一化
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_train = scaler.fit_transform(train_data[feature_columns])
        scaled_test = scaler.transform(test_data[feature_columns])
        
        # 创建序列数据集
        target_idx = feature_columns.index(target_column) if target_column in feature_columns else 0
        
        def create_sequences(data, look_back, horizon):
            X, y = [], []
            for i in range(look_back, len(data) - horizon):
                X.append(data[i-look_back:i])
                y.append(data[i + horizon, target_idx])
            return np.array(X), np.array(y)
        
        # 创建数据集
        X_train, y_train = create_sequences(scaled_train, look_back, forecast_horizon)
        X_test, y_test = create_sequences(scaled_test, look_back, forecast_horizon)
        
        print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
        print(f"测试集形状: X={X_test.shape}, y={y_test.shape}")
        
    except Exception as e:
        print(f"预处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return X_train, X_test, y_train, y_test, scaler, feature_columns

if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(
            'merged_stock_sentiment.csv',
            look_back=60,
            forecast_horizon=1
        )
    except Exception as e:
        print(f"预处理错误: {str(e)}")
        import traceback
        traceback.print_exc()