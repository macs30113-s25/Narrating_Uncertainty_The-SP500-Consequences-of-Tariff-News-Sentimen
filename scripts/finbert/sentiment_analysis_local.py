#!/usr/bin/env python3
import os
import sys
import subprocess
import warnings
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import time
import numpy as np
from tqdm import tqdm
import argparse
import re
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

# ================== 环境设置 ==================
# 禁用TensorFlow日志和警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


# ================== 主脚本 ==================
def validate_model_path(model_path):
    """验证模型路径是否包含必需的文件"""
    # 检查模型文件格式（支持 .bin 和 .safetensors）
    model_files = [
        "pytorch_model.bin", 
        "model.safetensors"
    ]
    
    # 检查是否存在任一模型文件
    model_exists = any(os.path.exists(os.path.join(model_path, f)) for f in model_files)
    
    # 检查其他必需文件
    REQUIRED_FILES = ['config.json', 'vocab.txt']
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(model_path, f))]
    
    if not model_exists:
        print(f"错误: 未找到模型文件 (应为 pytorch_model.bin 或 model.safetensors)")
        return False
        
    if missing:
        print(f"错误: 模型目录中缺少必需文件: {', '.join(missing)}")
        return False
    
    print(f"模型验证通过。在 {model_path} 中找到所有必需文件")
    return True

def load_model(model_path):
    """加载模型和tokenizer"""
    try:
        # 验证模型路径
        if not validate_model_path(model_path):
            return None, None
            
        print(f"从 {model_path} 加载模型...")
        
        # 加载 tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # 检查 GPU 可用性
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用 {device.upper()} 进行推理")
        
        # 加载模型（支持 safetensors 格式）
        model = BertForSequenceClassification.from_pretrained(
            model_path,
            device_map=device if device == "cuda" else None
        )
        
        # 如果使用 CPU，确保模型在 CPU 上
        if device == "cpu":
            model = model.to("cpu")
            
        return tokenizer, model
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None, None

def analyze_batch(texts, tokenizer, model):
    """批量分析文本情感"""
    try:
        if not texts:
            return []
            
        # 处理文本
        inputs = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        # 将输入移动到GPU（如果可用）
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # 模型预测
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 将logits移回CPU
        logits = outputs.logits
        if torch.cuda.is_available():
            logits = logits.cpu()
        
        # 计算概率
        probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
        
        # 构建结果
        results = []
        for i, prob in enumerate(probs):
            # 确定情感标签
            sentiment = "neutral"
            if prob[2] > 0.5 and prob[2] > prob[0]:  # positive
                sentiment = "positive"
            elif prob[0] > 0.5 and prob[0] > prob[2]:  # negative
                sentiment = "negative"
            
            results.append({
                "negative": float(prob[0]),
                "neutral": float(prob[1]),
                "positive": float(prob[2]),
                "sentiment": sentiment
            })
        return results
    except Exception as e:
        print(f"批处理错误: {e}")
        return [{"error": str(e)}] * len(texts)

def determine_columns(file_type, df):
    """根据文件类型确定要分析的列"""
    if file_type == "all_external":
        return ["Article"]  # All_external.csv 只分析 Article 列
    
    elif file_type == "nasdaq":
        # Nasdaq 文件分析多个列
        return ["Article_title", "Article", "Lsa_summary", "Luhn_summary", "Textrank_summary", "Lexrank_summary"]
    
    else:
        # 自动检测
        available_columns = set(df.columns)
        text_columns = []
        
        # 标题类列
        title_cols = [col for col in available_columns if "title" in col.lower()]
        if title_cols:
            text_columns.extend(title_cols)
        
        # 文章内容类列
        article_cols = [col for col in available_columns if "article" in col.lower() and col not in title_cols]
        if article_cols:
            text_columns.extend(article_cols)
        
        # 摘要类列
        summary_cols = [col for col in available_columns if "summary" in col.lower()]
        if summary_cols:
            text_columns.extend(summary_cols)
        
        # 如果没找到特定列，使用通用文本列
        if not text_columns:
            text_candidates = ["text", "content", "body", "description"]
            text_columns = [col for col in text_candidates if col in available_columns]
        
        return text_columns if text_columns else ["Article"]  # 默认尝试

def clean_filename(filename):
    """清理文件名，移除特殊字符"""
    return re.sub(r'[^a-zA-Z0-9_]', '_', filename)

def run_analysis(input_file, output_dir, model_path, batch_size=4, file_type="auto", time_ranges=None):
    """运行分析流程 - 按指定时间范围筛选数据"""
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件未找到: {input_file}")
        return None
    
    print(f"\n{'='*50}")
    print(f"处理文件: {os.path.basename(input_file)}")
    print(f"使用模型: {os.path.abspath(model_path)}")
    print(f"{'='*50}")
    
    # 1. 读取整个文件
    print(f"读取输入文件: {input_file}")
    
    try:
        # 读取整个文件
        df = pd.read_csv(input_file, low_memory=False)
        print(f"读取完成，数据形状: {df.shape}")
        
        # 检查日期列是否存在
        if 'Date' not in df.columns:
            print("警告: 数据中没有日期列，将处理全部数据")
        else:
            # 解析日期列
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None)
            df = df.dropna(subset=['Date'])
            
            # 添加年份列
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            
            # 默认时间范围（如果未提供）
            if time_ranges is None:
                time_ranges = [
                    (pd.Timestamp('2019-05-01'), pd.Timestamp('2019-09-30')),
                    (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-03-31'))
                ]
                print(f"使用默认时间范围: 2019-05至2019-09 和 2020-01至2020-03")
            
            # 创建时间范围过滤条件
            time_filters = []
            for start, end in time_ranges:
                time_filters.append((df['Date'] >= start) & (df['Date'] <= end))
            
            # 合并所有时间范围条件
            combined_filter = pd.Series(False, index=df.index)
            for f in time_filters:
                combined_filter |= f
            
            # 应用时间范围过滤
            filtered_df = df[combined_filter]
            
            if len(filtered_df) == 0:
                print("警告: 没有找到指定时间范围内的数据，将处理全部数据")
            else:
                df = filtered_df
                print(f"按时间范围过滤后数据形状: {df.shape}")
                print(f"时间范围分布:\n{df.groupby(['Year', 'Month']).size()}")
                
    except Exception as e:
        print(f"读取或处理CSV时出错: {e}")
        return None
    
    # 2. 确定文件类型
    if file_type == "auto":
        filename = os.path.basename(input_file).lower()
        if "all_external" in filename:
            file_type = "all_external"
        elif "nasdaq" in filename:
            file_type = "nasdaq"
        else:
            file_type = "auto"
            print("文件类型未识别，使用自动检测")
    
    # 3. 确定要分析的列
    text_columns = determine_columns(file_type, df)
    print(f"要分析的文本列: {', '.join(text_columns)}")
    
    # 4. 加载模型
    tokenizer, model = load_model(model_path)
    if tokenizer is None or model is None:
        print("加载模型失败，退出。")
        return None
    
    # 5. 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    base_filename = clean_filename(os.path.splitext(os.path.basename(input_file))[0])
    
    # 6. 对每个文本列进行分析
    all_results = {}
    
    for text_column in text_columns:
        if text_column not in df.columns:
            print(f"\n警告: 列 '{text_column}' 在数据中未找到，跳过")
            continue
            
        print(f"\n{'='*40}")
        print(f"分析列: {text_column}")
        print(f"{'='*40}")
        
        # 准备文本数据
        texts = df[text_column].astype(str).fillna("").tolist()
        total_texts = len(texts)
        
        # 使用进度条
        results = []
        processing_times = []
        
        with tqdm(total=total_texts, desc="处理文本") as pbar:
            # 分批处理文本
            for i in range(0, total_texts, batch_size):
                batch_texts = texts[i:i+batch_size]
                
                start_time = time.time()
                
                # 批量分析情感
                batch_results = analyze_batch(batch_texts, tokenizer, model)
                
                # 记录处理时间
                proc_time = time.time() - start_time
                processing_times.append(proc_time)
                
                # 创建结果记录
                for j, result in enumerate(batch_results):
                    idx = i + j
                    if idx < len(df):
                        row = df.iloc[idx]
                        result_record = {
                            "original_index": idx,
                            "text_column": text_column,
                            "text": row[text_column][:200] + "..." if len(str(row[text_column])) > 200 else str(row[text_column]),
                            **result
                        }
                        
                        # 添加关键元数据
                        if "Date" in df.columns:
                            result_record["date"] = row["Date"]
                        if "Stock_symbol" in df.columns:
                            result_record["stock_symbol"] = row["Stock_symbol"]
                        if "Publisher" in df.columns:
                            result_record["publisher"] = row["Publisher"]
                        if "Year" in df.columns:
                            result_record["year"] = row["Year"]
                        if "Month" in df.columns:
                            result_record["month"] = row["Month"]
                            
                        results.append(result_record)
                
                # 更新进度
                pbar.update(len(batch_texts))
                avg_time = np.mean(processing_times) if processing_times else 0
                pbar.set_postfix({
                    "平均时间": f"{avg_time:.2f}s/批",
                    "批大小": batch_size,
                    "速度": f"{batch_size/avg_time:.1f}文本/s" if avg_time > 0 else "N/A",
                    "预计剩余": f"{(total_texts - pbar.n)/(batch_size/max(avg_time, 0.01)):.1f}s" if avg_time > 0 else "N/A"
                })
        
        # 保存该列的结果
        result_df = pd.DataFrame(results)
        column_output_file = os.path.join(output_dir, f"{base_filename}_{clean_filename(text_column)}_results.csv")
        result_df.to_csv(column_output_file, index=False)
        print(f"\n'{text_column}' 的结果已保存到: {column_output_file}")
        
        # 存储结果用于最终报告
        all_results[text_column] = {
            "result_df": result_df,
            "output_file": column_output_file
        }
    
    # 7. 生成综合报告
    if not all_results:
        print("\n没有有效的文本列被分析。")
        return None
    
    report_data = []
    
    for col, data in all_results.items():
        result_df = data["result_df"]
        valid_results = result_df[~result_df['sentiment'].isna()]
        
        report_entry = {
            "文件名": os.path.basename(input_file),
            "文本列": col,
            "总记录数": len(result_df),
            "处理记录数": len(valid_results),
            "积极数": len(valid_results[valid_results['sentiment'] == 'positive']),
            "消极数": len(valid_results[valid_results['sentiment'] == 'negative']),
            "中性数": len(valid_results[valid_results['sentiment'] == 'neutral']),
            "错误数": len(result_df) - len(valid_results),
            "输出文件": data["output_file"]
        }
        report_data.append(report_entry)
    
    report_df = pd.DataFrame(report_data)
    report_file = os.path.join(output_dir, f"{base_filename}_analysis_report.csv")
    report_df.to_csv(report_file, index=False)
    print(f"\n综合分析报告已保存到: {report_file}")
    
    # 8. 生成时间范围分布报告
    if 'Year' in df.columns and 'Month' in df.columns:
        time_distribution = df.groupby(['Year', 'Month']).size().reset_index(name='记录数')
        time_report_file = os.path.join(output_dir, f"{base_filename}_time_distribution.csv")
        time_distribution.to_csv(time_report_file, index=False)
        print(f"时间范围分布报告已保存到: {time_report_file}")
    
    return report_df

def process_all_files(input_dir, output_dir, model_path, batch_size=4, time_ranges=None):
    """处理目录中的所有CSV文件 - 按时间范围筛选"""
    # 确保目录存在
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有CSV文件
    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    
    if not csv_files:
        print(f"在 {input_dir} 中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件待处理")
    
    all_reports = []
    
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        print(f"\n{'='*50}")
        print(f"开始处理文件: {csv_file}")
        print(f"{'='*50}")
        
        report = run_analysis(
            input_file=input_path,
            output_dir=output_dir,
            model_path=model_path,
            batch_size=batch_size,
            time_ranges=time_ranges
        )
        
        if report is not None:
            all_reports.append(report)
    
    # 保存合并报告
    if all_reports:
        combined_report = pd.concat(all_reports, ignore_index=True)
        combined_report_file = os.path.join(output_dir, "combined_analysis_report.csv")
        combined_report.to_csv(combined_report_file, index=False)
        print(f"\n合并报告已保存到: {combined_report_file}")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='FinBERT 多文件情感分析流水线 - 按时间范围筛选')
    parser.add_argument('--input', type=str, help='输入CSV文件路径')
    parser.add_argument('--input_dir', type=str, default='./input', help='包含CSV文件的目录')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的FinBERT模型路径')
    parser.add_argument('--batch_size', type=int, default=16, help='处理批大小')
    
    # 添加时间范围参数
    parser.add_argument('--time_ranges', nargs='+', action='append', 
                        help='指定时间范围，格式: "开始日期 结束日期"。例如: --time_ranges "2019-05-01 2019-09-30" --time_ranges "2020-01-01 2020-03-31"')
    
    args = parser.parse_args()
    
    # 处理时间范围参数
    parsed_time_ranges = []
    if args.time_ranges:
        for tr in args.time_ranges:
            if len(tr) == 2:
                try:
                    start = pd.Timestamp(tr[0])
                    end = pd.Timestamp(tr[1])
                    parsed_time_ranges.append((start, end))
                    print(f"添加时间范围: {start.date()} 至 {end.date()}")
                except Exception as e:
                    print(f"忽略无效时间范围: {tr} - {e}")
    
    # 如果没有提供时间范围，使用默认范围
    if not parsed_time_ranges:
        default_ranges = [
            (pd.Timestamp('2019-05-01'), pd.Timestamp('2019-09-30')),
            (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-03-31'))
        ]
        parsed_time_ranges = default_ranges
        print("使用默认时间范围: 2019-05-01至2019-09-30 和 2020-01-01至2020-03-31")
    
    print("\n" + "="*50)
    print("FinBERT 多文件情感分析 - 按时间范围筛选")
    print(f"模型路径: {os.path.abspath(args.model_path)}")
    print(f"批大小: {args.batch_size}")
    print(f"时间范围: {', '.join([f'{s.date()}至{e.date()}' for s, e in parsed_time_ranges])}")
    print("="*50)
    
    # 验证模型路径是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型目录未找到: {args.model_path}")
        exit(1)
    
    if args.input:
        # 处理单个文件
        run_analysis(
            input_file=args.input,
            output_dir=args.output_dir,
            model_path=args.model_path,
            batch_size=args.batch_size,
            time_ranges=parsed_time_ranges
        )
    else:
        # 处理整个目录
        process_all_files(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            batch_size=args.batch_size,
            time_ranges=parsed_time_ranges
        )