import json
import csv
import os
import re
from datetime import datetime
from urllib.parse import urlparse

def extract_links(json_data):
    """自动检测并提取不同JSON结构中的URL列表"""
    # 处理嵌套结构（包含"articles"键）
    if isinstance(json_data, dict) and "articles" in json_data and isinstance(json_data["articles"], list):
        return json_data["articles"]
    
    # 处理简单列表结构
    elif isinstance(json_data, list):
        return json_data
    
    # 尝试其他可能的键名
    possible_keys = ["urls", "links", "items", "data"]
    for key in possible_keys:
        if isinstance(json_data, dict) and key in json_data and isinstance(json_data[key], list):
            return json_data[key]
    
    # 如果无法识别结构，尝试提取所有包含"http"的字符串
    if isinstance(json_data, dict):
        all_links = []
        for value in json_data.values():
            if isinstance(value, str) and value.startswith("http"):
                all_links.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.startswith("http"):
                        all_links.append(item)
        if all_links:
            return all_links
    
    # 最后尝试直接处理
    if isinstance(json_data, list) and len(json_data) > 0 and isinstance(json_data[0], str) and json_data[0].startswith("http"):
        return json_data
    
    raise ValueError("无法识别的JSON结构")

def extract_url_info(url):
    """从Bloomberg URL中提取日期和标题信息"""
    try:
        # 解析URL路径
        path = urlparse(url).path
        
        # 使用正则表达式匹配Bloomberg标准URL格式
        pattern = r'/news/articles/(\d{4}-\d{2}-\d{2})/([^/?]+)'
        match = re.search(pattern, path)
        
        if match:
            date_str = match.group(1)
            title_slug = match.group(2)
            
            # 将日期字符串转换为标准日期对象
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            # 将标题slug转换为可读标题
            title = title_slug.replace('-', ' ').title()
            
            return date_obj, title, title_slug
        
        # 尝试其他可能的URL格式
        if '/news/articles/' in url:
            parts = path.split('/')
            if len(parts) >= 4:
                date_str = parts[3]
                title_slug = parts[4] if len(parts) > 4 else "unknown"
                
                # 验证日期格式
                if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                    title = title_slug.replace('-', ' ').title()
                    return date_obj, title, title_slug
        
        return None, "无法解析标题", "无法解析日期"
    
    except Exception as e:
        print(f"解析URL错误: {url} - {str(e)}")
        return None, "解析错误", "解析错误"

def process_json_file(input_path, output_csv):
    """处理单个JSON文件并提取URL信息"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"已加载: {input_path}")
        
        all_links = extract_links(data)
        print(f"发现 {len(all_links)} 个链接")
        
        # 创建CSV文件并写入标题
        file_exists = os.path.exists(output_csv)
        with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(['原始URL', '日期', '标题', '标题Slug', '文件名'])
            
            # 处理每个链接
            success_count = 0
            for i, url in enumerate(all_links):
                if not isinstance(url, str) or not url.startswith("http"):
                    continue
                
                date_obj, title, title_slug = extract_url_info(url)
                filename = os.path.basename(input_path)
                
                # 写入CSV
                writer.writerow([
                    url,
                    date_obj.strftime("%Y-%m-%d") if date_obj else "",
                    title,
                    title_slug,
                    filename
                ])
                success_count += 1
        
        print(f"成功提取 {success_count} 条记录并保存到 {output_csv}")
        return success_count
    
    except json.JSONDecodeError:
        print(f"错误: {input_path} 不是有效的JSON文件")
        return 0
    except Exception as e:
        print(f"处理文件时出错: {input_path} - {str(e)}")
        return 0

if __name__ == "__main__":
    # 配置输入输出
    input_files = [
        "/Users/wangbaihui/Bloomberg_ai_links.json",
        # 添加更多文件...
    ]
    output_csv = "/Users/wangbaihui/url_info.csv"
    
    # 处理每个文件
    total_success = 0
    for input_file in input_files:
        if os.path.exists(input_file):
            print(f"\n{'='*50}")
            print(f"处理文件: {input_file}")
            total_success += process_json_file(input_file, output_csv)
        else:
            print(f"文件不存在: {input_file}")
    
    print(f"\n所有文件处理完成! 总共提取了 {total_success} 条URL信息到 {output_csv}")