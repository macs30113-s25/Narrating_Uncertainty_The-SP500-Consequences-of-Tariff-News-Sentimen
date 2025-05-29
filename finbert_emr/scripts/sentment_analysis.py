import sys
import json
import torch
import boto3
from pyspark.sql import SparkSession, functions as F
from transformers import BertTokenizer, BertForSequenceClassification
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType, FloatType

def load_model(model_path):
    """加载 FinBERT 模型"""
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def analyze_sentiment(text, tokenizer, model):
    """分析文本情感"""
    try:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
        
        return {
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2]),
            "sentiment": "positive" if probs[2] > probs[0] and probs[2] > 0.5 else 
                         "negative" if probs[0] > probs[2] and probs[0] > 0.5 else "neutral"
        }
    except Exception as e:
        print(f"Error processing text: {e}")
        return {
            "negative": 0.0,
            "neutral": 0.0,
            "positive": 0.0,
            "sentiment": "error"
        }

if __name__ == "__main__":
    input_path = sys.argv[1]  
    output_path = sys.argv[2]  
    model_path = sys.argv[3]  
    text_column = sys.argv[4] if len(sys.argv) > 4 else "content"  
    
    spark = SparkSession.builder \
        .appName("FinBERT-CSV-Sentiment-Analysis") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    
    
    tokenizer, model = load_model(model_path)
    
    
    sentiment_schema = StructType([
        StructField("negative", FloatType()),
        StructField("neutral", FloatType()),
        StructField("positive", FloatType()),
        StructField("sentiment", StringType())
    ])
    
    sentiment_udf = udf(
        lambda text: analyze_sentiment(text, tokenizer, model), 
        sentiment_schema
    )
    
    
    print(f"Reading CSV data from: {input_path}")
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .option("quote", "\"") \
        .option("escape", "\"") \
        .csv(input_path)
    
    
    print(f"Total records: {df.count()}")
    print("Data schema:")
    df.printSchema()
    
    
    if text_column not in df.columns:
        print(f"Error: Text column '{text_column}' not found in data")
        print("Available columns:", df.columns)
        spark.stop()
        sys.exit(1)
    
   
    print("Processing sentiment analysis...")
    result_df = df.withColumn("sentiment_result", sentiment_udf(col(text_column)))
    
    
    result_df = result_df.select(
        "*",  
        col("sentiment_result.negative").alias("negative_score"),
        col("sentiment_result.neutral").alias("neutral_score"),
        col("sentiment_result.positive").alias("positive_score"),
        col("sentiment_result.sentiment").alias("sentiment")
    )
    
    
    sentiment_distribution = result_df.groupBy("sentiment").count()
    print("Sentiment distribution:")
    sentiment_distribution.show()
    
    
    result_df.write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(f"{output_path}/results")
    
    sentiment_distribution.write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(f"{output_path}/summary")
    
    print(f"Results saved to: {output_path}")
    
   
    report = {
        "total_records": result_df.count(),
        "positive_count": result_df.filter(col("sentiment") == "positive").count(),
        "negative_count": result_df.filter(col("sentiment") == "negative").count(),
        "neutral_count": result_df.filter(col("sentiment") == "neutral").count(),
        "error_count": result_df.filter(col("sentiment") == "error").count(),
        "columns": df.columns,
        "text_column": text_column
    }
    
    with open("report.json", "w") as f:
        json.dump(report, f)
    
    
    s3 = boto3.client('s3')
    bucket_name = output_path.split("/")[2]
    s3_path = "/".join(output_path.split("/")[3:]) if len(output_path.split("/")) > 3 else ""
    s3.upload_file("report.json", bucket_name, f"{s3_path}/report.json")
    
    print("Analysis completed successfully")
    spark.stop()