import pandas as pd
import json

# Read the JSON file
with open('sp500.json', 'r') as file:
    data = json.load(file)

# Convert JSON data to DataFrame
df = pd.DataFrame(data)

df.to_csv('sp500_raw.csv', index=False)

print("Data has been successfully saved from sp500.json to sp500_raw.csv")




# Read sp500_raw.csv (baseline data)
df_base = pd.read_csv('sp500_raw.csv')

# Read aggregated_news_emotion_results.csv (sentiment score data)
df_news = pd.read_csv('aggregated_news_emotion_results.csv')

# Ensure the 'date' columns are datetime type for correct matching
df_base['date'] = pd.to_datetime(df_base['date'])
df_news['date'] = pd.to_datetime(df_news['date'])

# Merge the two DataFrames using a left join, keeping all rows from sp500_gujia
merged_df = pd.merge(df_base, df_news[['date', 'score']], on='date', how='left')

# Optional: Save the result to a new file
merged_df.to_csv('sp500_data.csv', index=False, encoding='utf-8-sig')

print("Data has been successfully merged and saved to sp500_with_sentiment_score.csv")

