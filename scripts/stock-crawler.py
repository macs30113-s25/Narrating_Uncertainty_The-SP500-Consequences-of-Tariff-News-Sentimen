import yfinance as yf
import json
import boto3
from datetime import datetime

S3_BUCKET = 'assignment7-charlotte'
S3 = boto3.client('s3')

def fetch_sp500_prices(start_date, end_date):
    print(f'[INFO] Fetching S&P 500 data from {start_date} to {end_date}')

    data = yf.download('^GSPC', start=start_date, end=end_date, auto_adjust=False)
    if data.empty:
        print('[WARNING] No data fetched.')
        return None
    return data

def upload_to_s3(key, data):
    """
    Uploads data to an S3 bucket in JSON format.
    """
    try:
        S3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'),
            ContentType='application/json'
        )
        print(f'[INFO] Uploaded to s3://{S3_BUCKET}/{key}')
    except Exception as e:
        print(f'[ERROR] Failed to upload to S3: {e}')

def main():
   #year = datetime.now().year 
    year = 2025
    start_date = f'{year}-01-01'
    end_date = f'{year}-05-20'

    data = fetch_sp500_prices(start_date, end_date)
    if data is None:
        return

    # Convert data format to list of dictionaries
    records = []
    for idx, row in data.iterrows():
        records.append({
            'date': idx.strftime('%Y-%m-%d'),
            'open': float(row['Open']),
            'high': float(row['High']),
            'low': float(row['Low']),
            'close': float(row['Close']),
            'adj_close': float(row['Adj Close']),
            'volume': int(row['Volume'])
        })

    # Uplode the json file to S3
    s3_key = f'raw/stock/sp500_{start_date}_to_{end_date}.json'
    upload_to_s3(s3_key, records)

if __name__ == '__main__':
    main()
