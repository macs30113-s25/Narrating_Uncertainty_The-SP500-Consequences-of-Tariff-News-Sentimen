import json
import boto3
import urllib.parse
from datetime import datetime

s3_client = boto3.client('s3')
sagemaker_client = boto3.client('sagemaker-runtime')

ENDPOINT_NAME = 'finbert-endpoint'
OUTPUT_BUCKET = 'charlotte-5433'

def lambda_handler(event, context):
    try:
        record = event['Records'][0]
        source_bucket = record['s3']['bucket']['name']
        source_key = urllib.parse.unquote_plus(record['s3']['object']['key'])

        response = s3_client.get_object(Bucket=source_bucket, Key=source_key)
        file_content = response['Body'].read().decode('utf-8')
        data = json.loads(file_content)

        results = []
        for item in data:
            content = item.get('summary', '')
            if not content:
                continue

            payload = {"inputs": content}
            sm_response = sagemaker_client.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            prediction = json.loads(sm_response['Body'].read().decode())

            results.append({
                'source': item.get('source'),
                'title': item.get('title'),
                'published': item.get('published'),
                'inference_result': prediction
            })

        output_key = f'inference-results/{source_key.replace(".json", "_results.json")}'

        s3_client.put_object(
            Bucket=OUTPUT_BUCKET,
            Key=output_key,
            Body=json.dumps(results, indent=2).encode('utf-8'),
            ContentType='application/json'
        )

        return {
            'statusCode': 200,
            'body': f'Successfully processed and uploaded results to s3://{OUTPUT_BUCKET}/{output_key}'
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': f'Error: {str(e)}'
        }
