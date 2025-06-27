import os
from pprint import pprint

import schedule
import time
import requests
from utils import fetch_historical_data, calculate_indicators, plot_chart
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
MODEL_URL = os.getenv('MODEL_URL')

# Function to analyze the chart using the image-to-text model
def analyze_chart_with_llm(image_path):
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    files = {
        "image": open(image_path, "rb")
    }
    data = {
        "parameters": {
            "max_new_tokens": 512
        }
    }
    response = requests.post(MODEL_URL, headers=headers, files=files, json=data)
    pprint(response)
    if response.status_code == 200:
        result = response.json()
        analysis = result.get('generated_text', '')
        print("LLM Analysis:")
        print(analysis)
        return analysis
    else:
        print(f"Error analyzing chart: {response.status_code} - {response.text}")
        return None

# Function to capture and analyze the chart
def capture_and_analyze_chart():
    df = fetch_historical_data()
    df = calculate_indicators(df)
    image_path = plot_chart(df)
    analysis = analyze_chart_with_llm(image_path)
    if analysis:
        # Optionally, save the analysis to a file
        with open(f'logs/analysis_{int(time.time())}.txt', 'w') as f:
            f.write(analysis)

# Schedule the function to run every 10 minutes
schedule.every(1).minutes.do(capture_and_analyze_chart)

# Run the scheduler
if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(1)
