import os
import sys
import requests
from datetime import datetime, timedelta

def read_api_key_from_env_file():
    home = os.path.expanduser("~")
    env_path = os.path.join(home, "ARENA_evals", ".env")
    
    try:
        with open(env_path, "r") as file:
            for line in file:
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=")[1].strip()
    except FileNotFoundError:
        print(f"Error: .env file not found at {env_path}")
        sys.exit(1)
    except IOError:
        print(f"Error: Unable to read .env file at {env_path}")
        sys.exit(1)
    
    print("Error: OPENAI_API_KEY not found in .env file")
    sys.exit(1)

def check_openai_quota(api_key):
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    # Use a fixed current date to avoid issues with system clock
    current_date = datetime(2023, 9, 17)  # Use a known valid date
    end_date = current_date.strftime("%Y-%m-%d")
    start_date = (current_date - timedelta(days=100)).strftime("%Y-%m-%d")
    
    url = f"https://api.openai.com/v1/usage?start_date={start_date}&end_date={end_date}"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        total_usage = sum(item['total_usage'] for item in data['data'])
        # Convert from cents to dollars
        total_usage_dollars = total_usage / 100
        
        print(f"Total usage from {start_date} to {end_date}: ${total_usage_dollars:.2f}")
        print("Note: This script shows usage, not the remaining quota.")
        print("Check your account page for specific quota information.")
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        if response.status_code == 401:
            print("This may be due to an invalid API key. Please check your API key in the .env file.")
        sys.exit(1)

if __name__ == "__main__":
    api_key = read_api_key_from_env_file()
    check_openai_quota(api_key)
