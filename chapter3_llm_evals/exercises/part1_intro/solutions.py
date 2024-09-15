#%%
import os
from pathlib import Path
import sys
import openai
from openai import OpenAI
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional 
import random
import warnings
import time

# Make sure exercises are in the path
chapter = r"chapter3_llm_evals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part1_intro").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)

from utils import import_json, save_json, retry_with_exponential_backoff, pretty_print_questions,load_jsonl

MAIN = __name__ == '__main__'
# %% # Intro to API Calls
if MAIN:
    # Configure your API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    )

    print(response, "\n")  # See the entire ChatCompletion object
    print(response.choices[0].message.content)  # See the response message only

def apply_system_format(content : str) -> dict:
    return {
        "role" : "system",
        "content" : content
    }

def apply_user_format(content : str) -> dict:
    return {
        "role" : "user",
        "content" : content
    }

def apply_assistant_format(content : str) -> dict:
    return {
        "role" : "assistant",
        "content" : content
    }

def apply_message_format(user : str, system : Optional[str]) -> List[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages

#%%

@retry_with_exponential_backoff
def generate_response(client, model: str, messages:Optional[List[dict]]=None, user:Optional[str]=None, system:Optional[str]=None, temperature: float = 1, verbose: bool = False) -> str:
    """
    Generate a response to the `messages` from the OpenAI API.

    Args:
        model (str): model name
        messages (dict): array of formatted messages with role and content
        user (str): user message (alternative to `messages`)
        system (str): system message (alternative to `messages`)
    """

    # Initialize messages if not provided
    if messages is None:
        messages = apply_message_format(user=user, system=system)
    
    # Print messages if verbose
    if verbose:
        for message in messages:
            print(f"{message['role'].upper()}:\n{message['content']}\n")
      
    # API call
    response = client.chat.completions.create(
        model=model, 
        messages=messages, 
        temperature=temperature
    )
    return response.choices[0].message.content

if MAIN:
    response = generate_response(client, "gpt-4o-mini", user="What is the capital of France?")
    print(response)

#%%
def retry_with_exponential_backoff(
    func,
    max_retries=20,
    initial_sleep_time=1,
    backoff_factor: float = 1.5,
    jitter: bool = True,
):
    """
    Retry a function with exponential backoff

    Args:
    func: function to retry
    max_retries: maximum number of retries
    intial_sleep_time: initial sleep time
    backoff_factor: factor to increase sleep time by
    jitter: if True, randomly vary the backoff_factor by a small amount

    """

    def wrapper(*args, **kwargs):
        sleep_time = initial_sleep_time

        for _ in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "rate_limit_exceeded" in str(e):
                    sleep_time *= backoff_factor * (1 + jitter * random.random())
                    time.sleep(sleep_time)
                else:
                    raise
        raise Exception(f"Maximum retries {max_retries} exceeded")

    return wrapper

#%%
if MAIN:
    # See questions from the anthropic dataset 
    anthropic_dataset_name = "power-seeking-inclination" # Choose a dataset
    dataset = load_jsonl(f"data/anthropic/{anthropic_dataset_name}.jsonl")
    question_sample = random.sample(dataset, 5)
    pretty_print_questions(question_sample)
# %%
