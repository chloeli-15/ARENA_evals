# %%

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
# %%
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
# %%
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
# %%
@retry_with_exponential_backoff
def generate_response(client, 
                      model: str, 
                      messages:Optional[List[dict]]=None, 
                      user:Optional[str]=None, 
                      system:Optional[str]=None, 
                      temperature: float = 1, 
                      verbose: bool = False) -> Optional[str]:
    """
    Generate a response using the OpenAI API.

    Args:
        model (str): The name of the OpenAI model to use (e.g., "gpt-4o-mini").
        messages (Optional[List[dict]]): A list of message dictionaries with 'role' and 'content' keys. 
                                         If provided, this takes precedence over user and system args.
        user (Optional[str]): The user's input message. Used if messages is None.
        system (Optional[str]): The system message to set the context. Used if messages is None.
        temperature (float): Controls randomness in output. Higher values make output more random. Default is 1.
        verbose (bool): If True, prints the input messages before making the API call. Default is False.

    Returns:
        str: The generated response from the OpenAI model.

    Note:
        - If both 'messages' and 'user'/'system' are provided, 'messages' takes precedence.
        - The function uses a retry mechanism with exponential backoff for handling transient errors.
        - The client object should be properly initialized before calling this function.
    """

    if model != "gpt-4o-mini":
        warnings.warn(f"Warning: The model '{model}' is not 'gpt-4o-mini'.")
    ""
    if messages is None:
        messages = apply_message_format(user, system=system)

    if verbose:
        for message in messages:
            print(messages["content"])


    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature
    )
    
    return response.choices[0].message.content

response = generate_response(client, "gpt-4o-mini", user="What is the capital of France?")
print(response)
# %%
# See questions from the anthropic dataset 
anthropic_dataset_name = "power-seeking-inclination" # Choose a dataset
dataset = load_jsonl(f"data/anthropic/{anthropic_dataset_name}.jsonl")
question_sample = random.sample(dataset, 5)
pretty_print_questions(question_sample)
# %%
# See how the model responds to the questions
for question in question_sample:
    response = generate_response(client, "gpt-4o-mini", user=question["question"])
    print(f"Question: {question['question']}\n\nModel Answer: {response}\n")
# %%
my_questions = import_json("mcq_examples.json")

# Ask the model to answer the questions without answer options
for question in my_questions:
    response = generate_response(client, "gpt-4o-mini", user=question["question"])
    print(f"Question: {question['question']}\n\nModel Answer: {response}\n")

# Ask the model to answer the questions with answer options
for question in my_questions:
    question_text = question["question"] + "\nChoices:\n"
    for option, answer in question["answers"].items():
        question_text += f"\n({option}) {answer}"
    response = generate_response(client, "gpt-4o-mini", user=question_text)
    print(f"Question: {question['question']}\n\nModel Answer: {response}\n")
# %%

