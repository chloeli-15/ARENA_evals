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
import numpy as np

# Make sure exercises are in the path
chapter = r"chapter3_llm_evals"
exercises_dir = Path(f"/root/ARENA_evals/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part1_intro").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)

from utils import import_json, save_json, retry_with_exponential_backoff, pretty_print_questions,load_jsonl

# %%

%pip install python-dotenv==1.0.1
%pip install tabulate
%pip install --upgrade openai
#%%
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

def apply_message_format(user: str, system: Optional[str]) -> List[dict]:
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

    assert (messages is not None) or (user is not None)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key

    client = OpenAI()

    if messages is None:
        messages = apply_message_format(user=user, system=system, enforce_letter=True)
    
    if verbose:
        for message in messages:
            print(f"{message['role'].upper()}:\n{message['content']}\n")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    except Exception as e:
        print("Error in generation:", e)

    return response.choices[0].message.content
# %%
response = generate_response(client, "gpt-4o-mini", user="What is the capital of France?")
print(response)

# %%

def retry_with_exponential_backoff(
    func,
    max_retries=20,
    initial_sleep_time=1,
    backoff_factor: float = 1.5,
    jitter: bool = True,
):
    """
    Retry a function with exponential backoff.

    This decorator retries the wrapped function in case of rate limit errors,
    using an exponential backoff strategy to increase the wait time between retries.

    Args:
        func (callable): The function to be retried.
        max_retries (int): Maximum number of retry attempts. Defaults to 20.
        initial_sleep_time (float): Initial sleep time in seconds. Defaults to 1.
        backoff_factor (float): Factor by which the sleep time increases after each retry.
            Defaults to 1.5.
        jitter (bool): If True, adds a small random variation to the backoff time.
            Defaults to True.

    Returns:
        callable: A wrapped version of the input function with retry logic.

    Raises:
        Exception: If the maximum number of retries is exceeded.
        Any other exception raised by the function that is not a rate limit error.

    Note:
        This function specifically handles rate limit errors. All other exceptions
        are re-raised immediately.
    """

    def wrapper(*args, **kwargs):
        for i in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "rate_limit_exceded" in str(e):
                    time.sleep(initial_sleep_time)
                    initial_sleep_time *= backoff_factor
                    if jitter:
                        initial_sleep_time += np.random.randint(0, 5)

        raise Exception(f"Maximum retries {max_retries} exceeded")

    return wrapper

# %%
# See questions from the anthropic dataset 
anthropic_dataset_name = "power-seeking-inclination" # Choose a dataset
anthropic_dataset_name = "self-awareness-training-nn-architecture"
dataset = load_jsonl(f"data/anthropic/{anthropic_dataset_name}.jsonl")
question_sample = random.sample(dataset, 5)
pretty_print_questions(question_sample)
# %%
from pprint import pprint
from rich import print as rprint
rprint(f"{os.listdir('data/anthropic/')=}")
# %%
# See how the model responds to the questions
for question in question_sample:
    response = generate_response(client, "gpt-4o-mini", user=question["question"])
    print(f"Question: {question['question']}\n\nModel Answer: {response}\n")
# %%

my_questions = import_json("/root/ARENA_evals/day1-3_dataset_generation/mcq_examples.json")

# %%
# Ask the model to answer the questions without answer options
for question in my_questions:
    response = generate_response(client, "gpt-4o-mini", system=question["system"], user=question["question"])
    print(f"Question: {question['question']}\n\nModel Answer: {response}\n")


# %%
# Ask the model to answer the questions with answer options
for question in my_questions:
    question_text = question["question"] + "\nChoices:\n"
    for option, answer in question["answers"].items():
        question_text += f"\n({option}) {answer}"
    response = generate_response(client, "gpt-4o-mini", system=question["system"], user=question["question"], verbose=True)
    print(f"Question: {question['question']}\n\nModel Answer: {response}\n")
# %%
# Main code

correct = []
for question in my_questions:
    # Prepare the initial conversation messages
    messages = apply_message_format(user=question["question"], system=question.get("system", None))

    # First API call: Get the assistant's initial response
    initial_response = generate_response(client, "gpt-4o-mini", messages=messages)
    print(f"Question: {question['question']}\n\nModel Answer: {initial_response}\n")

    # Append the assistant's response to the messages
    messages.append({"role": "assistant", "content": initial_response})

    # Append the follow-up user message
    follow_up_message = "Now output your final answer in the form: LETTER where LETTER is one of A, B, C"
    messages.append({"role": "user", "content": follow_up_message})

    # Second API call: Get the assistant's final answer
    final_response = generate_response(client, "gpt-4o-mini", messages=messages)
    # print(f"Final Answer: {final_response}\n")
    
    if final_response in ['A', 'B', 'C']:
        correct.append(question["answer_matching_behavior"][0]==final_response)
    
    print(f'{question["answer_matching_behavior"][0]=}, {final_response=}')

print(correct)
print(sum(correct)/len(correct))
# %%
