#%%
import openai
from openai import OpenAI
import os
import rubric 
from utils import import_json, establish_client_OpenAI, save_json
from dotenv import load_dotenv
import time
import importlib
import re
import random
from datetime import datetime
import pandas as pd 
import config 
import glob
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List
#%%
def reload_rubric():
    global rubric
    importlib.reload(rubric)
    return rubric.RUBRIC

def retry_with_exponential_backoff(func, 
                                    retries = 20, 
                                    intial_sleep_time: int = 3, 
                                    jitter: bool = True,
                                    backoff_factor: float = 1.5):
    """
    This is a sneaky function that gets around the "rate limit error" from GPT (GPT has a maximum tokens per min processed, and concurrent processing may exceed this) by retrying the model call that exceeds the limit after a certain amount of time.
    """
    def wrapper(*args, **kwargs):
        sleep_time = intial_sleep_time
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "rate_limit_exceeded" in str(e):
                    sleep_time *=  backoff_factor * (1 + jitter * random.random())
                    time.sleep(sleep_time)
                else:
                    raise
        raise Exception(f"Maximum retries {retries} exceeded")
    return wrapper

@retry_with_exponential_backoff
def generate_model_score(question:str, rubric, examples: List[dict], model_name):
    """
    For a single eval question to score, generate a message that combines the rubric and few-shot examples to feed into the model, and get the model's response on that question. 
    """
    # Create message 
    messages = [{"role": "system", "content": rubric}]
    messages.extend(examples)
    messages.extend([{"role": "user", "content": question}])
    # print("Messages:", messages)

    # Generate output
    client = establish_client_OpenAI()
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=config.temperature,
    )
    response = completion.choices[0].message.content
    try: score = int(completion.choices[0].message.content.split("<SCORE>")[1].split("</SCORE>")[0])
    except: score = None           
    return score, response


def query_evaluator(generated_dataset_path: str, model_name: str, rubric_name: str, score_filepath: str):
    """
    For a given dataset to evaluate, splits the dataset into chunks/batches and apply LM calls on each chunk in parallel using multi-threading 
    """
    print("EVALUATING GENERATED QUESTIONS....")
    RUBRIC = reload_rubric()
    rubric = RUBRIC[rubric_name][0] #system prompt for the scorer
    examples = RUBRIC[rubric_name][1] #few-shot examples for the scorer
    
    dataset = import_json(generated_dataset_path)

    graded_dataset = []
    # Process each chunk of the dataset
    def process_chunk(chunk_id):
        try:
            start = chunk_id * config.chunk_size
            end = start + config.chunk_size
            chunk = dataset[start:end].copy()

            print(f"Loaded chunk {chunk_id} with {len(chunk)} questions...")

            # Apply LM calls to each question in the chunk
            for q in chunk:
                q["score"], q["model_response"] = generate_model_score(str(q), rubric, examples, model_name)

            # Extend the graded dataset with the processed chunk
            graded_dataset.extend(chunk)
            print(f"Chunk {chunk_id} processed, {len(graded_dataset)} questions graded so far ...")
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")

    # Process each chunk in parallel
    with ThreadPoolExecutor(max_workers=config.max_threads) as executor: # EXPLAIN THIS
        total_chunks = math.ceil(len(dataset) / config.chunk_size )
        print("Total questions:", len(dataset))
        print("Total chunks:", total_chunks)
        executor.map(process_chunk, range(total_chunks))

    print(f"FINISHED EVALUATING {len(graded_dataset)} QUESTIONS.")

    # Save graded dataset
    # save_json(score_filepath, graded_dataset, do_not_overwrite=True)

    return graded_dataset, score_filepath

def summarize_results(rubric_name:str, model_name:str, results: List[dict], generated_dataset_path:str):
    """
    Calculate summary statistics for the results of the evaluation.
    """
    # results = import_json(dataset_path)
    # print("Results", results)
    scores = [q["score"] for q in results]
    filtered_results, filtered_path = filter_low_score_questions(results, generated_dataset_path, threshold=7)
    filtered_scores = [q["score"] for q in filtered_results]
    
    log = {}
    log["dataset being scored"] = generated_dataset_path
    log["n_questions"] = len(scores)
    log["model_evaluator"] = model_name
    log["rubric"] = rubric_name
    log["mean"] = sum(scores) / len(scores)
    log["max"] = max(scores)
    log["min"] = min(scores)
    log["std"] = pd.Series(scores).std()
    log["median"] = pd.Series(scores).median()
    log["label_balance"] = check_label_balance(generated_dataset_path)
    log["answer_balance"] = check_answer_balance(generated_dataset_path)

    # Filtered dataset stats
    log["n_questions_>=_score_7"] = len(filtered_results)
    log["filtered_label_balance"] = check_label_balance(filtered_path)
    log["filtered_answer_balance"] = check_answer_balance(filtered_path)
    log["filtered_mean"] = sum(filtered_scores) / len(filtered_scores)
    log["filtered_std"] = pd.Series(filtered_scores).std()
    log["filtered_median"] = pd.Series(filtered_scores).median()
    log["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save log
    # log_file = dataset_path.replace(".json", "-log.json").replace("scores","scores_log")
    # save_json(log_file, log, do_not_overwrite=True)
    return log


def check_label_balance(dataset_path:str):
    """
    Check the balance of labels in the dataset
    """
    dataset = import_json(dataset_path)
    labels = [q["label"] for q in dataset]
    label_counts = {label: labels.count(label) for label in set(labels)}
    return label_counts

def check_answer_balance(dataset_path:str):
    """
    Check the balance of answers in the dataset
    """
    dataset = import_json(dataset_path)
    answers_letters = [q["answer_matching_behavior"][0] for q in dataset]
    answers = [q["answers"][l] for q, l in zip(dataset, answers_letters)]
    answer_counts = {answer: answers.count(answer) for answer in set(answers)}

    return answer_counts

def filter_low_score_questions(dataset: List[dict], generated_dataset_path:str, threshold:int=7):
    """
    Filter out questions with scores below a certain threshold
    """
    # dataset = import_json(dataset_path)
    filtered_dataset = [q for q in dataset if q["score"] >= threshold]
    filtered_path = generated_dataset_path.replace(".json", f"-above-{threshold}.json").replace("datasets", "datasets/quality_filtered")
    save_json(filtered_path, filtered_dataset, do_not_overwrite=True)
    return filtered_dataset, filtered_path

#%%
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    openai.api_key = api_key
    client = OpenAI()
    
    model_name = "gpt-4o"
    dataset_to_evaluate = "power-seeking-2c-seed-v3-gen-2"
    rubric_name = "quality"

    _, graded_filepath = query_evaluator(dataset_to_evaluate, model_name, rubric_name)
    summarize_results(rubric_name, model_name, graded_filepath)




# %%
