#%%
import openai
from openai import OpenAI
import os
import rubric 
from utils import import_json_as_string, import_json, establish_client_OpenAI
from dotenv import load_dotenv
import json
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
from typing import Optional
#%%
def reload_rubric():
    global rubric
    importlib.reload(rubric)
    return rubric.RUBRIC

def generate_model_score(question:str, rubric, examples, model_name):
    """
    For a given eval question to score, generate a message that combines the rubric and few-shot examples to feed into the model, and get the model's response on that question. 
    """

    # Create message 
    messages = [{"role": "system", "content": rubric}]
    messages.extend(examples)
    messages.extend([{"role": "user", "content": question}])
    print("Messages:", messages)

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
    print("Response:", response)
    print("Score:", score, type(score))
    return score, response


def query_evaluator(dataset_to_evaluate:str, model_name:str, rubric_name:str, dataset_path: Optional[str] = None):
    """
    For a given dataset to evaluate, splits the dataset into chunks/batches and apply LM calls on each chunck in parallel using multi-threading 
    """
    print("EVALUATING GENERATED QUESTIONS....")
    RUBRIC = reload_rubric()
    rubric = RUBRIC[rubric_name][0]
    examples = RUBRIC[rubric_name][1]
    if dataset_path: dataset = import_json(dataset_path)
    else: dataset = import_json(f"datasets/{dataset_to_evaluate}.json")
    print(f"Dataset to evaluate ({len(dataset)}):", dataset)
    graded_dataset = []
    date = time.strftime("%Y-%m-%d")
    graded_filepath = f"scores/{rubric_name}/{dataset_to_evaluate}-{date}.json"

    # Process each chunk of the dataset
    def process_chunk(chunk_id):
        start = chunk_id * config.chunk_size
        end = start + config.chunk_size
        chunk = dataset[start:end].copy()

        print(f"Processing chunk {chunk_id} with {len(chunk)} questions...")
        # Apply LM calls to each question in the chunk
        for q in chunk:
            q["score"], q["model_response"] = generate_model_score(str(q), rubric, examples, model_name)
        
        if all(q["score"] is not None for q in chunk): #need to change this to check if all questions have been graded
            graded_dataset.extend(chunk)

    # Process each chunk in parallel 
    """NEED TO EXPLAIN THIS SECTION"""
    with ThreadPoolExecutor(max_workers=config.max_threads) as executor:
        total_chunks = math.ceil(len(dataset) / config.chunk_size )
        print("Total questions:", len(dataset))
        print("Total chunks:", total_chunks)
        start_time = time.time() #for testing purpose; delete later
        executor.map(process_chunk, range(total_chunks))
        end_time = time.time() #for testing purpose; delete later
        elapsed_time = end_time - start_time #for testing purpose; delete later
        print("Elapsed time:", elapsed_time) #for testing purpose; delete later
    
    # Save graded dataset
    with open(graded_filepath, 'w') as f:
        json.dump(graded_dataset, f)
    
    print(f"{len(graded_dataset)} GRADED QUESTIONS SAVED TO {graded_filepath}")
    return graded_dataset, graded_filepath

def summarize_results(rubric_name:str, model_name:str, dataset_path:str):
    """
    Calculate summary statistics for the results of the evaluation.
    """
    result_filenames = glob.glob(dataset_path)
    print("Result filenames:", result_filenames)

    logs =[]
    for filename in result_filenames:
        results = import_json(filename)
        print("Results", results)
        scores = [q["score"] for q in results]
        log = {}
        log["dataset"] = dataset_path
        log["n_questions"] = len(scores)
        log["model_evaluator"] = model_name
        log["rubric"] = rubric_name
        log["mean"] = sum(scores) / len(scores)
        log["max"] = max(scores)
        log["min"] = min(scores)
        log["std"] = pd.Series(scores).std()
        log["median"] = pd.Series(scores).median()
        log["label_balance"] = check_label_balance(filename)
        log["answer_balance"] = check_answer_balance(filename)
        
        # Save log
        log_file = filename.replace(".json", "-log.json").replace("scores","scores_log")
        with open(log_file, 'w') as f:
            json.dump(log, f)
        logs.append(log)
    return logs


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

#%%
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    openai.api_key = api_key
    client = OpenAI()
    
    model_name = "gpt-4o"
    datasets_to_evaluate = ["test-questions"]
    rubric_name = "quality"

    for dataset_to_evaluate in datasets_to_evaluate:
        query_evaluator(dataset_to_evaluate, model_name, rubric_name)

        summarize_results(rubric_name, dataset_to_evaluate)



# %%
