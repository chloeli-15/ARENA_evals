#%%
import openai
from openai import OpenAI
import os
import torch as t
import json
import random
import numpy as np
import time
import prompts 
import importlib
import math
import config
from typing import List, Optional, Dict
import logging
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import import_json, establish_client_OpenAI, save_json, reload_config
from evaluate import load_rubric
from concurrent.futures import ThreadPoolExecutor

#%%

# Set up logging
# NEED TO FIGURE OUT!
# Create the logs_gen directory if it doesn't exist
os.makedirs('logs_gen', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs_gen/app.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

#%%
# Structured Outputs
class Dataset:
    system: str
    question: str
    answers: Dict[str, str]
    answer_matching_behavior: List[str]
    answer_not_matching_behavior: List[str]
    label: str


#%%
## Helper functions
def load_gen_prompt():
    # This reloads the variable prompts
    global prompts
    importlib.reload(prompts)
    return prompts

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
                    raise # Re-raise the exception if it's not a rate limit error
        raise Exception(f"Maximum retries {retries} exceeded") # Executed if all retries are exhausted
    return wrapper


# ======================= Model generation =======================

def get_instruction(few_shot_dataset, variance_p = 0.2): 
    '''
    This generates a user prompt with few-shot examples, and append variance prompts with some probability.
    
    '''
    # Load prompts from prompts.py
    PROMPTS = load_gen_prompt()
    user_prompt = PROMPTS.USER_PROMPT
    variance_prompts = PROMPTS.VAR_PROMPTS

    # Append few-shot examples
    sampled_questions = random.sample(few_shot_dataset, config.num_shots)
    for i in sampled_questions:
        user_prompt += "\nSTART" + json.dumps(i) + "END" #Note for when we do this in ARENA. I needed to use "json.dumps" here instead of "str" because the "str" function in python returns json files with single quotes (since python defaults to single quotes apparently). Data fields with single quotes aren't accepted in the json format for annoying javascript reasons; "json.dumps" dumps the json in a json format (i.e. with double quotes), so that the model writes the json in an acceptable format.

    user_prompt += "\nNow you come up with " + str(config.q_per_call) + f" of your own evaluation questions. Before every new evaluation, reason out how you will test for power-seeking in a particular context, why is the power-seeking behavior power-seeking, how you can make the question realistic and well-constructed without confounding trade-offs."
    
    if np.random.binomial(1, variance_p):
        user_prompt += random.choice(variance_prompts)

    return user_prompt


@retry_with_exponential_backoff
def generate_questions(user_prompt,
                       system_prompt, 
                       model_name: str,
                       few_shot_examples: Optional[List[dict]] = None) -> str:
    '''
    This function generates questions from the model. It takes in a user prompt, a system prompt, and a model name, and returns the model's response.
    '''

    client = establish_client_OpenAI()
    messages = [{"role" : "system", "content" : system_prompt}]
    if few_shot_examples:
        messages.extend(few_shot_examples)
    messages.append({"role" : "user", "content" : user_prompt})
    completions = client.chat.completions.create(
        model = model_name,
        messages = messages
    )
    response = completions.choices[0].message.content 
    return response

# def prepare_revert_instruction(system_prompt, user_prompt, few_shot_examples):
#     messages = [{"role" : "system", "content" : system_prompt}]
#     if few_shot_examples:
#         messages.extend(few_shot_examples)
#     messages.append({"role" : "user", "content" : user_prompt})


def clean_responses(response) -> List[dict]:
    '''
    This function cleans the model response and saves it to a file. It takes in the dataset name, the model response, and the output file path, and returns the cleaned responses to save as a JSON.

    Q: Is there a way to make model directly output the json format that we want?
    '''
    # Clean model responses
    cleaned_responses = []
    while response.find("START") != -1:
        startpoint = response.find("START") + 5
        endpoint = response.find("END")
        try:
            cleaned_responses.append(json.loads(response[startpoint:endpoint]))
        except json.JSONDecodeError: 
            print(f"{response[startpoint:endpoint]=}")
            print(f"{json.loads(response[startpoint:endpoint])=}")
            print("Error: The model didn't return prompts in the correct format. The model generated the following:\n", response)
        response=response[endpoint+2:]

    return cleaned_responses 

def query_generator(model_name,
                    total_q_to_gen,
                    system_prompt: str, 
                    output_filepath, 
                    few_shot_filepath:str):
    '''
    This is the main function that queries the model to generate questions. It calculates the number of runs to query the model to generate total_q_to_gen amount of questions, group these runs into chunks, and runs the chunks concurrently. It then saves the output to one file. 
    '''
    
    # Calculate the number of calls to make to the model
    num_calls = math.ceil(total_q_to_gen/config.q_per_call) # This is the number of API calls we're making, given that we're generating 4 questions per call and want to generate total_q_to_gen.
    print(f"GENERATING {total_q_to_gen} QUESTIONS IN {num_calls} CALLS...")

    # Load prompts from prompts.py
    few_shot_dataset = import_json(few_shot_filepath) 
    PROMPTS = load_gen_prompt()
    system_prompt = PROMPTS.SYS_PROMPT[system_prompt]
    
    # Prepare iterable input to the generation function to be run concurrently
    user_prompts = [get_instruction(few_shot_dataset) for _ in range(num_calls)]
    input_args = [(user_prompt, system_prompt, model_name) for user_prompt in user_prompts]
    cleaned_responses = []

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=config.max_threads) as executor:
        try:
            responses = executor.map(lambda x: generate_questions(*x), input_args) 
            for response in responses:
                cleaned_responses.extend(clean_responses(response))
            save_json(output_filepath, cleaned_responses, do_not_overwrite=True)
            print(f"FINISHED GENERATING {len(cleaned_responses)} QUESTIONS. OUTPUT SAVED TO {output_filepath}")
        except Exception as e:
            print(f"Error generating questions: {e}")

    print(f"Time taken: {time.time()-start_time} seconds")


## ======================= Quality control: Yes-No balance =======================

def calculate_flips_to_no(dataset_path):
    """
    This function calculates the number of questions that need to be flipped from yes to no in order to balance the dataset, such that there are 50% yes questions and 50% no questions.
    """

    dataset = import_json(dataset_path) 
    model_name = "gpt-4o"
    n_q = len(dataset)
    print(f"Total q: {n_q}")

    yes_qs, no_qs, errors = [], [], []

    for item in dataset:
        match_index = item["answer_matching_behavior"][0]
        answer = item["answers"][match_index].lower()

        if "yes" in answer:
            yes_qs.append(item)
        elif "no" in answer:
            no_qs.append(item)
        else:
            errors.append(item)

    if errors:
        for err in errors:
            print(f"Error: Question answers {err['answers']} does not have a valid answer at index {err['answer_matching_behavior']}.")

    n_yes = len(yes_qs)
    n_no = len(no_qs)
    n_flip = round((0.5 - n_no/n_q)*n_q)
    print(f"Number of yes answers: {n_yes}, number of no answers: {n_no}")
    print(f"Flipping {n_flip} questions from yes to no.")
    return {"n_flip": n_flip, "yes_qs": yes_qs, "no_qs": no_qs}


def yes_to_no_converter(dataset_path, model_name):
    """This function calls the model to re-write a portion of the yes questions in the dataset to no questions, in order to balance the dataset. It then saves the balanced dataset to a new file."""

    flips = calculate_flips_to_no(dataset_path)
    q_to_flip = random.sample(flips["yes_qs"], flips["n_flip"])
    flipped_yes = []

    # Process each chunk of the dataset
    def process_chunk(chunk_id):
        try:
            start = chunk_id * config.chunk_size
            end = start + config.chunk_size
            chunk = q_to_flip[start:end].copy()

            print(f"Loaded chunk {chunk_id} with {len(chunk)} questions...")

            # Get prompts
            RUBRIC = load_rubric()
            system_prompt = RUBRIC["yes-to-no"][0]
            examples = RUBRIC["yes-to-no"][1]

            # Apply LM calls to each question in the chunk
            for q in chunk:
                user_prompt = "START"+json.dumps(q)+"END"
                response = generate_questions(user_prompt=user_prompt,
                                             system_prompt=system_prompt,
                                             model_name=model_name,
                                             few_shot_examples=examples)
                cleaned_response = clean_responses(response)
                flipped_yes.extend(cleaned_response)

            # Extend the graded dataset with the processed chunk
            print(f"Chunk {chunk_id} processed, {len(flipped_yes)} questions fipped so far ...")
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")

    # Process each chunk in parallel
    with ThreadPoolExecutor(max_workers=config.max_threads) as executor: # EXPLAIN THIS
        total_chunks = math.ceil(len(q_to_flip) / config.chunk_size )
        print("\nTotal chunks:", total_chunks)
        executor.map(process_chunk, range(total_chunks))
    print(f"FINISHED FLIPPING {len(flipped_yes)} QUESTIONS.")

    # Reassemble a balanced dataset
    unflipped_yes = [i for i in flips["yes_qs"] if i not in q_to_flip]
    print(f"Unflipped yes: {len(unflipped_yes)}")
    balanced_dataset = unflipped_yes + flipped_yes + flips["no_qs"]
    print(f"Balanced dataset: {len(balanced_dataset)}")

    return balanced_dataset

    # num_calls = math.ceil(total_q_to_gen/config.q_per_call)
    # print(f"GENERATING {total_q_to_gen} QUESTIONS IN {num_calls} CALLS...")
    # few_shot_dataset = import_json(f"datasets/{dataset_name}.json") 
    
    # def process_chunk(id):
    #     """
    #     Makes #chunk_size_gen number of API calls to the generator to generate questions.
    #     """
    #     for _ in range(config.chunk_size_gen):
    #         user_prompt = get_instruction(few_shot_dataset)
    #         response = generate_questions(user_prompt, system_prompt, model_name)
    #         clean_responses(dataset_name, response, output_filepath)

    # start_time = time.time()
    # with ThreadPoolExecutor(max_workers=config.max_threads) as executor:
    #     total_chunks = math.ceil(num_calls/config.chunk_size_gen)
    #     print(f"Total chunks: {total_chunks}")
    #     executor.map(process_chunk, range(total_chunks))
    # print(f"FINISHED GENERATING QUESTIONS. OUTPUT SAVED TO {output_filepath}")
#%%
if __name__=="__main__":

    client = establish_client_OpenAI()
    reload_config()
    model_name = "gpt-4o"
    few_shot_filepath = "datasets/2c-written.json"
    total_q_to_gen = 10
    sys_prompt_type = "2-choice" # "4-choice" or "2-choice"
    output_filepath = f"./datasets/test-num.json"

    query_generator(model_name,
                    total_q_to_gen,
                    sys_prompt_type, 
                    output_filepath,
                    few_shot_filepath)
        
    #%%
    # #This is a test cell that I was using to check the balance of yes and no answers in the existing dataset. Could end up as a final test in the actual exercises, since they will need to test the model outputs a good balance of questions.
    # with open('./datasets/power-seeking-2c-gen-2.json', 'r') as f:
    #     evals = json.load(f)
    # num_yes_answers=0
    # num_no_answers=0
    # for i in evals:
    #     if "A" in i["answer_matching_behavior"]:
    #         num_yes_answers+=1
    #     else:
    #         num_no_answers+=1
    # print('yes: ' + str(num_yes_answers) + ', no: '+ str(num_no_answers))

    

    # save_json("datasets/2c-generated-balanced.json", balanced_dataset, do_not_overwrite=True)
    # return balanced_dataset
