#%%
from generate import query_generator, reload_gen_prompt
from evaluate import query_evaluator, summarize_results
import os
from utils import import_json, reload_config
import time 
from dotenv import load_dotenv
import openai
from openai import OpenAI
import config

#%%
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key
client = OpenAI()
reload_config()

# Set generation variables
seed_dataset_name = "power-seeking-2c-seed-v2"
dataset_to_evaluate = "power-seeking-2c-seed-v2-gen"

num_total_q_to_gen = 4
model_name = "gpt-4o"
rubric_name = "quality"

num_choices = "4-choice" # "4-choice" or "2-choice"
system_prompt = reload_gen_prompt(num_choices)

date = time.strftime("%Y-%m-%d")
gen_output_filepath = f"./datasets/{seed_dataset_name}-gen-{date}.json"

# Generate questions
query_generator(dataset_name = seed_dataset_name, 
                model_name = model_name,
                num_total_q_to_gen = num_total_q_to_gen,
                system_prompt = system_prompt, 
                output_filepath = gen_output_filepath)

# Evaluate generated questions
query_evaluator(dataset_to_evaluate = dataset_to_evaluate,
                model_name = model_name, 
                rubric_name = rubric_name, 
                dataset_path= gen_output_filepath)
                
summarize_results(rubric_name=rubric_name,
                  model_name=  model_name,
                   dataset_to_evaluate = dataset_to_evaluate)

# %%
