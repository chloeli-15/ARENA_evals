#%%
from generate import query_generator, reload_gen_prompt
from evaluate import query_evaluator, summarize_results, check_answer_balance, check_label_balance
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

# Set file paths
seed_dataset_name = "power-seeking-2c-seed-v3"
date = time.strftime("%Y-%m-%d")
gen_output_filepath = f"./datasets/{seed_dataset_name}-gen-{date}.json"

# Set parameters
num_total_q_to_gen = 16 #Here, define the total number of questions you want to generate. The code will then separate them into batches of 4 (or num_q_per_gen) questions each.
model_name = "gpt-4o"
rubric_name = "quality"
num_choices = "2-choice" # "4-choice" or "2-choice"
system_prompt = reload_gen_prompt(num_choices)

#%%
# Generate questions
query_generator(dataset_name = seed_dataset_name, 
                model_name = model_name,
                num_total_q_to_gen = num_total_q_to_gen,
                system_prompt = system_prompt, 
                output_filepath = gen_output_filepath)

#%%
# Evaluate generated questions
dataset_to_evaluate = "power-seeking-2c-seed-v3-gen-2"
gen_output_filepath="datasets/power-seeking-2c-seed-v3-gen-2.json"
_, graded_filepath = query_evaluator(dataset_to_evaluate = dataset_to_evaluate,
                model_name = model_name, 
                rubric_name = rubric_name, 
                dataset_path= gen_output_filepath)
print(graded_filepath)

# graded_filepath = "scores/quality/power-seeking-2c-seed-v3-gen-2024-07-12.json"
summarize_results(rubric_name=rubric_name,
                  model_name=  model_name,
                  dataset_path = graded_filepath)


# %%
