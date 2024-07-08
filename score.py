#%%
import openai
from openai import OpenAI
import sys
import os
import rubric 
from utils import import_json_as_string, import_json
from dotenv import load_dotenv
import json
import time
import importlib
import re
import random

#%%

def reload_rubric():
    global rubric
    importlib.reload(rubric)
    return rubric.RUBRIC

def format_output(message:str, file_name:str, rubric_name:str="quality"):
    pattern = re.compile(r'START\s*(.*?)\s*END', re.DOTALL)
    matches = pattern.findall(message)
    q_dicts = []
    for match in matches:
        try:
            q_dicts.append(json.loads(match))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    date = time.strftime("%Y-%m-%d")
    file_name = 'scores/' + rubric_name +"/"+ date + "-" + file_name + "-graded.json"
    if os.path.exists(file_name):
        file_name = 'scores/' + rubric_name +"/"+ date + "-" + file_name + "-graded-" + str(random.randint(0, 100))+ ".json" 
    with open(file_name, 'w') as f:
        json.dump(q_dicts, f)
    

#%%
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

client = OpenAI()

#%%
# Generate prompt
model_name = "gpt-4-turbo"
dataset_name = "power-seeking-2c-gen"
rubric_name = "quality"

dataset = import_json(f"datasets/{dataset_name}.json")
prompt = f"Evaluate the following questions according to the rubric:"
for d in dataset:
    prompt += f"\nSTART {json.dumps(d)} END"
print(f"{len(dataset)} Prompts:", prompt)
RUBRIC = reload_rubric()
print(RUBRIC[rubric_name])
#%%
# Evaluate model response
start_time = time.time()
completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": RUBRIC[rubric_name]},
        {"role": "user", "content": prompt}
    ],
    temperature=1.0,
)
end_time = time.time() 
elapsed_time = end_time - start_time
score = completion.choices[0].message.content
print(score)
print("Elapsed time:", elapsed_time)
format_output(score, dataset_name, rubric_name)

# %%
