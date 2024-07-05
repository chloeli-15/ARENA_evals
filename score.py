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

#%%
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

client = OpenAI()

model_name = "gpt-4-turbo"

def reload_rubric():
    global rubric
    importlib.reload(rubric)
    return rubric.RUBRIC
#%%
# Generate prompt
dataset = import_json("datasets/power-seeking-scenes.json")
prompt = f"Evaluate the following questions according to the rubric:"
for d in dataset:
    prompt += f"\nSTART {json.dumps(d)} END"
print("Prompts:", prompt)
RUBRIC = reload_rubric()
print(RUBRIC["power-seeking"])
#%%
# Evaluate model response
start_time = time.time()
completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": RUBRIC["power-seeking"]},
        {"role": "user", "content": prompt}
    ],
    temperature=1.0,
)
end_time = time.time() 
elapsed_time = end_time - start_time
score = completion.choices[0].message.content
print(score)
print("Elapsed time:", elapsed_time)
# %%
