#%%
import openai
from openai import OpenAI
import os
import torch as t
import json
'''
client = OpenAI()
#API keys need to be a secret. This is annoying but can be set in the terminal and teaches good practice for API key usage going forward (I didn't know about this).
completion = client.chat.completions.create(
    model = "gpt-3.5-turbo",
    messages = [
        {"role": "system", "content": "You are an expert writer of evaluations for AI systems, specifically Large Language Models."},
        {"role" : "user", "content" : "Write a multiple choice evaluation for such a model to test if it is power-seeking"}
    ]
)
print(completion.choices[0].message)
'''
#%%
print(os.getcwd())
#os.chdir("")
with open("./ARENA_evals/datasets/power-seeking.json", "r") as f:
    data = json.load(f)
print(data)

f.close()
# %%
