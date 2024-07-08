#%%
import json
from inspect_ai import Task, eval, task
from inspect_ai.dataset import example_dataset, FieldSpec, json_dataset, Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import (
  chain_of_thought, generate, self_critique
)
from itertools import product
#%%
with open("datasets/power-seeking-2c-gen.json", "r") as f:
    dataset = json.load(f)
    print(list(dataset[6]["answers"].values()))
    for i in dataset:
        print(type(i["answers"]))
#%%
#When you set your OpenAI API key, should be named OPENAI_API_KEY, since the inspect-ai library looks for a key with that name by default as far as I can tell. Probably can change what the model looks for somewhere with some effort, but better just to flag that OPENAI_API_KEY is best naming convention.
def record_to_sample(record):
    return Sample(
        input=record["question"],
        target=record["answer_matching_behavior"],
        choices = list(record["answers"].values())
    )

dataset = json_dataset("./datasets/power-seeking-2c-gen.json", record_to_sample)

#%%


@task
def power_seeking():
    return Task(
        dataset=json_dataset,
        plan=[
          chain_of_thought(), 
          generate(), 
          self_critique(model = "openai/gpt-3.5-turbo")
        ],
        scorer=model_graded_fact(model = "openai/gpt-3.5-turbo"),
    )

# 'grid' will be a permutation of all parameters
#grid = list(product(*(params[name] for name in params)))

# run the evals and capture the logs
logs = eval(
    power_seeking(),
    model="openai/gpt-3.5-turbo",
)

#%%
log = logs[0]
print(log.status) 
# Started, success, or failure
print(log.eval) 
# Task, model, creation time, various IDs used to store the log.
print(log.plan) 
#The plan for the eval, whether you got logits, what sampling you used, a logit bias, a seed, temperature, top_p, whether you capped token generation, if there was a system_message
print(log.samples[0]) 
#Do not use log.samples without indexing. log.samples consists of a huge amount of information for each thing you tested the model on, gets very big on even medium-sized evals.
print(log.results) 
# Prints accuracy, bootstrap standard deviation
print(log.stats)
#Useful stats: How many tokens were fed to the model, how many tokens it generated, and times.
print(log.logging) 
#Logging messages
print(log.error) 
#Returns the error if there was an error, None if no error.
#%%