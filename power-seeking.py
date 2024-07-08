#%%
import json
from inspect_ai import Task, eval, task
from inspect_ai.dataset import example_dataset, FieldSpec, json_dataset, Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import (
  chain_of_thought, generate, self_critique
)
from itertools import product
#When you set your OpenAI API key, should be named OPENAI_API_KEY, since the inspect-ai library looks for a key with that name by default as far as I can tell. Probably can change what the model looks for somewhere with some effort, but better just to flag that OPENAI_API_KEY is best naming convention.
def record_to_sample(record):
    return Sample(
        input=record["question"],
        target=record["answers"][record["answer_matching_behavior"][0]],
        choices = list(record["answers"].values())
    )

power_seeking_dataset = json_dataset("./datasets/power-seeking-2c-gen.json", record_to_sample)
@task
def power_seeking():
    return Task(
        dataset=power_seeking_dataset,
        plan=[
          chain_of_thought(), 
          generate(), 
          self_critique(model = "openai/gpt-3.5-turbo")
        ],
        scorer=model_graded_fact(model = "openai/gpt-3.5-turbo"),
    )
#%%