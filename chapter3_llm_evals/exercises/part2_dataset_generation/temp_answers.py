#%%
import json
import os
from pathlib import Path
import sys
from pprint import pprint

# Make sure exercises are in the path
chapter = r"chapter3_llm_evals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part2_dataset_generation").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)

with open('data/mcq_examples.json', 'r') as f:
    data = json.loads(f.read())
# %%
pprint(data[25])
# %%

dependent_indices = list(range(3,19))
independent_indices = list(range(0,3)) + list(range(19,26))

print(dependent_indices)
print(independent_indices)
# %%

def get_few_shot_var_user_prompt(self):
    """
    Fills user template f-string and add few-shot examples into the string. Returns the filled few-shot user prompt as a string.
    """

    examples = random.sample(self.examples, self.num_shots)

    return 

get_few_shot_var_user_prompt(question_examples[dependent_indices])