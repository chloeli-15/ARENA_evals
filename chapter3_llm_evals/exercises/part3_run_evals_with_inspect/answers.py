# %%

import os
import json
from typing import Literal
from inspect_ai import Task, eval, task
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessage
from inspect_ai.dataset import FieldSpec, json_dataset, Sample, example_dataset
from inspect_ai.solver._multiple_choice import (
    Solver,
    solver,
    Choices,
    TaskState,
    answer_options,
)
from inspect_ai.solver._critique import (
    DEFAULT_CRITIQUE_TEMPLATE,
    DEFAULT_CRITIQUE_COMPLETION_TEMPLATE,
)
from inspect_ai.scorer import model_graded_fact, match, answer, scorer
from inspect_ai.scorer._metrics import (accuracy, std)
from inspect_ai.scorer._answer import AnswerPattern
from inspect_ai.solver import (
    chain_of_thought,
    generate,
    self_critique,
    system_message,
    Generate,
)
from inspect_ai.model import (
    ChatMessageUser,
    ChatMessageSystem,
    ChatMessageAssistant,
    get_model,
)
import random
import jaxtyping
from itertools import product

import sys
from pathlib import Path



# Make sure exercises are in the path; 
chapter = r"chapter3_llm_evals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part3_run_evals_with_inspect").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)


from utils import import_json, save_json, retry_with_exponential_backoff, pretty_print_questions, load_jsonl, omit
import part3_run_evals_with_inspect.tests as tests

# %%

def record_to_sample(record: dict) -> Sample:
    """
    Converts a item ("record") from the dataset into a Sample object, mapping the fields of the record to the fields of the Sample object.

    Args:
        record : A dictionary from the json dataset containing our evaluation questions

    Returns:
        Sample : A Sample object containing the information in the record
    """
    return Sample(
        input=[
            ChatMessageSystem(content=record["system"]),
            ChatMessageUser(content=record["question"]),
        ],
        target=record["answer_matching_behavior"],
        choices=list(record["answers"].values()),
        metadata={
            "behavior_category": record["behavior_category"],
            "system_prompt": True,
        },
    )
# %%

# from inspect_ai.dataset import example_dataset

# @task
# def theory_of_mind() -> Task:
#     return Task(
#         dataset=example_dataset("theory_of_mind"),
#         plan=[
#             chain_of_thought(),
#             generate(),
#             self_critique(model="openai/gpt-4o-mini"),
#         ],
#         scorer=model_graded_fact(model="openai/gpt-4o-mini"),
#     )
    
# log = eval(
#     theory_of_mind(),
#     model="openai/gpt-4o-mini",
#     limit=10,
#     log_dir="./exercises/logs", 
# )
# %%

@solver
def prompt_template(template: str) -> Solver:
    """
    Returns a solve function which modifies the user prompt with the given template.

    Args:
        template : The template string to use to modify the user prompt. Must include {question} to be replaced with the original user

    Returns:
        solve : A solve function which modifies the user prompt with the given template
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.user_prompt:
            state.user_prompt.text = template.format(prompt=state.user_prompt.text)
        return state

    return solve

tests.test_solver_functions(solver_functions=prompt_template(template=""))
# %%
