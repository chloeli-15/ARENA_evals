#%%
# %%

import os
import json
from typing import List, Literal
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

from pathlib import Path
import sys

import random
import jaxtyping
from itertools import product

# Make sure exercises are in the path; 
chapter = r"chapter3_llm_evals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part3_run_evals_with_inspect").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)


import part3_run_evals_with_inspect.tests as tests
from utils import import_json, save_json, retry_with_exponential_backoff, pretty_print_questions, load_jsonl, omit

# %%

def record_to_sample(record: dict) -> Sample:
    """
    Converts a item ("record") from the dataset into a Sample object, mapping the fields of the record to the fields of the Sample object.

    Args:
        record : A dictionary from the json dataset containing our evaluation questions

    Returns:
        Sample : A Sample object containing the information in the record
    """
    cm_system: ChatMessageSystem = ChatMessageSystem(content=record["system"])
    cm_user: ChatMessageUser = ChatMessageUser(content=record["question"])
    cm: List[ChatMessage] = [cm_system, cm_user]

    return Sample(
        input=cm,
        target= record['answer_matching_behavior'],
        choices= [answer for answer in record['answers'].values()],
        metadata={key: record[key] for key in ["behavior_category", "score", "model_response"]},
    )

eval_dataset = json_dataset(r"/root/ARENA_evals/chapter3_llm_evals/exercises/data/power_seeking_dataset.json", record_to_sample)


# %%

from inspect_ai.dataset import example_dataset

@task
def power_seeking_ai() -> Task:
    return Task(
        dataset=eval_dataset,
        plan=[
            chain_of_thought(),
            generate(),
            self_critique(model="openai/gpt-4o-mini"),
        ],
        scorer=model_graded_fact(model="openai/gpt-4o-mini"),
    )

#%%
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
            state.user_prompt.text = template.format(state.user_prompt.text)
        return state

    return solve

tests.test_solver_functions(
    solver_functions=prompt_template(template=""),
    test_dataset=list(example_dataset("theory_of_mind")),
)

#%%

@solver
def make_choice(prompt: str) -> Solver:
    """
    Returns a solve function which adds a user message at the end of the state.messages list with the given prompt.

    Args:
        prompt : The prompt to add to the user messages

    Returns:
        solve : A solve function which adds a user message with the given prompt to the end of the state.messages list
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages.append(ChatMessageUser(content=prompt))
        return state

    return solve


# %%

@solver
def system_message(system_message: str, **params: dict) -> Solver:
    def insert_system_message(messages: list[ChatMessage], message: ChatMessageSystem):
        try:
            index = next(i for i in range(len(messages), 0, -1) if messages[i-1].role == "system")
        except StopIteration:
            index = 0
        
        messages.insert(index, message)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        insert_system_message(state.messages, ChatMessageSystem(content=system_message))
        return state

    return solve

#%%

tests.test_solver_functions(
    solver_functions=[
        make_choice(prompt="Please output one of the choices above."),
        system_message("You are an AI, beep boop"),
        make_choice("PLEASE enter one of the choices above"),
        system_message("You are a friendly AI, beeep boooop"),
    ],
    test_dataset=list(example_dataset("theory_of_mind")),
)

# %%
