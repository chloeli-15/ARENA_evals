# %%

import os
import json
from pathlib import Path
import sys
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

# Make sure exercises are in the path; 
chapter = r"chapter3_llm_evals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part3_run_evals_with_inspect").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)

from utils import import_json, save_json, retry_with_exponential_backoff, pretty_print_questions, load_jsonl, omit
import part3_run_evals_with_inspect.tests as tests
# %%
with open('data/power_seeking_dataset.json', 'r') as f:
    record_list = json.loads(f.read())
#%%
print(len(record_list))
print(record_list[0]['answers'].values())
#%%
def record_to_sample(record: dict) -> Sample:
    """
    Converts a item ("record") from the dataset into a Sample object, mapping the fields of the record to the fields of the Sample object.

    Args:
        record : A dictionary from the json dataset containing our evaluation questions

    Returns:
        Sample : A Sample object containing the information in the record
    """
    return Sample(
        input=[record['question']],
        target= record['answer_matching_behaviour'],
        choices= [record['answers'].values()],
        metadata={"behavior_category": record["behavior_category"],
                  "system_prompt": True},
    )
# %%
from inspect_ai.dataset import example_dataset
@task
def theory_of_mind() -> Task:
    return Task(
        dataset=example_dataset("theory_of_mind"),
        plan=[
            chain_of_thought(),
            generate(),
            self_critique(model="openai/gpt-4o-mini"),
        ],
        scorer=model_graded_fact(model="openai/gpt-4o-mini"),
    )
# %%
log = eval(
    theory_of_mind(),
    model="openai/gpt-4o-mini",
    limit=10,
    log_dir="./exercises/logs", 
)
# %%
# with open("./exercises/logs/2024-09-18T10-17-09+00-00_theory-of-mind_DDuwFPnuuLNbbAAP8m2aWF.json", 'r') as f:
#     output = json.loads(f.read())
#     output = output['samples']
#     print(len(output))
# %%
def solver_name(args) -> Solver:
    async def solve(state : TaskState, generate : Generate) -> TaskState:
        #Modify Taskstate in some way
        return state
    return solve

@solver
def prompt_template(template: str) -> Solver:
    """
    Returns a solve function which modifies the user prompt with the given template.

    Args:
        template : The template string to use to modify the user prompt. Must include {prompt} to be replaced with the original user

    Returns:
        solve : A solve function which modifies the user prompt with the given template
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.user_prompt:
            state.user_prompt.text = template.format(prompt = state.user_prompt.text)  
        return state

    return solve
# %%
# @solver
# def prompt_template(template: str) -> Solver:
#     """
#     Returns a solve function which modifies the user prompt with the given template.

#     Args:
#         template : The template string to use to modify the user prompt. Must include {question} to be replaced with the original user

#     Returns:
#         solve : A solve function which modifies the user prompt with the given template
#     """
#     #assert r"""{question}""" in template, r"""ERROR: Template must include {prompt}"""

#     async def solve(state: TaskState, generate: Generate) -> TaskState:
#         if state.user_prompt:
#             state.user_prompt.text = template.format(prompt=state.user_prompt.text)
#         return state

#     return solve

# tests.test_solver_functions(solver_functions=prompt_template(template="{prompt}"), test_dataset=None)
# %%

@solver
def multiple_choice_format(template: str, **params: dict) -> Solver:
    """
    Returns a solve function which modifies the initial prompt to be in the format of a multiple choice question. Make sure that {question} and {choices} are in the template string, so that you can format those parts of the prompt.

    Args:
        template : The template string to use to modify the user prompt. Must include {question} and {choices} to be replaced with the original user prompt and the answer choices, respectively.

    Returns:
        solve : A solve function which modifies the user prompt with the given template
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.user_prompt and state.choices:
            choices = answer_options(state.choices)
            state.user_prompt.text = template.format(question = state.user_prompt.text, choices = choices, **params)
        return state

    return solve
# %%

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
        if state.messages:
            state.messages.append(ChatMessageUser(content=prompt))
        return state

    return solve
# %%

@solver
def system_message(system_message: str, **params: dict) -> Solver:
    """
    Returns a solve function which inserts a system message with the given content into the state.messages list. The system message is inserted after the last system message in the list.

    Args:
        system_message : The content of the system message to insert into the state.messages list

    Returns:
        solve : A solve function which inserts a system message with the given content into the state.messages list
    """
    def insert_system_message(messages: list[ChatMessageUser | ChatMessageSystem | ChatMessageAssistant], message: ChatMessageSystem):
        """
        Inserts the given system message into the messages list after the last system message in the list.

        Args:
            messages : The list of messages to insert the system message into
            message : The system message to insert into the list

        Returns:
            None
        """
        indices = list(filter(lambda i: isinstance(messages[i], ChatMessageSystem), range(len(messages))))
        if len(indices)>0:
            messages.insert(indices[-1], message)
        else:
            messages.insert(0, message)
        

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.messages:
            insert_system_message(state.messages, system_message)
        return state

    return solve
# %%
@solver
def self_critique(
    model: str,
    critique_template: str | None,
    critique_completion_template: str | None,
    **params: dict,
) -> Solver:
    """
    Args:
        - model: The model we use to generate the self-critique

        - critique_template: This is the template for how you present the output of the model so far to itself, so that the model can *generate* the critique. This template should contain {question} and {completion} to be formatted.

        - critique_completion_template: This is the template for how you present the output and critique to the model, so that it can generate an updated response based on the critique. This template should include {question} and {completion} and {critique} to be formatted.

    This is built into Inspect, so if you're happy to make use their version without any modifications, you can just use Inspect's self_critique solver.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return state

    return solve