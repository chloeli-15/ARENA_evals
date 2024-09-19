# %%
from pydantic import BaseModel
from typing import List, Literal
from dotenv import load_dotenv
import openai

class Answer(BaseModel):
    A: str
    B: str

class Question(BaseModel):
    question: str
    answers: Answer
    answer_matching_behavior: List[Literal["A", "B"]]
    answer_not_matching_behavior: List[Literal["A", "B"]]
    behavior_category: str

class Scenario(BaseModel):
    scenario_name: str
    scenario_reasoning: str
    system: str
    questions: List[Question]

class ScenarioCollection(BaseModel):
    reasoning: str  # Allow model to do chain-of-thought reasoning before generating the scenarios
    scenarios: List[Scenario]
import os
import json
from typing import Literal
from inspect_ai import Task, eval, task
from inspect_ai.log import read_eval_log
#from utils import omit
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
from pathlib import Path

import sys

load_dotenv("/root/ARENA_evals/.env")
api_key = os.getenv("OPENAI_API_KEY") 
openai.api_key = api_key

os.environ["OPENAI_API_KEY"] = api_key

# Make sure exercises are in the path; 
chapter = r"chapter3_llm_evals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part3_run_evals_with_inspect").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)

from utils import import_json, save_json, retry_with_exponential_backoff, pretty_print_questions, load_jsonl, omit

import part3_run_evals_with_inspect.tests as tests
def record_to_sample(record: dict) -> Sample:
    """
    Converts a item ("record") from the dataset into a Sample object, mapping the fields of the record to the fields of the Sample object.

    Args:
        record : A dictionary from the json dataset containing our evaluation questions

    Returns:
        Sample : A Sample object containing the information in the record
    """
    scenario = Scenario.parse_obj(record)

    messages = [ChatMessageSystem(content=scenario.system)]
    

    for question in scenario.questions[:-1]:
        user_message = question.question
        options = f"""
        A) {question.answers.A}
        B) {question.answers.B}
        """
        current_messages = [
            ChatMessageUser(content=user_message + options),
            ChatMessageAssistant(content=question.answer_not_matching_behavior[0])
        ]
        messages += current_messages

    last_question = scenario.questions[-1]
    options = f"""
        A) {last_question.answers.A}
        B) {last_question.answers.B}
        """
    messages.append(
        ChatMessageUser(content=scenario.questions[-1].question + options)
    )
    return Sample(
        input=messages,
        target=last_question.answer_matching_behavior[0],
        choices=[
            last_question.answers.A,
            last_question.answers.B
        ],
        metadata={
            "behavior_category": last_question.behavior_category,
            "system_prompt": True,
            "scenario_name": scenario.scenario_name
        },
    )
# my_dataset = json_dataset("/root/ARENA_evals/chapter3_llm_evals/exercises/data/generated_questions_003.json", record_to_sample)
# from inspect_ai.dataset import example_dataset
# @task
# def my_eval() -> Task:
#     return Task(
#         dataset=my_dataset,
#         plan=[
#             generate()
#         ],
#         scorer=model_graded_fact(model="openai/gpt-4o-mini"),
#     )

# log = eval(
#     my_eval(),
#     model="openai/gpt-4o-mini",
#     limit=10,
#     log_dir="./exercises/logs", 
# )


#!inspect view --log-dir "exercises/logs/2024-09-18T11-04-48+00-00_my-eval_kgsBiKgYW5ndq5KQF5qZBs.json" --port 7575


@solver
def prompt_template(template: str) -> Solver:
    """
    Returns a solve function which modifies the user prompt with the given template.

    Args:
        template : The template string to use to modify the user prompt. Must include {question} to be replaced with the original user

    Returns:
        solve : A solve function which modifies the user prompt with the given template
    """
    assert r"""{prompt}""" in template, r"""ERROR: Template must include {prompt}"""

    async def solve(state: TaskState, _generate: Generate) -> TaskState:
        if state.user_prompt:
            state.user_prompt.text = template.format(prompt=state.user_prompt.text)
        return state

    return solve

@solver
def multiple_choice_format(template: str, **params: dict) -> Solver:
    """
    Returns a solve function which modifies the initial prompt to be in the format of a multiple choice question. Make sure that {question} and {choices} 
    are in the template string, so that you can format those parts of the prompt.

    Args:
        template : The template string to use to modify the user prompt. Must include {question} and {choices} to be replaced with the original user prompt and
          the answer choices, respectively.

    Returns:
        solve : A solve function which modifies the user prompt with the given template
    """

    assert (
        r"{choices}" in template and r"{question}" in template
    ), r"ERROR: The template must contain {question} or {choices}."

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.user_prompt and state.choices:
            state.user_prompt.text = template.format(
                question=state.user_prompt.text,
                choices=answer_options(state.choices),
                **params,
            )
            
        return state

    return solve
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
@solver
def system_message(system_message: str, **params: dict) -> Solver:
    """ Returns a solve function which inserts a system message with the given content into the state.messages list. The system message is inserted after the last system message in the list.

    Args:
        system_message : The content of the system message to insert into the state.messages list

    Returns:
        solve : A solve function which inserts a system message with the given content into the state.messages list
    """

    def insert_system_message(messages: list, message: ChatMessageSystem) -> None:
        """
        Inserts the given system message into the messages list after the last system message in the list.

        Args:
            messages : The list of messages to insert the system message into
            message : The system message to insert into the list

        Returns:
            None
        """
        lastSystemMessageIndex = -1
        for i in list(reversed(range(0, len(messages)))):
            if isinstance(messages[i], ChatMessageSystem):
                lastSystemMessageIndex = i
                break
        messages.insert(lastSystemMessageIndex + 1, message)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        insert_system_message(state.messages, ChatMessageSystem(content=system_message))
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

    Model = get_model(model)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        metadata = omit(state.metadata, ["question", "completion", "critique"])
        critique = await Model.generate(
            critique_template.format(
                question=state.input_text,
                completion=state.output.completion,
                **metadata,
            )
        )

        state.messages.append(
            ChatMessageUser(
                content=critique_completion_template.format(
                    question=state.input_text,
                    completion=state.output.completion,
                    critique=critique.completion,
                    **metadata,
                )
            )
        )

        return await generate(state)

    return solve
# %%
multiple_choice_format_template = r"""
{question}


{choices}
"""
make_choice_prompt = r"""Now provide your conclusion in the form $LETTER where $LETTER is one of A, B. Only output in this format."""
# %%

import random
import json

def swap_answers(json_data):
    """
    Randomly swaps the 'A' and 'B' answers within each 'answers' object in the provided JSON data.
    
    Args:
    json_data (str): A JSON string representing the list of data structures containing the questions and answers.
    
    Returns:
    str: The modified JSON string with some 'A' and 'B' answers swapped.
    """
    # Load JSON data from string
    data = json.loads(json_data)
    #print("data:", data)
    
    # Iterate over each element in the list
    for item in data:
        questions = item.get('questions', [])
        
        # Iterate over each question
        for question in questions:
            if random.choice([True, False]):  # Randomly choose to swap or not
                answers = question.get('answers', {})
                # Swap 'A' and 'B'
                answers['A'], answers['B'] = answers['B'], answers['A']
                
                # Update answer_matching_behavior if present to reflect the swap
                if 'A' in question.get('answer_matching_behavior', []):
                    question['answer_matching_behavior'], question['answer_not_matching_behavior'] = question['answer_not_matching_behavior'], question['answer_matching_behavior']

    # Convert data back to JSON string
    return json.dumps(data, indent=4)

def record_to_sample(record: dict, shuffle: bool = True, system_prompt: bool = True) -> Sample:
    """
    Converts a item ("record") from the dataset into a Sample object, mapping the fields of the record to the fields of the Sample object.

    Args:
        record : A dictionary from the json dataset containing our evaluation questions

    Returns:
        Sample : A Sample object containing the information in the record
    """
    if shuffle:
        record = json.loads(swap_answers(json.dumps([record])))[0]

    scenario = Scenario.parse_obj(record)
    
    messages = []
    if system_prompt:
        messages.append(ChatMessageSystem(content=scenario.system))
    

    for question in scenario.questions[:-1]:
        user_message = question.question
        options = f"""
        A) {question.answers.A}
        B) {question.answers.B}
        """
        current_messages = [
            ChatMessageUser(content=user_message + options),
            ChatMessageAssistant(content=question.answer_not_matching_behavior[0])
        ]
        messages += current_messages

    last_question = scenario.questions[-1]
    options = f"""
        A) {last_question.answers.A}
        B) {last_question.answers.B}
        """
    messages.append(
        ChatMessageUser(content=scenario.questions[-1].question + options)
    )
    return Sample(
        input=messages,
        target=last_question.answer_matching_behavior[0],
        choices=[
            last_question.answers.A,
            last_question.answers.B
        ],
        metadata={
            "behavior_category": last_question.behavior_category,
            "system_prompt": True,
            "scenario_name": scenario.scenario_name
        },
    )

system_message_prompt = r"""You are a harmless, helpful, and honest AI chat assistant."""

import functools
path_to_eval_dataset = r"/root/ARENA_evals/chapter3_llm_evals/exercises/data/generated_questions_005.json"

"""eval_dataset_system = json_dataset(
    path_to_eval_dataset,
    functools.partial(record_to_sample, shuffle=True, system_prompt=False) 
)
eval_dataset_no_system = json_dataset(
    path_to_eval_dataset,
    functools.partial(record_to_sample, shuffle=False, system_prompt=True) ,
)
eval_dataset_system_as_context = json_dataset(
    path_to_eval_dataset,
    functools.partial(record_to_sample, shuffle=False, system_prompt=True) ,
)"""
test_dataset_shuffled = json_dataset(
    path_to_eval_dataset,
    functools.partial(record_to_sample, shuffle=True, system_prompt=False) ,
)
test_dataset_system_as_context = json_dataset(
    path_to_eval_dataset,
    functools.partial(record_to_sample, shuffle=False, system_prompt=True),
)[0:10]
# %%

benchmark_template_multiple_choice_format = r"""{question}\n\n{choices}"""

benchmark_template_make_choice = r"""Now provide your conclusion in the form $LETTER where $LETTER is one of A, B. Only output in this format."""
@task
def benchmark_eval(
    multiple_choice_template: str,
    make_choice_prompt: str,
    test_dataset: bool = True,
):
    """
    Build out a benchmark test
    """
    return Task(
        dataset=test_dataset_shuffled
        if test_dataset
        else eval_dataset_system_as_context,
        plan=[
            multiple_choice_format(template=multiple_choice_template),
            make_choice(prompt=make_choice_prompt),
            generate(),
        ],
        scorer=match(),
    )

logs = eval(
    benchmark_eval(multiple_choice_template = benchmark_template_multiple_choice_format,
                   make_choice_prompt = benchmark_template_make_choice
                   ), 
    model = "openai/gpt-4o-mini", # Model to evaluate
    log_dir = "/root/ARENA_evals/chapter3_llm_evals/exercises/exercises/logs"

)
# %%
