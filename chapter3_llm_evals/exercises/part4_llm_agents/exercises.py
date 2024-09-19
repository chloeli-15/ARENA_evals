# %%
#%%
import json
import os

import wikipedia
from wikipedia import WikipediaPage
from wikipedia import DisambiguationError, PageError
from openai import OpenAI
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from anthropic import Anthropic
from typing import Literal, Optional, Dict, List, Any
from abc import abstractmethod
import math
import re
import sys
from pathlib import Path
# Make sure exercises are in the path; 
chapter = r"chapter3_llm_evals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part3_run_evals_with_inspect").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)
from utils import countrylist
from utils import evaluate_expression, apply_user_format, apply_assistant_format, establish_client_anthropic, establish_client_OpenAI, retry_with_exponential_backoff

import part4_llm_agents.tests as tests
MAIN = __name__ == '__main__'
# Test the function

# %%
class ArithmeticTask:
    def __init__(self, num1: int | float, num2: int | float):
        self.num1 = num1
        self.num2 = num2
        self.operations: List[str] = ["+", "-", "*", "/", "%", "//"]
        self.correct_answers: Dict[str, float] = self._generate_answers()
        self.is_solved: Dict[str, bool] = {expr: False for expr in self.correct_answers}
        self.current_task_number = 0

    def _generate_answers(self) -> Dict[str, float]:
        """
        Generates a dictionary the correct answers for all possible tasks

        Returns:
            Dict[str, float]: A dictionary with the expression as key and the correct answer as value
        """
        problems = {f"{self.num1} {op} {self.num2}":eval(f"{self.num1} {op} {self.num2}") for op in self.operations}
        return problems

    @property
    def get_current_task(self) -> str:
        """
        Gets the current task for the agent

        Returns:
            str: A string containing the current task
        """
        return list(self.correct_answers.keys())[self.current_task_number]

    @property
    def current_task_instruction(self) -> str:
        """
        Gets a string containing instructions for the current task for the agent. This will be fed to the agent as a user prompt.

        Returns:
            str: A string containing the instructions for the current task
        """
        return f"Compute the following arithmetic expression {self.get_current_task} and output the answer."

    def check_solved(self) -> bool:
        """
        Checks if all tasks have been solved

        Returns:
            bool: True if all tasks have been solved, False otherwise
        """
        return all(self.is_solved.values())

    def check_answer(self, model_answer: str) -> bool:
        """
        Checks if the model's answer is correct

        Args:
            model_answer (str): The model's answer

        Returns:
            bool: True if the model's answer is correct, False otherwise
        """
        try:
            number = float(model_answer)
            return math.isclose(number, self.correct_answers[self.get_current_task])
        except:
            return False

    def update_current_task(self):
        """
        Sets is_solved for the current task to True and increments self.current_task_number by one
        """
        self.is_solved[self.get_current_task] = True
        self.current_task_number += 1

tests.ArithmeticTaskTests(ArithmeticTask)

x = ArithmeticTask(10, 15)
for problem, answer in x.correct_answers.items():
    print(f"{problem} = {answer}")