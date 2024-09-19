#%%

import json
import os
from dotenv import load_dotenv

import openai
import anthropic

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
from pathlib import Path
import sys
from typing import Dict

#Make sure exercises are in the path
chapter=r"chapter3_llm_evals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part4_llm_agent_evals").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)

from utils import import_json, save_json, retry_with_exponential_backoff, pretty_print_questions, load_jsonl, omit
from utils import countrylist
from utils import evaluate_expression, apply_user_format, apply_assistant_format, establish_client_anthropic, establish_client_OpenAI, retry_with_exponential_backoff
import part4_llm_agents.tests as tests
import part4_llm_agents.solutions as solutions

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

client = OpenAI()

#%%
import operator

class ArithmeticTask:
    def __init__(self, num1: int | float, num2: int | float):
        self.num1 = num1
        self.num2 = num2
        self.operations: List[str] = ["+", "-", "*", "/", "%", "//"]
        self.correct_answers: Dict[str, float] = self._generate_answers()
        #iterate over keys
        self.is_solved: Dict[str, bool] = {expr: False for expr in self.correct_answers}
        self.current_task_number = 0
        

    def _generate_answers(self) -> Dict[str, float]:
        """
        Generates a dictionary the correct answers for all possible tasks

        Returns:
            Dict[str, float]: A dictionary with the expression as key and the correct answer as value
        """
        ops_dict = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '//': operator.floordiv,
            '%': operator.mod
        }
        
        ans={}
        for op in self.operations:
            ans[f"{self.num1} {op} {self.num2}"]=ops_dict[op](self.num1,self.num2)
        return ans

    """ this @proerty allows you to compute a value dynamically each time it's accessed. 
    If you used self.perimeter as a regular attribute, 
    you'd have to remember to update it every time side_length changes."""
    @property
    def get_current_task(self) -> str:
        """
        Gets the current task for the agent

        Returns:
            str: A string containing the current task
        """
        return f"{self.num1} {self.operations[self.current_task_number]} {self.num2}"

    @property
    def current_task_instruction(self) -> str:
        """
        Gets a string containing instructions for the current task for the agent. This will be fed to the agent as a user prompt.

        Returns:
            str: A string containing the instructions for the current task
        """
        #""" is docstring, " is string
        return f"""Calculate the result of the following expression: {str(self.num1)} {self.operations[self.current_task_number]} {str(self.num2)}.
            Give your final answer in the format: <answer>NUMBER</answer>, where NUMBER is a numerical value"""

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
        #call the self.get_current)task as a property, but kinda behaves like a function that is dynamically updated
        return math.isclose(float(model_answer), self.correct_answers[self.get_current_task],rel_tol=1e-5, abs_tol=1e-8)

    def update_current_task(self):
        """
        Sets is_solved for the current task to True and increments self.current_task_number by one
        """
        self.is_solved[self.get_current_task]=True
        #self.current_task_number+=1
        self.current_task_number = (self.current_task_number +1) % len(self.operations)
    

tests.ArithmeticTaskTests(ArithmeticTask)

x = ArithmeticTask(10, 15)
for problem, answer in x.correct_answers.items():
    print(f"{problem} = {answer}")
    
#%%


