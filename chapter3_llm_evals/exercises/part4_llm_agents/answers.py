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
from pathlib import Path
import sys
from dotenv import load_dotenv
import openai
#Make sure exercises are in the path
exercises_dir = Path(f"/root/ARENA_evals/chapter3_llm_evals/exercises").resolve()
section_dir = (exercises_dir / "part4_llm_agent_evals").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)
from utils import import_json, save_json, retry_with_exponential_backoff, pretty_print_questions, load_jsonl, omit
from utils import countrylist
from utils import evaluate_expression, apply_user_format, apply_assistant_format, establish_client_anthropic, establish_client_OpenAI, retry_with_exponential_backoff
import part4_llm_agents.tests as tests
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
client = OpenAI()
#%%
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
        answers = {}
        for operation in self.operations:
            expression = f"{self.num1}{operation}{self.num2}"
            answers[expression] = evaluate_expression(expression)
        return answers

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
        return f"Compute the following expression: {self.get_current_task}"

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
        return model_answer == self.correct_answers[self.get_current_task]

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

#%%
class CalculateTool():
    name = "calculate"

    @staticmethod
    def execute(expression: str, task: Any = None) -> str:
        """
        Evaluates the string expression in Python using `evaluate_expression()` and returns the result as a string

        Args:
            expression (str): The arithmetic expression to evaluate
            task (Any): Not used in this function

        Returns:
            str: The result of the arithmetical expression as a string
        """
        return str(evaluate_expression(expression))

    @property
    def description(self):
        """
        Provides the description of the tool

        Returns:
            dict: The description of the tool
        """
        desc = {
            "type": 'function',
            'function': {
                "name": self.name,
                "description": "This computes basic arithmetic using two numbers and an operator. Call this whenever you need to do any of the following operations: '+', '-', '*', '/', '%', '//'. Example inputs: '1*2', '5//3'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The arithmetic expression to evaluate, consisting of two numbers and an operator.",
                        },
                    },
                    "required": ["expression"],
                    "additionalProperties": False,
                }
            }
        }
        return desc

tests.run_calculate_tool_tests(CalculateTool)

Calculator = CalculateTool()

#%%
messages = [{"role": "user", "content": "Calculate 2+3"}]
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[Calculator.description],
    tool_choice="auto",
)

print(response.choices[0].message.content)
print(response.choices[0].message.tool_calls)

#%%
def apply_tool_call_format(
    tool_call: ChatCompletionMessageToolCall, content: str
) -> dict:
    """
    Formats the response of a tool call to be returned to the model.
    Args:
        - tool_call (ChatCompletionMessageToolCall) : The tool call object
        - content (str) : This is the tool response (i.e. results from executing the tool)

    Returns:
        - dict : The formatted tool response to be returned to the model
    """
    return {
        "role": "tool",
        "content": content, # e.g. "5"
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name
    }

messages = [{"role": "user", "content": "Calculate 5/3. Be precise."}]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[Calculator.description],
    tool_choice="auto",
)

messages.append(response.choices[0].message)

tool_call = response.choices[0].message.tool_calls[0]
arguments = json.loads(tool_call.function.arguments)
answer = Calculator.execute(expression=arguments['expression'])
answer_formatted = apply_tool_call_format(tool_call, content=answer)
messages.append(answer_formatted)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[Calculator.description],
    tool_choice="auto",
)

print(response.choices[0].message.content)
#%%
class SimpleAgent:
    def __init__(
        self,
        task: Any = None,
        model: Literal["gpt-4o-mini"] = "gpt-4o-mini",
        tools: Optional[List[Any]] = None,
        chat_history: Optional[List[dict]] = None,
    ):
        self.model = model
        self.task = task
        self.tools = tools
        self.client = OpenAI()
        self.chat_history = chat_history if chat_history else []

    @retry_with_exponential_backoff
    def get_response(self, use_tool: bool = True) -> ChatCompletionMessage:
        """
        Get the response from the model via an API call, with the option of tool calling.

        Args:
            use_tool (bool): Whether to use tool calling or not

        Returns:
            ChatCompletionMessage: The response from the model
        """

        if use_tool:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.chat_history,
                tools=[tool.description for tool in self.tools],
                tool_choice="auto",
            )

        else:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.chat_history,
            )

        self.chat_history.append(response.choices[0].message)

        # if use_tool and response.choices[0].message.tool_calls is not None:
        #     tool_responses = self.execute_tool_calls(response.choices[0].message)

        return response.choices[0].message

    def execute_tool_calls(self, message: ChatCompletionMessage) -> List[str]:
        """
        Execute the tool calls in the message and return a list of tool_responses.

        Args:
            message (ChatCompletionMessage): The message containing the tool calls

        Returns:
            List[str]: A list of tool responses (as strings, we'll format them correctly in run())
        """
        tool_responses = []
        for tool_call in message.tool_calls:
            tool = [tool for tool in self.tools if tool.name == tool_call.function.name][0]
            arguments = json.loads(tool_call.function.arguments)
            response = tool.execute(**arguments, task=self.task)
            tool_responses.append(str(response))

        return tool_responses

    def run(self, with_tool: bool = True) -> ChatCompletionMessage:
        """
        Default implementation of run method.
        This can be overridden in subclasses for specific behavior.

        Args:
            with_tool (bool): Whether to use tool calling or not

        Returns:
            str: The response from the model
        """
        print(f"Running SimpleAgent...")
        instruction = self.task.current_task_instruction
        self.chat_history.append(apply_user_format(instruction))
        response = self.get_response(use_tool=with_tool)
        return response

tests.test_execute_tool_calls(SimpleAgent, CalculateTool, ArithmeticTask)

#%%
my_simple_agent = SimpleAgent(ArithmeticTask(10, 15), tools=[Calculator])
my_simple_agent.run()

#%%
class ArithmeticAgent(SimpleAgent):
    """
    ArithmeticAgent class for doing simple arithmetic tasks.

    Inherits from SimpleAgent which includes the following attributes and methods:

    Attributes:
        model (str): The model used for generating responses (inherited)
        tool_descriptions (List[dict]): List of tool descriptions (inherited)
        client (OpenAI): OpenAI client for API calls (inherited)
        task (Any): The current task being executed (inherited)
        chat_history (List[dict]): History of interactions (inherited)

    Methods:
        get_response(use_tool: bool = True) -> ChatCompletionMessage:
            Get response from the model (inherited)

        execute_tool_calls(message: ChatCompletionMessage) -> List[str]:
            Execute tool calls from the model's response (inherited)

        run(with_tool: bool = True) -> bool:
            Run one loop of the Wikipedia agent
    """

    def __init__(
        self,
        model: Literal["gpt-4o-mini"] = "gpt-4o-mini",
        task: Any = None,
        tools: Optional[List[Any]] = [Calculator],
        chat_history: List[dict] = None,
        verbose: bool = True,
    ):
        super().__init__(model=model, task=task, tools=tools, chat_history=chat_history)
        self.verbose = verbose

    def handle_tool_calls(self, response: ChatCompletionMessage):
        """
        Handle the tool calls from the model response. This function should:
        - Execute the tool calls
        - Append the tool calls and responses to the chat history

        Args:
            response (ChatCompletionMessage): The response from the model
        """
        tool_responses = self.execute_tool_calls(response)
        for tool_call, tool_response in zip(response.tool_calls, tool_responses):
            self.chat_history.append(apply_tool_call_format(tool_call, content=tool_response))

    def handle_refusal(self, response: ChatCompletionMessage):
        """
        Handle the refusal from the model response. This function should only be called if the model refuses to answer and should:
        - Append the refusal to the chat history
        - Update the task state

        Args:
            response (ChatCompletionMessage): The response from the model
        """
        # Append the refusal to the chat history
        self.chat_history.append(apply_assistant_format(response.refusal))
        # Update the task state
        self.task.update_current_task()

    def generate_and_check_final_answer(self) -> Literal["Correct", "Incorrect"]:
        """
        This function should:
        - Get the model to generate a final answer to the question (after it has seen the tool response)
        - Then check this final answer against the correct answer.
        - If the answer is correct, update the task state.
        - Then append to chat history (and return) "Correct" if the answer is correct and "Incorrect" if the answer is incorrect.

        Args:
            None

        Returns:
            str: "Correct" or "Incorrect"
        """
        # Get the model to generate a final answer to the question (after it has seen the tool response)
        response = self.get_response(use_tool=False)
        # Then check this final answer against the correct answer.
        answer = self.parse_answer(response)
        print(f"{answer=}")
        # If the answer is correct, update the task state.
        is_correct = self.task.check_answer(answer)
        # Then append to chat history (and return) "Correct" if the answer is correct and "Incorrect" if the answer is incorrect.
        if is_correct:
            message = 'Correct'
        else:
            message = 'Incorrect'
        # self.chat_history.append(apply_user_format(message))
        return message

    def run(self, with_tool: bool):
        """
        Run one loop of the agent, which involves:
        - getting a task
        - getting a response from the model
        - handling the model response, including tool calls, refusals, no tool calls, parsing and checking final answers, errors.
        - managing memory: storing the history of messages to self.chat_history
        - managing task state: staying on the same task or moving to the next task at the end of the loop
        """
        # getting a task
        if self.task.check_solved():
            return
        # getting a response from the model
        self.chat_history.append({"role": "user", "content": f"Compute {self.task.current_task_instruction}"})
        response = self.get_response(use_tool=with_tool)
        # handling the model response, including tool calls, refusals, no tool calls, parsing and checking final answers, errors.
        if response.refusal is not None:
            self.handle_refusal(response)
        else:
            if with_tool and response.tool_calls is not None:
                self.handle_tool_calls(response)   
            is_correct = self.generate_and_check_final_answer()
            print(f"{is_correct=}")
            if is_correct:
                self.task.update_current_task()
        # managing memory: storing the history of messages to self.chat_history

        # managing task state: staying on the same task or moving to the next task at the end of the loop


    def parse_answer(self, message: ChatCompletionMessage) -> float | int:
        """
        Extract the numerical answer from the string output of the model

        Args:
            message (ChatCompletionMessage): The response from the model

        Returns:
            float: The numerical answer extracted from the model
            None: If no valid number is found
        """
        # Extract the content from the message
        content = message.content

        # Regular expression pattern to match numbers (including decimals and scientific notation)
        pattern = r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?'
        
        # Find all matches in the text
        matches = re.findall(pattern, content)
        
        if matches:
            # Return the first match as a float
            return float(matches[-1])
        else:
            # Return None if no number is found
            return None

#%%
arithmetic_task_1 = ArithmeticTask(31.1, 8)
arithmetic_agent_1 = ArithmeticAgent(
    task=arithmetic_task_1, verbose=True, tools=[Calculator]
)


def agent_loop(agent, task, num_loops: int = 10):
    """
    Run the agent loop for a given number of loops

    Args:
        agent (ArithmeticAgent): The agent to run
        task (ArithmeticTask): The task to solve
        num_loops (int): The number of loops to run
    """
    for loop in range(num_loops):
        agent.run(with_tool=True)

agent_loop(arithmetic_agent_1, arithmetic_task_1)

#%%
arithmetic_task_1.is_solved