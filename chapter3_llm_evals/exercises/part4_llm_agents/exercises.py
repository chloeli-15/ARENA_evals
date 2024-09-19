# %%

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

# Make sure exercises are in the path
chapter = r"chapter3_llm_evals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part4_llm_agent_evals").resolve()
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)

from utils import (
    import_json,
    save_json,
    retry_with_exponential_backoff,
    pretty_print_questions,
    load_jsonl,
    omit,
)
from utils import countrylist
from utils import (
    evaluate_expression,
    apply_user_format,
    apply_assistant_format,
    establish_client_anthropic,
    establish_client_OpenAI,
    retry_with_exponential_backoff,
)
import part4_llm_agents.tests as tests
import part4_llm_agents.solutions as solutions

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

client = OpenAI()

# %%
import operator


class ArithmeticTask:
    def __init__(self, num1: int | float, num2: int | float):
        self.num1 = num1
        self.num2 = num2
        self.operations: List[str] = ["+", "-", "*", "/", "%", "//"]
        self.correct_answers: Dict[str, float] = self._generate_answers()
        # iterate over keys
        # True means model has attempted and sucessfully solved it, or attempted and refused
        # # false means attempted and failed, or not attempted yet
        self.is_solved: Dict[str, bool] = {expr: False for expr in self.correct_answers}
        self.current_task_number = 0

    def _generate_answers(self) -> Dict[str, float]:
        """
        Generates a dictionary the correct answers for all possible tasks

        Returns:
            Dict[str, float]: A dictionary with the expression as key and the correct answer as value
        """
        ops_dict = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
            "//": operator.floordiv,
            "%": operator.mod,
        }

        ans = {}
        for op in self.operations:
            ans[f"{self.num1} {op} {self.num2}"] = ops_dict[op](self.num1, self.num2)
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
        # """ is docstring, " is string
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
        # call the self.get_current)task as a property, but kinda behaves like a function that is dynamically updated
        return math.isclose(
            float(model_answer),
            self.correct_answers[self.get_current_task],
            rel_tol=1e-5,
            abs_tol=1e-8,
        )

    def update_current_task(self):
        """
        Sets is_solved for the current task to True and increments self.current_task_number by one

        solved means attempted, not correct!!
        """
        self.is_solved[self.get_current_task] = True
        # self.current_task_number+=1
        self.current_task_number = (self.current_task_number + 1) % len(self.operations)


tests.ArithmeticTaskTests(ArithmeticTask)

x = ArithmeticTask(10, 15)
for problem, answer in x.correct_answers.items():
    print(f"{problem} = {answer}")

# %%
""" @staticmethod in python doesnt modify neither instance nor class variables
@classmethod in python modifies class variables but not instance variables"""


class CalculateTool:
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
            str: The description of the tool
        """

        return {
            "type": "function",
            "function": {
                "name": CalculateTool.name,
                "description": """Evaluates arithmetic expressions. Supports operations: +, -, *, /, **, //, %.
                Handles parentheses and follows order of operations. Works with integers and floats.
                Division by zero raises an error. Does not support advanced math functions.
                Results are strings to preserve precision. Large numbers may use scientific notation.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Arithmetic expression to evaluate.",
                        }
                    },
                    "required": ["expression"],
                    "additionalProperties": False,
                },
            },
        }


tests.run_calculate_tool_tests(CalculateTool)

Calculator = CalculateTool()

# %%
messages = [{"role": "user", "content": "Calculate 2+3"}]
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[Calculator.description],
    tool_choice="auto",
)
""" why is message.content None?
When LLMs use tools, they often don't generate any text output.
This can be a problem later when you try to get the model to do chain-of-thought reasoning.
To get around this, it can be better to make two calls to the model for more complex tool use:
one call to get the model to reason about the actions it should take, 
and then another to get the model to use a tool to take those actions."""
print(response.choices[0].message.content)
print(response.choices[0].message.tool_calls)

# %% Return tool call results


# model gives results as a ChatCompletionMessage! need to use below function to access impt info
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
        "content": content,  # e.g. "5"
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name,
    }


messages = [{"role": "user", "content": "Calculate 4/3. Be precise."}]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[Calculator.description],
    tool_choice="auto",
)

# extend basically concats the 2 lists together
# the first elem is the ChatCompletionMessageToolCall which provides context of the previous call
# the second elem is the args for the next tool call, as a dict
messages.extend(
    [  # ChatCompletionsMessage object, signalling the model to use the tool, history of what its done before
        response.choices[0].message,
        apply_tool_call_format(
            # first tool_call object of all the tool calls in this ChatCompletionsMessage
            tool_call=response.choices[0].message.tool_calls[0],
            content=Calculator.execute(
                expression=json.loads(
                    response.choices[0].message.tool_calls[0].function.arguments
                )["expression"]
            ),
        ),
    ]
)

print(response.choices[0].message.content)

response_to_tool_calls = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[Calculator.description],
    tool_choice="auto",
)
print(response_to_tool_calls.choices[0].message.content)

# %% Implement Simple Agent
# each agent has one task and multiple tools available


class SimpleAgent:
    def __init__(
        self,
        task: Any = None,
        model: Literal["gpt-4o-mini"] = "gpt-4o-mini",
        tools: Optional[List[Any]] = None,
        chat_history: Optional[List[dict]] = None,
    ):
        self.model = model
        # this is a single task object
        self.task = task
        # self.tools is a list of Tool instances, each instance of CalculateTool etc
        self.tools = tools
        self.client = OpenAI()
        # list of dictionaries, each dict is a request to chat completion, {role:... , content: ...}
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
        response = client.chat.completions.create(
            model=self.model,
            messages=self.chat_history,
            # this is actually a list of tools description,
            # so that the model understands the tools at disposal
            tools=[tool.description for tool in self.tools] if use_tool else None,
            tool_choice="auto" if use_tool else None,
        )

        return response.choices[0].message

    def execute_tool_calls(self, message: ChatCompletionMessage) -> List[str]:
        """
        Execute the tool calls in the message and return a list of tool_responses.

        Tools within a single time step must be independent
        Args:
            message (ChatCompletionMessage): The message containing the tool calls

        Returns:
            List[str]: A list of tool responses (as strings, we'll format them correctly in run())
        """

        tool_responses = []
        # the model returns a single ChatMessageCompletion, with multiple tools it wants to use
        # hence this ChatMessageCompletion has multiple ChatMessageCompletionTools
        # each tool_call is a ChatMessageCompetionToolCall
        for tool_call in message.tool_calls:
            # each tool_call has a task associated with it
            if not self.task:
                raise ValueError("Task is not set. Cannot execute tool calls.")

            # first find the relevant Tool object in self.tools, by using Tool.name
            # one tool_call has one associated function
            name = tool_call.function.name
            # return first Tool object that matches this name
            tool = next(
                (tool for tool in self.tools if tool.name == name),
            )
            # convert json string to python dict
            # example "expression",
            arguments = json.loads(tool_call.function.arguments)
            # beware of positional args here
            result = tool.execute(**arguments, task=self.task)

            tool_responses.append(result)
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
        print("Running SimpleAgent...")
        instruction = self.task.current_task_instruction
        self.chat_history.append(apply_user_format(instruction))
        response = self.get_response(use_tool=with_tool)
        return response


tests.test_execute_tool_calls(SimpleAgent, CalculateTool, ArithmeticTask)


# %%
my_simple_agent = SimpleAgent(task=ArithmeticTask(10, 15), tools=[Calculator])
my_simple_agent.run()


# %%
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
        if self.verbose:
            print("\nTool calls:", response.tool_calls)

        # Append response to chat history, appends the the whole ChatCompletionMessage
        self.chat_history.append(response)

        # Execute the tool calls and append tool responses to chat history
        tool_calls = response.tool_calls
        try:
            tool_responses = self.execute_tool_calls(response)
            for tool_call, tool_response in zip(tool_calls, tool_responses):
                self.chat_history.append(
                    apply_tool_call_format(tool_call, tool_response)
                )
                if self.verbose:
                    print(
                        f"\nTool call: {tool_call.function.name}, ARGS: {tool_call.function.arguments}"
                    )
                    print(f"Tool response: {tool_response}")
        except Exception as e:
            print(f"\nError handling tool calls: {e}")

    def handle_refusal(self, response: ChatCompletionMessage):
        """
        Handle the refusal from the model response. This function should only be called if the model refuses to answer and should:
        - Append the refusal to the chat history
        - Update the task state

        Args:
            response (ChatCompletionMessage): The response from the model
        """
        if self.verbose:
            print("\nModel Refusal:", response.refusal)
        self.chat_history.append(apply_assistant_format(response.refusal))
        self.task.update_current_task()

    def generate_and_check_final_answer(self) -> Literal["Correct", "Incorrect"]:
        """
        This function should:
        - Get the model to generate a final answer to the question (after it has seen the tool response)
        - Then check this final answer against the correct answer.
        - If the answer is correct, update the task state.
        - if the answer is incorrect, we want to try again, hence we dont update it
        - Then append to chat history (and return) "Correct" if the answer is correct and "Incorrect" if the answer is incorrect.

        Args:
            None

        Returns:
            str: "Correct" or "Incorrect"
        """

        try:
            final_response = self.get_response(use_tool=False)
            self.chat_history.append(final_response)
            # this returns a float
            final_answer = self.parse_answer(final_response)

            if self.task.check_answer(str(final_answer)):
                self.task.update_current_task()
                self.chat_history.append(apply_user_format("Correct"))
                return "Correct"
            else:
                # retry the task
                self.chat_history.append(apply_user_format("Incorrect"))
                return "Incorrect"

        except Exception as e:
            if self.verbose:
                print(f"\nError parsing model answer: {e}")
            raise

    def run(self, with_tool: bool):
        """
        Run one loop of the agent, which involves:
        - getting a task instru
        - getting a response from the model
        - handling the model response, including tool calls, refusals, no tool calls, parsing and checking final answers, errors.
        - managing memory: storing the history of messages to self.chat_history
        - managing task state: staying on the same task or moving to the next task at the end of the loop
        """
        # get task instruction
        task_instruction = self.task.current_task_instruction
        self.chat_history.append(apply_user_format(task_instruction))

        # get initial tool response from model
        response = self.get_response(use_tool=with_tool)

        # handle this response
        # if it has tool calls
        if response.tool_calls:
            self.handle_tool_calls(response)
            # get the final answer
            self.generate_and_check_final_answer()

        # no tool calls, try again in the next time step
        elif response.refusal:
            self.handle_refusal(response)

        # no tool calls and it did not refuse, its final
        else:
            self.generate_and_check_final_answer()

    def parse_answer(self, message: ChatCompletionMessage) -> float:
        """
        Extract the numerical answer from the string output of the model

        Args:
            message (ChatCompletionMessage): The response from the model

        Returns:
            float: The numerical answer extracted from the model
        """
        response = message.content
        if response.find("<answer>") != -1:
            startpoint = response.find("<answer>") + 8
            endpoint = response.find("</answer>")
            return float(response[startpoint:endpoint])


# %%
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
    for _ in range(num_loops):
        if not task.check_solved():
            agent.run(with_tool=True)


agent_loop(arithmetic_agent_1, arithmetic_task_1)
# %%
for message in arithmetic_agent_1.chat_history:
    try:
        print(f"""{str(message.role)}:\n {str(message.content)}\n""")
    except:
        print(f""" {message["role"]}:\n {message["content"]}\n""")
# %%
