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
        problems = {f"{self.num1} {op} {self.num2}":evaluate_expression(f"{self.num1} {op} {self.num2}") for op in self.operations}
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

# %%
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

        description = r"""This is a calculator tool.
        
        What it does:
        The calculate tool can perform arithmetic on two pairs of numbers.

        When it should be used:
        This tool should be used when you want to perform basic arithmetic.

        Tool limitations:
        - This tool cannot operate on more or less than two operands
        - You can call the tool multiple times to evaluate more complex expressions
        - This tool will suffer from precision loss due to floating point arithmetic
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Arithmetic expression to evaluate of the format {number} {arithmetic operator} {number}"
                        },
                    },
                    "required": ["expression"],
                    "additionalProperties": False,
                }
            }
        }

tests.run_calculate_tool_tests(CalculateTool)

Calculator = CalculateTool()
# %%
messages = [{"role": "user", "content": "Calculate 77//3"}]
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[Calculator.description],
    tool_choice="auto",
)

print(response.choices[0].message.content)
print(response.choices[0].message.tool_calls)

# %%
# %%
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

# %%
messages.append(response.choices[0].message)

# %%

tool_call = response.choices[0].message.tool_calls[0]
args = tool_call.function.arguments
expression = json.loads(args)["expression"]

result = Calculator.execute(expression)
tool_message = apply_tool_call_format(tool_call, result)
messages.append(tool_message)

print(messages)

# %%
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[Calculator.description],
    tool_choice="auto",
)

# %%
response
# %%
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
                model=self.model,
                messages=self.chat_history,
                tools=[tool.description for tool in self.tools],
                tool_choice="auto",
            )
        else:
            response = client.chat.completions.create(
                model=self.model,
                messages=self.chat_history
            )

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
            args = tool_call.function.arguments

            kwargs = json.loads(args)
            func = [tool for tool in self.tools if tool.name == tool_call.function.name][0]

            print('Tool args', kwargs)
            result = func.execute(**kwargs, task=self.task)
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
        print(f"Running SimpleAgent...")
        instruction = self.task.current_task_instruction
        self.chat_history.append(apply_user_format(instruction))
        response = self.get_response(use_tool=with_tool)
        return response

tests.test_execute_tool_calls(SimpleAgent, CalculateTool, ArithmeticTask)
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

    def handle_tool_calls(self, message: ChatCompletionMessage):
        """
        Handle the tool calls from the model response. This function should:
        - Execute the tool calls
        - Append the tool calls and responses to the chat history

        Args:
            response (ChatCompletionMessage): The response from the model
        """

        self.chat_history.append(message)

        # Execute the tool calls
        results = self.execute_tool_calls(message)

        # Append the tool calls and responses to the chat history
        tool_messages = [apply_tool_call_format(tool_call, result)
                                for tool_call, result in zip(message.tool_calls, results)]

        self.chat_history.extend(tool_messages)

        return self.get_response(use_tool=True)


    def handle_refusal(self, message: ChatCompletionMessage):
        """
        Handle the refusal from the model response. This function should only be called if the model refuses to answer and should:
        - Append the refusal to the chat history
        - Update the task state

        Args:
            response (ChatCompletionMessage): The response from the model
        """
        if message.refusal:
            self.chat_history.append(message.refusal)
            self.task.update_current_task()
            return self.get_response().message
        return message

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
        message = self.get_response()
        correct = self.task.check_answer(self.parse_answer(message))
        if correct:
            self.task.update_current_task()
        self.chat_history.append(message)
        return "Correct" if correct else "Incorrect"

    def run(self, with_tool: bool):
        """
        Run one loop of the agent, which involves:
        - getting a task
        - getting a response from the model
        - handling the model response, including tool calls, refusals, no tool calls, parsing and checking final answers, errors.
        - managing memory: storing the history of messages to self.chat_history
        - managing task state: staying on the same task or moving to the next task at the end of the loop
        """
        instruction = self.task.current_task_instruction
        self.chat_history.append(apply_user_format(instruction))
        message = self.get_response(use_tool=with_tool)

        message = self.handle_tool_calls(message)

        message = self.handle_refusal(message)

        correct_label = self.generate_and_check_final_answer()

        print('Finished running task:', self.task.current_task_instruction, correct_label)


    def parse_answer(self, message: ChatCompletionMessage) -> float:
        """
        Extract the numerical answer from the string output of the model

        Args:
            message (ChatCompletionMessage): The response from the model

        Returns:
            float: The numerical answer extracted from the model
        """
        # Find all integers and floating point numbers in the string
        numbers = re.findall(r'-?\d+(?:\.\d+)?', message.content)
        
        # Return the last one if any were found, otherwise return None
        return float(numbers[-1]) if numbers else None

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
    for n in range(num_loops):
        agent.run(with_tool=True)


# agent_loop(arithmetic_agent_1, arithmetic_task_1)
# %%

#Retrieve a Wikipedia page from its title
page = wikipedia.page("Large language model")

# Access basic page information
print("Title:", page.title)
print("\nURL", page.url)
print(f"\nSummary (word count {len( page.summary.split())}):", page.summary)
print(
    f"\nContent (word count {len( page.content.split())}):",
    page.content[:1000],
    "......",
)
print(
    f"""\nLinks (link count {len(page.links)}): [{", ".join(page.links[:7])}, ......]"""
)

#%%

# Fixes PageError by allowing redirects
page = wikipedia.page("Animalss", redirect=True)
print(page.title)

# Fixes DisambiguationError by selecting the first option

try:
    page = wikipedia.page("Python")
except DisambiguationError as e:
    page = wikipedia.page(e.options[0])
print(page.title)
# %%

def get_page(title: str) -> WikipediaPage:
    """
    Get a Wikipedia page object given a title. If the title is ambiguous, choose the first option. If the title is not found, try to find a similar title.

    Args:
        title (str): The title of the Wikipedia page

    Returns:
        WikipediaPage: The Wikipedia page
    """
    try:
        return wikipedia.page(title, auto_suggest=False, redirect=True)
    except DisambiguationError as e:
        return wikipedia.page(e.options[0], auto_suggest=False, redirect=True)
    except PageError as e:
        return wikipedia.page(title, auto_suggest=True, redirect=True)

# %%

def get_permitted_links(current_page: WikipediaPage) -> list[str]:
    """
    Get "permitted" links (i.e. links that are in the content of the page) from a Wikipedia page.

    Args:
        current_page (WikipediaPage): The current Wikipedia page

    Returns:
        list[str]: A list of permitted links from current_page

    """
    return [link for link in current_page.links if link in current_page.content]

# %%

class WikiGame:
    def __init__(
        self,
        starting_page: str,
        goal_page: str,
    ):
        """
        Initialize the Wikipedia game object.

        Args:
            starting_page (str): The page the agent starts on.
            goal_page (str): The page the agent is trying to reach.
        """

        # Task state variables
        self.page_history: List[str] = [starting_page]
        self.starting_page: WikipediaPage = self.get_page(starting_page)
        self.goal_page: WikipediaPage = self.get_page(goal_page)
        self.current_page: WikipediaPage = self.starting_page

    # ========================= Helper Functions (given) =========================

    # Get page and page summary
    @staticmethod
    def get_page(title: str) -> WikipediaPage:
        """
        Get a Wikipedia page object given a title. If the title is ambiguous, choose the first option. If the title is not found, try to find a similar title.

        Args:
            title (str): The title of the Wikipedia page

        Returns:
            WikipediaPage: The Wikipedia page
        """
        try:
            return wikipedia.page(title, auto_suggest=False, redirect=True)
        except DisambiguationError as e:
            return wikipedia.page(e.options[0], auto_suggest=False, redirect=True)
        except PageError as e:
            return wikipedia.page(title, auto_suggest=True, redirect=True)

    def get_page_summary(self, page: WikipediaPage | None = None) -> str:
        """
        Get summary of a wikipedia page, to the last full stop within the first 500 characters. This is used to give a brief overview of the page to the agent.

        Args:
            page (WikipediaPage): The Wikipedia page object.

        Returns:
            str: The summary of the Wikipedia page.
        """
        page = page if page else self.goal_page
        summary = page.content[:500]
        last_period_index = summary.rfind(".")
        return summary[: last_period_index + 1] if last_period_index != -1 else summary

    # Get and check permitted links
    def get_permitted_links(self, title: Optional[str] = None) -> list[str]:
        """
        Returns a list of permitted links (i.e. links in the main page content) for the current page.

        Args:
            title (Optional[str]): The title of the Wikipedia page. If None, uses the current page.

        Returns:
            list[str]: The permitted links.
        """
        if title:
            page = self.get_page(title)
            all_links = page.links
            content = page.content
            permitted_links = [link for link in all_links if link in content]
            if title in permitted_links:
                permitted_links.remove(title)
        else:
            all_links = self.current_page.links
            content = self.current_page.content
            permitted_links = [link for link in all_links if link in content]
            if self.current_page.title in permitted_links:
                permitted_links.remove(self.current_page.title)
        return permitted_links

    def is_permitted_link(self, link: str) -> bool:
        """
        Returns True if the link is in the permitted links for the current page, False otherwise.

        Args:
            link (str): The link to check.

        Returns:
            bool: True if the link is permitted, False otherwise
        """
        return link.lower() in (x.lower() for x in self.get_permitted_links())

    # ========================= Task-specific instructions (to implement) =========================

    @property
    def system_instruction(self) -> dict:
        """
        Generate the starting instructions for the game, formatted as a system prompt.

        Returns:
            dict: The starting instructions. The "role" is "system" for system messages.
        """
        out_dict = {
            "role": 'system',
            "content": r'''You are playing the WikiGame, where the target is to reach a goal page from a starting page following
            links. The aim is to do it in the fewest number of page navigations.'''
        }
        return out_dict

    @property
    def on_page_instruction(self) -> dict:
        """
        Tell the agent what page they are on and give a summary of the page, formatted as a user prompt.

        Returns:
            dict: The instructions for the current page. The "role" is "user" for user messages.
        """
        out_dict = apply_user_format(f"""You are currently on {self.current_page.title}, and the goal is to reach {self.goal_page.title}.
                                     The content summary of the current page is: {self.get_page_summary(self.current_page)}.""")
        return out_dict

    @property
    def next_step_instruction(self) -> dict:
        """
        Ask the agent "What's the next step?" after making a tool call, formatted as a user prompt.

        Returns:
            dict: The instructions for the next step. The "role" is "user" for user messages.
        """
        out_dict = apply_user_format(f"Which link should we follow to reach our goal the fastest?")
        return out_dict

    # ========================= Task State management (to implement) =========================

    def check_win(self) -> bool:
        """
        Check if the agent has won the game.

        Returns:
            bool: True if the agent has won, False otherwise.
        """
        return self.current_page.pageid == self.goal_page.pageid

tests.run_wiki_game_tests(WikiGame)
# %%

class GetContentTool():
    name: str = "get_content"

    @staticmethod
    def execute(task: WikiGame | Any) -> str:
        """
        Get all the content for the wikipedia page you are currently on. Anything which corresponds to a link is wrapped in <link></link> tags.

        Args:
            task (WikiGame | Any): The current task object.

        Returns:
            str: The content of the page with links wrapped
        """
        content = task.current_page.content
        permitted_links = get_permitted_links(task.current_page)
        for word in sorted(permitted_links, key=len, reverse=True):
            content = re.sub(
                r"""(\s|[,.)!?;:'"])(""" + re.escape(word) + r""")(\s|[,.)!?;:'"s])""",
                r"\1<link>\2</link>\3",
                content,
                count=1,
                flags=re.IGNORECASE,
            )
        return content

    @property
    def description(self):
        """
        Provides the description of the GetContentTool.

        Returns:
            dict: The description of the GetContentTool for the API
        """
        description = r"""This is a get content tool, which gives the contents of 
        the page you are currently on.
        
        What it does:
        The generate content tool provides you with the content of a Wikipedia page.

        When it should be used:
        This tool should be used when you want to obtain the content of a particular 
        Wikipedia page. Permitted links will be surrounded with a link tag '<link></link>'.

        Tool limitations:
        - No limitations.
        """
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": { }
            }
        }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": { }
            }
        }


class MovePageTool():
    name: str = "move_page"

    @staticmethod
    def execute(new_page: str, task: WikiGame) -> str:
        """
        Changes your current page to a specified new page which is accessible via a link from the current page. You can only call this function once at a time, as it will take you to a different page.

        Args:
            task (WikiGame): The current task object.
            new_page (str): The title of the new page to move to.

        Returns:
            str: A message indicating the result of the move
        """
        
        if task.is_permitted_link(new_page):
            task.current_page = WikiGame.get_page(new_page)
            task.page_history.append(new_page)
        
            return f"Moved to page: {new_page}."
        else:
            return "This link is not permitted. Could not move page."

    @property
    def description(self):
        """
        Provides the description of the MovePageTool

        Returns:
            dict: The description of the MovePageTool for the API
        """
        description = r"""This is a move page tool, which allows you to move to a 
        page given its title.
        
        What it does:
        Changes the wikipedia page you are currently on to the wikipedia page specified
        by the title.

        When it should be used:
        This tool should be used if you want to change wikipedia pages.

        Tool limitations:
        - If the link is not a permitted link, it will not move pages and you will get the
            message 'This link is not permitted. Could not move page.'.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_page": {
                            "type": "string",
                            "description": "Title of the page you want to move to"
                        },
                    },
                    "required": ["new_page"],
                    "additionalProperties": False,
                }
            }
        }


get_content_tool_inst = GetContentTool()
move_page_tool_inst = MovePageTool()
wiki_game_tools = [get_content_tool_inst, move_page_tool_inst]
# %%

class WikiAgent(SimpleAgent):
    """
    Inherits from SimpleAgent and adds the ability to handle tool calls and refusals in the Wikipedia game context.

    Attributes:
        model (str): The model used for generating responses (inherited)
        tools (List[Any]): List of tools (inherited)
        client (OpenAI): OpenAI client for API calls (inherited)
        task (Any): The current task being executed
        chat_history (List[dict]): History of interactions (inherited)

    Methods:
        get_response(use_tool: bool = True) -> ChatCompletionMessage:
            Get response from the model (inherited)

        execute_tool_calls(message: ChatCompletionMessage) -> List[str]:
            Execute tool calls from the model's response (inherited)

        run(with_tool: bool = True) -> bool:
            Run one loop of the Wikipedia agent (modified below)

    """

    def __init__(
        self,
        task: Any,
        tools: List[Any],
        model="gpt-4o-mini",
        chat_history: List[dict] = None,
        verbose: bool = True,
    ):
        super().__init__(model=model, tools=tools, task=task)

        self.chat_history = chat_history if chat_history else []
        self.full_chat_history = (
            chat_history if chat_history else []
        )  # All messages that have been sent in the chat history.
        self.verbose = verbose
        self.move_counter = 0
        self.start()

    # ========================= Memory (to implement) =========================

    def update_history(
        self, message: str | ChatCompletionMessage | List[str | ChatCompletionMessage]
    ):
        """
        Update self.chat_history and self.full_chat_history with a message or list of messages.

        Args:
            message (str | List[str]): The message to add to the chat history
        """

        if isinstance(message, list):
            self.chat_history.extend(message)
            self.full_chat_history.extend(message)
        else:
            self.chat_history.append(message)
            self.full_chat_history.append(message)


    def reset_history(self):
        """
        Empty self.chat_history of the agent.
        """
        self.chat_history = []

    # ========================= Observation parsing (to implement) =========================
    def handle_tool_calls(self, response: ChatCompletionMessage):
        """
        Handles tool_calls in the wikipedia game context:
            - Executes the tool calls using execute_tool_calls
            - Appends the original tool call & tool_responses to the chat_history
            - If the agent has moved to a new page, resets the chat_history
            - If not, get the next_step_message instruction from the task and append it to chat_history

        Args:
            response (ChatCompletionMessage): The response from the model
        """
        assert response.tool_calls

        self.chat_history.append(response)

        # Execute the tool calls
        results = self.execute_tool_calls(response)

        # Append the tool calls and responses to the chat history
        tool_messages = [apply_tool_call_format(tool_call, result)
                                for tool_call, result in zip(response.tool_calls, results)]

        if self.verbose:
            print(f"Tool message response : {results}")

        self.update_history(tool_messages)

        if any(result.startswith("Moved to page") for result in results):
            self.move_counter += 1
            self.reset_history()
            self.start()
        else:
            self.update_history(self.task.next_step_instruction)


    def handle_refusal(self, response: ChatCompletionMessage):
        """
        Handles refusals in the wikipedia game context:

        Args:
            response (ChatCompletionMessage): The response from the model
        """
        if self.verbose:
            print(f"Response rejected for {response.content}.")
        self.update_history(response)
        self.task.next_step_instruction

    # ========================= Implementation logic (to implement) =========================
    def start(self):
        """
        A function to put the starting instructions in agent.chat_history when the agent starts a new page or starts the game.
        """
        self.update_history(self.task.system_instruction)
        self.update_history(self.task.on_page_instruction)
        self.update_history(self.task.next_step_instruction)
        

    def run(self):
        """
        This is the main function that runs the agent in the wikipedia game for 1 loop. It:
            - Gets the current task instruction
            - Gets the response from the model
            - Handles the response in the cases:
                - tool calls (using handle_tool_calls)
                - refusals (using handle_refusal)
                - no tool calls (using update_history)
        """
        message = self.get_response()
        if message.tool_calls:
            self.handle_tool_calls(message)
        else:
            self.update_history(message)

        if message.refusal: 
            self.handle_refusal(message)


# %%

def agent_loop(agent, game, num_loops=10, result_list=None):
    """
    Run the agent loop for a given number of loops

    Args:
        agent (WikiAgent): The agent to run
        game (WikiGame): The game to play
        num_loops (int): The number of loops to run
    """
    agent.start()
    for i in range(num_loops):
        print("Loop #", i)
        agent.run()
        if agent.task.check_win():
            if result_list is not None:
                result_list.append(("Won", agent.move_counter))
            print(f"You have won. It took {agent.move_counter} clicks.")
            return
    if result_list is not None:
        result_list.append(("Lost", agent.move_counter))
    print("Goal not reached.")
    return
    
# %%
game_1 = WikiGame("Barack Obama", "India")
agent = WikiAgent(task=game_1, tools=wiki_game_tools, verbose=True)
agent_loop(agent, game_1, 30)
# %%
game_2 = WikiGame("Albert Einstein", "Aristotle")
agent = WikiAgent(task=game_2, tools=wiki_game_tools, verbose = False)
# agent_loop(agent, game_2, 30)
# %%

for message in agent.chat_history:
    try:
        print(f"{str(message.role)}:\n {str(message.content)}")
    except:
        print(f"""{message["role"]}:\n {message["content"]}""")

# %%
class WikiGamePrompting(WikiGame):
    """
    Inherits from WikiGame and adds improved prompting.

    Attributes:
        starting_page (str): The title of the starting page (inherited)
        goal_page (str): The title of the goal page (inherited)
        current_page (WikipediaPage): The current Wikipedia page (inherited)
        page_history (List[str]): The history of pages visited (inherited)

    Methods:
        get_page(title: str) -> WikipediaPage: Get a Wikipedia page object given a title (inherited)

        get_page_summary(page: WikipediaPage | None = None) -> str: Get the summary of a Wikipedia page (inherited)

        get_permitted_links(title: Optional[str] = None) -> list[str]: Get permitted links for the current page (inherited)

        is_permitted_link(link: str) -> bool: Check if a link is permitted (inherited)

        system_instruction -> dict: Generate the starting instructions for the game (modified below)

        on_page_instruction -> dict: Generate instructions for the current page (modified below)

        next_step_instruction -> dict: Generate instructions for the next step (modified below)

        check_win() -> bool: Check if the game has been won (inherited)

    """

    @property
    def system_instruction(self):
        """
        Provide improved starting instructions for the game.

        Returns:
            dict: The starting instructions. "role" is "system" for system messages.
        """
        return {
            "role": "system",
            "content": f"You are a wikipedia-racing AI. Your goal is to reach {self.goal_page.title} by accessing links from wikipedia pages. Your current page is {self.current_page.title}.",
        }

    @property
    def on_page_instruction(self):
        """
        Provide improved instructions for the current page.

        Returns:
            dict: The instructions for the current page. "role" is "user" for user messages.
        """
        return {
            "role": "user",
            "content": f"""You are currently on page: {self.current_page.title}. Make sure you start by reasoning about what steps you should take to get to the article on {self.goal_page.title}. When coming up with a strategy, make sure to pay attention to the path you have already taken, and if your current strategy doesn't seem to be working out, try something else. In case you're unsure, {self.goal_page.title} has the following summary:\n\n[Begin Summary]\n{self.get_page_summary(self.goal_page)}\n[End Summary]\n\nThe path you have taken so far is {" -> ".join(self.page_history)}.
            """,
        }

    @property
    def next_step_instruction(self):
        """
        Provide improved instructions for the next step.

        Returns:
            dict: The instructions for the next step. "role" is "user" for user messages.
        """

        return {
            "role": "user",
            "content": f"What's your next step to get to {self.goal_page.title}?",
        }

# %%

simple_result_list = []
prompting_result_list = []
num_games = 3
num_iters = 40 

def run_simple():
    for _ in range(num_games):
        game = WikiGame("Drupe", "17th parallel north")
        agent = WikiAgent(task=game, tools=wiki_game_tools)
        agent_loop(agent, game, num_iters, result_list=simple_result_list)

def run_prompting():
    for _ in range(num_games):
        game = WikiGamePrompting("Drupe", "17th parallel north")
        agent = WikiAgent(task=game, tools=wiki_game_tools)
        agent_loop(agent, game, num_iters, result_list=prompting_result_list)

import threading

simple_thread = threading.Thread(target=run_simple)
prompting_thread = threading.Thread(target=run_prompting)

RUN_COMPARISON=False
if RUN_COMPARISON:
    simple_thread.start()
    prompting_thread.start()

    simple_thread.join()
    prompting_thread.join()

    print(simple_result_list)
    print(prompting_result_list)

# %%
class WikiGameReAct(WikiGamePrompting):
    """
    Inherits from WikiGamePrompting and adds the ReAct framework.

    Attributes:
        starting_page (str): The title of the starting page (inherited)
        goal_page (str): The title of the goal page (inherited)
        current_page (WikipediaPage): The current Wikipedia page (inherited)
        page_history (List[str]): The history of pages visited (inherited)

    Methods:

        - get_page(title: str) -> WikipediaPage: Get a Wikipedia page object given a title (inherited)
        - get_page_summary(page: WikipediaPage | None = None) -> str: Get the summary of a Wikipedia page (inherited)
        - get_permitted_links(title: Optional[str] = None) -> list[str]: Get permitted links for the current page (inherited)
        - is_permitted_link(link: str) -> bool: Check if a link is permitted (inherited)
        - system_instruction -> dict: Generate the starting instructions for the game (inherited)
        - on_page_instruction -> dict: Generate instructions for the current page (inherited)
        - next_step_instruction -> dict: Generate instructions for the next step (inherited)
        - check_win() -> bool: Check if the game has been won (inherited)

    """

    def __init__(self, starting_page: str, goal_page: str, tools=None):
        super().__init__(starting_page, goal_page)
        self.tools = tools

    @property
    def system_instruction(self):
        """
        Provided a description of the tools in the system message. When generate is called with tools this is redundant, but when generate is called without tools, this is useful.

        Returns:
            dict: The starting instructions. "role" is "system" for system messages.
        """
        tool_descriptions = []
        if self.tools:
            tool_descriptions = "\n".join([json.dumps(tool.description) for tool in self.tools])
        out_dict = {
            "role": 'system',
            "content": \
                f'''You are playing the WikiGame, where the target is to reach a goal
                page from a starting page following links.
                The aim is to do it in the fewest number of page navigations.

                The following tools will be available to you:
                {tool_descriptions}
                '''
        }
        return out_dict

class WikiAgentReAct(WikiAgent):
    """
    Inherits from WikiAgent and adds the ReAct framework.

    Attributes:
        model (str): The model used for generating responses (inherited)
        tools (List[Any]): List of tools (inherited)
        client (OpenAI): OpenAI client for API calls (inherited)
        task (Any): The current task being executed (inherited)
        chat_history (List[dict]): History of interactions (inherited)

    Methods:
        - get_response(use_tool: bool = True) -> ChatCompletionMessage: Get response from the model (inherited)
        - execute_tool_calls(message: ChatCompletionMessage) -> List[str]: Execute tool calls from the model's response (inherited)
        - update_history(message : str | ChatCompletionMessage | List[str | ChatCompletionMessage]): Update self.chat_history and self.full_chat_history with a message or list of messages. (inherited)
        - reset_history(): Empty self.chat_history of the agent. (inherited)
        - handle_tool_calls(response: ChatCompletionMessage): Handles tool_calls in the wikipedia game context. (inherited)
        - handle_refusal(response: ChatCompletionMessage): Handles refusals in the wikipedia game context. (inherited)
        - start(): A function to put the starting instructions in agent.chat_history when the agent starts a new page or starts the game. (inherited)
        - run() -> bool: Run one loop of the Wikipedia agent (inherited)

    """

    def generate_reason(self) -> ChatCompletionMessage:
        """

        Generate a reason for the agent to take an action. This function should:
            - Get the model to reason about the current state of the game (without tools)
            - Return the response from the model

        Returns:
            ChatCompletionMessage: The response from the model
        """
        self.update_history([
            {
                "role": "user",
                "content": f"Reason about the current state of the game. \
                    What actions can you take to reach {self.task.goal_page}?"
            }
        ])
        return self.get_response(use_tool = False)

    def generate_action(self) -> ChatCompletionMessage:
        """

        Generate an action for the agent to take. This function should:
            - Get the model to generate an action for the agent to take (with tools)
            - Return the response from the model

        Returns:
            ChatCompletionMessage: The response from the model

        """
        self.update_history([
            {
                "role": "user",
                "content": f"Given your previous reasoning, select the most appropriate action \
                    to reach the target page {self.task.goal_page}."
            }
        ])
        return self.get_response(use_tool = True)


    def generate_reason_and_action(self):
        """

        Generate a Reason and Action for the agent to take. This function should:
            - Generate a Reason
            - Add the Reason to the chat history
            - Generate an Action
            - Return the Action so that tool calls can be handled

        Returns:
            ChatCompletionMessage: The action from the model

        """
        message = self.generate_reason()
        self.update_history(message)
        return self.generate_action()

    def run(self):
        """
        Run one loop of the agent.

        This function should:
            - Generate a Reason and Action
            - Handle the tool calls, refusals, and no tool calls in the model response
        """

        message = self.generate_reason_and_action()
        print('generate_reason_and_action', message)
        if message.tool_calls:
            self.handle_tool_calls(message)
        else:
            self.update_history(message)

        if message.refusal: 
            self.handle_refusal(message)

# %%
game = WikiGameReAct("Drupe", "17th parallel north", tools=wiki_game_tools)
agent = WikiAgentReAct(task=game, tools=wiki_game_tools)
agent_loop(agent, game, num_iters, result_list=prompting_result_list)

# %%
class TestPathTool():
    """
    Implements a tool that allows the agent to test paths from the current state of the game.

    Attributes:
        name (str): The name of the tool

    Methods:
        execute(task: Any, path: str) -> str: Test if a given path is valid.

        description -> dict: Provides the description of the TestPathTool tool for the API
    """

    name = "test_path"

    def execute(self, task: WikiGame, path: str) -> str:
        """
        Test if a given path is valid.

        Args:
            path (str): A string representing a path, e.g., "Barack Obama -> Indonesia -> India"

        Returns:
            str: A message indicating whether the path is valid or where it fails.
        """
        invalid_path_msg = "This is not a valid path."
        segments = path.split(" -> ")
        if task.current_page.title != segments[0]:
            return invalid_path_msg
        
        def is_permitted(current_page, link):
            permitted_links = [link for link in current_page.links if link in current_page.content]
            return link.lower() in (x.lower() for x in permitted_links)
            
        current_page = task.current_page
        for segment in segments[1:]:
            if not is_permitted(current_page, segment):
                return invalid_path_msg
            current_page = WikiGame.get_page(segment)
        return f"The path {path} is valid"

    @property
    def description(self) -> dict:
        description = """
        The path tool checks the validity of a given path of links.
        If any of the links are broken or unpermitted, it will return an error.

        What it does:
        Given a path in the format 'Page1 -> Page2 -> Page3... -> PageN' the tool will check the following properties:
        1. Each successive link exists in the current page
        2. The terminal page corresponds to the target page
        3. Links can be navigated to

        Limitations:
        1. The page titles must be separated by ' -> ' for a well-formed path
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "String describing the path through wikipedia page titles."
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                }
            }
        }

TestPathTool_inst = TestPathTool()
wiki_game_tools = [get_content_tool_inst, move_page_tool_inst, TestPathTool_inst]

# %%
game = WikiGameReAct("Barack Obama", "India", tools = wiki_game_tools)
agent = WikiAgentReAct(game, model="gpt-4o-mini", tools = wiki_game_tools)
agent_loop(agent, game, 40)