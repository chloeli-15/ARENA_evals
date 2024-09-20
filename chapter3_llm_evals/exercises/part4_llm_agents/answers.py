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
from utils import evaluate_expression, apply_user_format, apply_assistant_format, apply_system_format, establish_client_anthropic, establish_client_OpenAI, retry_with_exponential_backoff
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

# tests.ArithmeticTaskTests(ArithmeticTask)

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

        self.chat_history.append(response)

        # if use_tool and response.choices[0].message.tool_calls is not None:
        #     tool_responses = self.execute_tool_calls(response.choices[0].message)

        return response

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
        agent.run(with_tool=False)

# agent_loop(arithmetic_agent_1, arithmetic_task_1)

#%%
arithmetic_task_1.is_solved

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

#%%
def get_permitted_links(current_page: WikipediaPage) -> list[str]:
    """
    Get "permitted" links (i.e. links that are in the content of the page) from a Wikipedia page.

    Args:
        current_page (WikipediaPage): The current Wikipedia page

    Returns:
        list[str]: A list of permitted links from current_page

    """
    all_links = current_page.links
    return [link for link in all_links if link in current_page.content]

page = wikipedia.page("Human", redirect= False, auto_suggest=True)
get_permitted_links(page)


#%%
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
        return {"role": "system", "content": f"You are playing the wikipedia game. You start on a start page and attempt to navigate to a target page. For each step, select one of a list of valid links. Your goal is to navigate to the target page in as few steps as possible."}

    @property
    def on_page_instruction(self) -> dict:
        """
        Tell the agent what page they are on and give a summary of the page, formatted as a user prompt.

        Returns:
            dict: The instructions for the current page. The "role" is "user" for user messages.
        """
        return {"role": "user", "content": f"You are currently on the page '{self.current_page.title}'. Here is a summary of this page:\n{self.current_page.summary[:1000]}\nYour target page is '{self.goal_page.title}'."}

    @property
    def next_step_instruction(self) -> dict:
        """
        Ask the agent "What's the next step?" after making a tool call, formatted as a user prompt.

        Returns:
            dict: The instructions for the next step. The "role" is "user" for user messages.
        """
        return {"role": "user", "content": f"What's your next step? Here is a link of valid links you can choose from:\n{self.get_permitted_links(self.current_page.title)}"}

    # ========================= Task State management (to implement) =========================

    def check_win(self) -> bool:
        """
        Check if the agent has won the game.

        Returns:
            bool: True if the agent has won, False otherwise.
        """
        return self.current_page == self.goal_page

tests.run_wiki_game_tests(WikiGame)

#%%
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
        desc = {
            "type": 'function',
            'function': {
                "name": self.name,
                "description": "Use this tool to get the content of the current wikipedia page, including links you can select to go to that page. Links are wrapped in <link></link> tags.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                }
            }
        }
        return desc


class MovePageTool():
    name: str = "move_page"

    @staticmethod
    def execute(new_page: str, task: Any) -> str:
        """
        Changes your current page to a specified new page which is accessible via a link from the current page. You can only call this function once at a time, as it will take you to a different page.

        Args:
            task (WikiGame): The current task object.
            new_page (str): The title of the new page to move to.

        Returns:
            str: A message indicating the result of the move
        """
        new_page = new_page.replace("_", " ")
        if task.is_permitted_link(new_page):
            task.current_page = task.get_page(new_page)
            task.page_history.append(new_page)
            return f"You are now on the page for '{new_page}."
        else:
            return f"'{new_page} is not a valid link from the current page."

    @property
    def description(self):
        """
        Provides the description of the MovePageTool

        Returns:
            dict: The description of the MovePageTool for the API
        """
        desc = {
            "type": 'function',
            'function': {
                "name": self.name,
                "description": "Use this tool to move to the new wikipedia page you have selected.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_page": {
                            "type": "string",
                            "description": "Title of the new page you wish to navigate to from the current page.",
                        },
                    },
                    "required": ["new_page"],
                    "additionalProperties": False,
                }
            }
        }
        return desc


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
            [self.update_history(msg) for msg in message]
            return
        elif isinstance(message, str):
            message = ChatCompletionMessage(content=message)
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
        tool_responses = self.execute_tool_calls(response)
        for tool_call, tool_response in zip(response.tool_calls, tool_responses):
            self.chat_history.append(apply_tool_call_format(tool_call, content=tool_response))
            self.full_chat_history.append(apply_tool_call_format(tool_call, content=tool_response))

    def handle_refusal(self, response: ChatCompletionMessage):
        """
        Handles refusals in the wikipedia game context:

        Args:
            response (ChatCompletionMessage): The response from the model
        """
        # Append the refusal to the chat history
        self.chat_history.append(apply_assistant_format(response.refusal))
        # Update the task state, nothing to do

    # ========================= Implementation logic (to implement) =========================
    def start(self):
        """
        A function to put the starting instructions in agent.chat_history when the agent starts a new page or starts the game.
        """
        sys_msg = apply_system_format(self.task.system_instruction)
        user_msg = apply_user_format(self.task.on_page_instruction)
        self.chat_history.extend([sys_msg, user_msg])

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
        # getting a task
        if self.task.check_win():
            print("Complete! ", self.task.page_history)
            return
        # getting a response from the model
        self.start()
        response = self.get_response(use_tool=True)
        self.full_chat_history.append(apply_assistant_format(response.choices[0].message))
        # handling the model response, including tool calls, refusals, no tool calls, parsing and checking final answers, errors.
        if response.refusal is not None:
            self.handle_refusal(response)
        else:
            if response.tool_calls is not None:
                self.handle_tool_calls(response)   
            at_goal = self.task.check_win()
            print(f"{at_goal=}")
            if at_goal:
                print("Complete! ", self.task.page_history)
        # managing memory: storing the history of messages to self.chat_history

        # managing task state: staying on the same task or moving to the next task at the end of the loop

def agent_loop(agent, game, num_loops=10):
    """
    Run the agent loop for a given number of loops

    Args:
        agent (WikiAgent): The agent to run
        game (WikiGame): The game to play
        num_loops (int): The number of loops to run
    """

    agent.task = game
    for i in range(num_loops):
        agent.run()

# %%

game_1 = WikiGame("Barack Obama", "India")
agent = WikiAgent(task=game_1, tools=wiki_game_tools)
agent_loop(agent, game_1, 30)

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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.chat_history,
            tools=[tool.description for tool in self.tools] if use_tool else None,
            tool_choice="auto" if use_tool else None,
        )
        return response.choices[0].message

    def execute_tool_calls(self, message: ChatCompletionMessage) -> List[str]:
        """
        Execute the tool calls in the message and return a list of tool_responses.

        Args:
            message (ChatCompletionMessage): The message containing the tool calls

        Returns:
            List[str]: A list of tool responses (as strings)
        """
        tool_calls = message.tool_calls

        tool_responses = []
        for tool_call in tool_calls:
            if not self.task:
                raise ValueError("Task is not set. Cannot execute tool calls.")
            func = next(
                (tool for tool in self.tools if tool.name == tool_call.function.name),
            )
            arguments = json.loads(tool_call.function.arguments)
            tool_response = func.execute(**arguments, task=self.task)
            tool_responses.append(tool_response)

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
        instruction = self.task.instruction
        self.chat_history.append(apply_user_format(instruction))
        response = self.get_response(use_tool=with_tool)
        return response

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
        self.page_history: List[str] = [starting_page]
        print(starting_page)
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

    # ========================= Task State Management (to implement) =========================

    @property
    def system_instruction(self) -> dict:
        """
        Generate the starting instructions for the game.

        Returns:
            dict: The starting instructions. "role" is "system" for system messages.
        """
        return {
            "role": "system",
            "content": "You are a wikipedia-racing AI. Your aim is to reach the goal page by accessing links from a series of wikipedia pages.",
        }

    @property
    def on_page_instruction(self) -> dict:
        """
        Generate instructions for the current page.

        Returns:
            dict: The instructions for the current page. "role" is "user" for user messages.
        """
        return {
            "role": "user",
            "content": f"""You are currently on page: {self.current_page.title}. Your goal page is {self.goal_page.title}.""",
        }

    @property
    def next_step_instruction(self) -> dict:
        """
        Generate instructions for the next step.

        Returns:
            dict: The instructions for the next step. "role" is "user" for user messages.
        """
        return {
            "role": "user",
            "content": f"What's your next step?",
        }

    def check_win(self) -> bool:
        return self.current_page == self.goal_page

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
        )  # All messages that have been sent in the chat history. We have to erase each time a new page is reached for context window reasons.
        self.verbose = verbose
        self.start()

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

        # Update history
        self.update_history(response)

        # Execute the tool calls
        tool_responses = self.execute_tool_calls(response)

        # Add tool calls and responses to the history
        for tool_call, tool_response in zip(response.tool_calls, tool_responses):
            self.update_history(apply_tool_call_format(tool_call, tool_response))

            if self.verbose:
                print(
                    f"\nTOOL CALL: \nTool = {tool_call.function.name}, Args = {tool_call.function.arguments} \nTOOL RESPONSE:\n {tool_response[:300]}"
                )

        # Move to new page if necessary
        if any("Moving page" in tool_response for tool_response in tool_responses):
            self.reset_history()
            print(
                f"""{("-" * 50)} \n\nMOVED TO PAGE \n\nPATH HISTORY (N={len(self.task.page_history)}): {" -> ".join(self.task.page_history)} \n\n{("-"*50)}"""
            )

            # Give starting instructions if moved to a new page
            self.start()

        # Otherwise ask the agent what the next step is

        else:
            next_step_message = self.task.next_step_instruction
            self.update_history(next_step_message)
            if self.verbose:
                print(f"""\nUSER: \n{next_step_message["content"]}""")

    def handle_refusal(self, response: ChatCompletionMessage):
        """
        Handles refusals in the wikipedia game context:

        Args:
            response (ChatCompletionMessage): The response from the model
        """
        self.update_history(apply_assistant_format(response.refusal))
        if self.verbose:
            print(f"\nMODEL REFUSAL: {response.refusal}")

    def start(self):
        """
        A function to put the starting instructions in agent.chat_history when the agent starts a new page or starts the game.
        """
        instruction_messages = [
            self.task.system_instruction,
            self.task.on_page_instruction,
        ]
        self.update_history(instruction_messages)
        if self.verbose:
            print(
                f"\nSYSTEM: \n{instruction_messages[0]['content']} \n\nUSER: \n{instruction_messages[1]['content']}"
            )

    def run(self):
        """
        This function runs the agent in the wikipedia game context. It:
            - Gets the current task instruction
            - Gets the response from the model
            - Handles the response in the cases:
                - tool calls (using handle_tool_calls)
                - refusals (using handle_refusal)
                - no tool calls (using update_history)
        """

        if self.task.check_win():
            print("Complete! ", self.task.page_history)
            return
        
        # Get the response from the model
        response = self.get_response()

        # Handle the response
        ## If tool calls, do the tool calls and return the response
        if response.tool_calls:
            self.handle_tool_calls(response)

        ## If no tool call: Handle edge cases
        ### Check if there's a refusal to answer:
        elif response.refusal:
            self.handle_refusal(response)

        # Else response content does not contain tool calls or refusal, and we add it to the chat_history in an assistant format.
        else:
            self.update_history(apply_assistant_format(response.content))
            if self.verbose:
                print(f"\nMODEL RESPONSE: \n{response.content}")

game_1 = WikiGame("Caesium", "Pope_Silverius")
agent = WikiAgent(task=game_1, tools=wiki_game_tools)
agent_loop(agent, game_1, 100)
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

        get_instructions(system: bool, on_page: bool, next_step: bool) -> str: Generate instruction messages based on the current game state (inherited)

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
            "role": "system", "content": 
            f"You are playing the wikipedia game. You start on a start page and attempt to navigate to a target page. For each step, select one of a list of valid links. Your goal is to navigate to the target page in as few steps as possible. For your general strategy, start by navigating from your starting page to more general pages which likely contain many links, and then narrow in from there to the target page, like this: Narrow article (with few links) -> General article (with many links) -> Narrow article (with few links)."}

    @property
    def on_page_instruction(self):
        """
        Provide improved instructions for the current page.

        Returns:
            dict: The instructions for the current page. "role" is "user" for user messages.
        """
        return {
            "role": "user", "content": 
            f"You are currently on the page '{self.current_page.title}'. Here is a summary of this page:\n{self.current_page.summary[:1000]}\nYour target page is '{self.goal_page.title}'. Brainstorm some possible paths through links, and then use the 'test_path' tool to check the path validity."}

    @property
    def next_step_instruction(self):
        """
        Provide improved instructions for the next step.

        Returns:
            dict: The instructions for the next step. "role" is "user" for user messages.
        """
        return {
            "role": "user", "content": 
            f"Examine the validity of the paths you preposed. What's your next step? Here is a link of valid links you can choose from:\n{self.get_permitted_links(self.current_page.title)}. Take your time in choosing which link to select. Here is a list of the pages you've already visited, which you generally shouldn't visit again: {self.page_history}"}

#Improved WikiGame and WikiAgent
# game = WikiGame("Linux", "Dana Carvey")
game = WikiGamePrompting("Caesium", "Pope Silverius")
agent = WikiAgent(game, model="gpt-4o-mini", tools=wiki_game_tools)
# agent_loop(agent, game, 30)

#%%
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
        return {
            "role": "system", "content": 
            f"You are playing the wikipedia game. You start on a start page and attempt to navigate to a target page. For each step, select one of a list of valid links. Your goal is to navigate to the target page in as few steps as possible. For your general strategy, start by navigating from your starting page to more general pages which likely contain many links, and then narrow in from there to the target page, like this: Narrow article (with few links) -> General article (with many links) -> Narrow article (with few links). Here are instructions for the tools you have available: {[tool.description for tool in self.tools]}"}

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
        instruction = "Think through the available links, and evaluate your top choices for which is most likely to get you toward the target page. Then, use the available 'test_path' tool to check the validity of possible paths."
        self.update_history(apply_user_format(instruction))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.chat_history,
        )

        return response

    def generate_action(self) -> ChatCompletionMessage:
        """

        Generate an action for the agent to take. This function should:
            - Get the model to generate an action for the agent to take (with tools)
            - Return the response from the model

        Returns:
            ChatCompletionMessage: The response from the model

        """
        instruction = "Given your previous reasoning, now select which link you want to follow to get toward your target page."
        self.update_history(apply_user_format(instruction))
        # for i, hist in enumerate(self.chat_history):
        #     print(i, hist)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.chat_history,
            tools=[tool.description for tool in self.tools],
            tool_choice="auto",
        )

        return response

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
        reason_response = self.generate_reason()
        if hasattr(reason_response, 'refusal') and reason_response.refusal is not None:
            self.handle_refusal(reason_response)
        self.update_history(apply_assistant_format(reason_response.choices[0].message.content))
        action_response = self.generate_action()
        return action_response


    def run(self):
        """
        Run one loop of the agent.

        This function should:
            - Generate a Reason and Action
            - Handle the tool calls, refusals, and no tool calls in the model response
        """

        if self.task.check_win():
            print("Complete! ", self.task.page_history)
            return

        self.start()
        
        response = self.generate_reason_and_action().choices[0].message

        # Get the response from the model
        # response = self.get_response()

        # Handle the response
        ## If tool calls, do the tool calls and return the response
        if response.tool_calls:
            self.handle_tool_calls(response)

        ## If no tool call: Handle edge cases
        ### Check if there's a refusal to answer:
        elif response.refusal:
            self.handle_refusal(response)

        # Else response content does not contain tool calls or refusal, and we add it to the chat_history in an assistant format.
        else:
            self.update_history(apply_assistant_format(response.content))
            if self.verbose:
                print(f"\nMODEL RESPONSE: \n{response.content}")

# WikiGame and WikiAgent with ReAct
game = WikiGameReAct("Drupe", "17th parallel north", tools=wiki_game_tools)
agent = WikiAgentReAct(game, model="gpt-4o-mini", tools = wiki_game_tools)
agent_loop(agent, game,40)

#%%
game = WikiGamePrompting("Drupe", "17th parallel north",)
agent = WikiAgent(game, model="gpt-4o-mini", tools=wiki_game_tools)
agent_loop(agent, game, 30)

#%%
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

    def execute(self, task: Any, path: str) -> str:
        """
        Test if a given path is valid.

        Args:
            path (str): A string representing a path, e.g., "Barack Obama -> Indonesia -> India"

        Returns:
            str: A message indicating whether the path is valid or where it fails.
        """
        print(path)
        pages = path.split(' -> ')
        for start_page, end_page in zip(pages[:-1], pages[1:]):
            page = wikipedia.page(start_page, redirect= True, auto_suggest=True)
            if end_page.lower().replace('_', ' ') not in [link.lower().replace('_', ' ') for link in page.links if link in page.content]:
                return 'invalid'
        return 'valid'


    @property
    def description(self) -> dict:
        """
        Provides the description of the tool

        Returns:
            dict: The description of the tool
        """
        desc = {
            "type": 'function',
            'function': {
                "name": self.name,
                "description": "Takes in a string representing a path, e.g., 'Barack Obama -> Indonesia -> India', and returns whether that path is valid or invalid.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "string representing a path to check for validity, e.g., 'Barack Obama -> Indonesia -> India'.",
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                }
            }
        }
        return desc

TestPathTool_inst = TestPathTool()
wiki_game_tools = [get_content_tool_inst, move_page_tool_inst, TestPathTool_inst]

game = WikiGameReAct("Drupe", "17th parallel north", tools = wiki_game_tools)
agent = WikiAgentReAct(game, model="gpt-4o-mini", tools = wiki_game_tools)
agent_loop(agent,game, 40)
