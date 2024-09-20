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
    apply_system_format,
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


def agent_loop(agent, task, num_loops: int = 1):
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

# %%#################
# WIKIPEDIA AGENT #
###################

# Retrieve a Wikipedia page from its title
page = wikipedia.page("Large language model")

# # Access basic page information
# print("Title:", page.title)
# print("\nURL", page.url)
# print(f"\nSummary (word count {len( page.summary.split())}):", page.summary)
# print(
#     f"\nContent (word count {len( page.content.split())}):",
#     page.content[:1000],
#     "......",
# )
# print(
#     f"""\nLinks (link count {len(page.links)}): [{", ".join(page.links[:7])}, ......]"""
# )

# %%
# page = wikipedia.page("python")
# %%
# page = wikipedia.page("Animalss", auto_suggest=False)


# %%
def get_page(title: str) -> WikipediaPage:
    """
    Get a Wikipedia page object given a title. If the title is ambiguous, choose the first option. If the title is not found, try to find a similar title.

    redirect allows wikipedia to find the closest possible page, with some small errors
    auto_suggest also suggests a close page, but may change the result quite drastically

    Args:
        title (str): The title of the Wikipedia page

    Returns:
        WikipediaPage: The Wikipedia page
    """
    try:
        return wikipedia.page(title, auto_suggest=False, redirect=True)
    # take the first option
    except DisambiguationError as e:
        return wikipedia.page(e.options[0], auto_suggest=False, redirect=True)
    except PageError as e:
        return wikipedia.page(title, auto_suggest=True, redirect=True)


# %%
# def get_permitted_links(current_page: WikipediaPage) -> list[str]:
#     """
#     Get "permitted" links (i.e. links that are in the content of the page) from a Wikipedia page.

#     Args:
#         current_page (WikipediaPage): The current Wikipedia page

#     Returns:
#         list[str]: A list of permitted links from current_page

#     """
#     content_page_links = []
#     for link in current_page.links:
#         if link in page.content:
#             content_page_links.append(link)
#     return content_page_links


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
        prompt = """
            You are an AI assistant playing the Wikipedia Game.
            Your goal is to navigate from a starting Wikipedia page to a goal page using only the links available on each page.
            Here are your instructions:
            You start on a given Wikipedia page. Your objective is to reach a specified goal page by following links.
            On each turn, you can either:
            You will be given information about the current page via a summary.
            You can only move to pages that are directly linked from your current page.
            Try to reach the goal page in as few moves as possible; efficiency is key.
            Analyze the content of each page to find connections to the goal topic and inform your navigation choices."""
        return apply_system_format(prompt)

    @property
    def on_page_instruction(self) -> dict:
        """
        Tell the agent what page they are on and give a summary of the page, formatted as a user prompt.

        Returns:
            dict: The instructions for the current page. The "role" is "user" for user messages.
        """
        page = self.current_page
        page_summary = self.get_page_summary(page)
        instruction = f""""CURRENT PAGE: You are on this page {page.title}, PAGE SUMMARY: {page_summary}, GOAL PAGE: {self.goal_page.title}"""
        return apply_user_format(instruction)

    @property
    def next_step_instruction(self) -> dict:
        """
        Ask the agent "What's the next step?" after making a tool call, formatted as a user prompt.

        Returns:
            dict: The instructions for the next step. The "role" is "user" for user messages.
        """
        # Verify this instruction
        next_instruction = "What is your next step? If you want to move, please only move to a page, where the title of that page is enclosed by <link>title</link>"
        return apply_user_format(next_instruction)

    # ========================= Task State management (to implement) =========================

    def check_win(self) -> bool:
        """
        Check if the agent has won the game.

        Returns:
            bool: True if the agent has won, False otherwise.
        """
        return self.goal_page.title == self.current_page.title


tests.run_wiki_game_tests(WikiGame)


# %%
class GetContentTool:
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
        permitted_links = task.get_permitted_links()
        print("PERMITTED", permitted_links)
        for word in sorted(permitted_links, key=len, reverse=True):
            content = re.sub(
                r"""(\s|[,.)!?;:'"])(""" + re.escape(word) + r""")(\s|[,.)!?;:'"s])""",
                r"\1<link>\2</link>\3",
                content,
                count=1,
                flags=re.IGNORECASE,
            )
        print("CONTENT", content[:500])
        return content

    @property
    def description(self):
        """
        Provides the description of the GetContentTool.

        NOTE: do not supply the Task object; we supply this manually ourselves as part of scaffolding. The model does not need to pass this.

        Returns:
            dict: The description of the GetContentTool for the API
        """

        return {
            "type": "function",
            "function": {
                "name": GetContentTool.name,
                "description": "Retrieves the full content of the current Wikipedia page. All wiki-links within the content are wrapped in <link></link> tags for easy identification. Use this to analyze the current page and find relevant links to navigate towards the goal.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }


class MovePageTool:
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

        # replace " " with "_"
        new_page_normalized = new_page.replace("_", " ")
        # check if valid or not
        if task.is_permitted_link(new_page_normalized):
            task.current_page = task.get_page(new_page_normalized)
            task.page_history.append(task.current_page.title)
            return f"Moving page to {task.current_page.title}"
        else:
            return f"Couldn't move page to {new_page}. This is not a valid link."

    @property
    def description(self):
        """
        Provides the description of the MovePageTool

        Returns:
            dict: The description of the MovePageTool for the API
        """
        return {
            "type": "function",
            "function": {
                "name": MovePageTool.name,
                "description": "Changes your current page to a specified new page which is accessible via a link from the current page. You can only call this function ONCE, as it will take you to a different page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_page": {
                            "type": "string",
                            "description": "The title of the new Wikipedia page to move to.",
                        },
                    },
                    "required": ["new_page"],
                },
            },
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
        # isinstance(object, reference_type) will check for referenceType and all its subclasses
        # type(object)==reference_type will only check for the exact type
        # do not use List from typing.List for runtime checking; use `list` instead
        if type(message) is list:
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
        if self.verbose:
            print("\nTool calls:", response.tool_calls)

        # update chat history and full chat history
        self.update_history(response)

        # Execute the tool calls and append tool responses to chat history
        tool_calls = response.tool_calls
        try:
            tool_responses = self.execute_tool_calls(response)
            for tool_call, tool_response in zip(tool_calls, tool_responses):
                self.update_history(apply_tool_call_format(tool_call, tool_response))
                if self.verbose:
                    print(
                        f"\nTool call: {tool_call.function.name}, ARGS: {tool_call.function.arguments}"
                    )
                    print(f"Tool response: {tool_response[:100]}")

            # check if move to a new page
            # if one of the tool calls is a move page, then we want to reset this chat history and also have start which appends the system and on_page_instruction
            if any(
                tool_call.function.name == MovePageTool.name for tool_call in tool_calls
            ):
                self.reset_history()
                self.start()
            else:
                self.update_history(self.task.next_step_instruction)

        except Exception as e:
            print(f"\nError handling tool calls: {e}")

    def handle_refusal(self, response: ChatCompletionMessage):
        """
        Handles refusals in the wikipedia game context:

        Args:
            response (ChatCompletionMessage): The response from the model
        """
        print("\nModel Refusal:", response.refusal)
        self.update_history(apply_assistant_format(response.refusal))

    # ========================= Implementation logic (to implement) =========================
    def start(self):
        """
        A function to put the starting instructions in agent.chat_history when the agent starts a new page or starts the game.
        """
        instruction_messages = [
            self.task.system_instruction,
            self.task.on_page_instruction,
        ]
        self.update_history(instruction_messages)

        print(
            f"\nSYSTEM: \n{instruction_messages[0]['content'][:100]} \n\nUSER: \n{instruction_messages[1]['content'][:100]}"
        )

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
        # task_instruction = self.task.current_task_instruction
        # self.chat_history.append(apply_user_format(task_instruction))

        # get initial tool response from model
        response = self.get_response(use_tool=True)

        # handle this response
        # if it has tool calls
        if response.tool_calls:
            self.handle_tool_calls(response)

        # no tool calls, try again in the next time step
        elif response.refusal:
            self.handle_refusal(response)

        # no tool calls and it did not refuse, its final
        else:
            # TODO: see if response has role as system
            self.update_history(response)


# %%
def agent_loop(agent, game, num_loops=10):
    """
    Run the agent loop for a given number of loops

    Args:
        agent (WikiAgent): The agent to run
        game (WikiGame): The game to play
        num_loops (int): The number of loops to run
    """

    for i in range(num_loops):
        print(f"TIME STEP {i}")
        if not agent.task.check_win():
            agent.run()
        else:
            print(f"\nAll tasks solved in {i} steps.")
            return
    print("TASK FAILED")


# %%
game_1 = WikiGame("Fried Rice", "Stalin")
agent = WikiAgent(task=game_1, tools=wiki_game_tools)
agent_loop(agent, game_1, 30)

# # %%
# for message in agent.full_chat_history:
#     try:
#         print(f"{str(message.role)}:\n {str(message.content)}")
#     except:
#         print(f"""{message["role"]}:\n {message["content"]}""")
# # %%


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

    NOTES:
    The major benefits to prompting come from putting the goal page title in the next step instruction and system prompt.
    You want to add the current page and goal page wherever you can, especially in the recent prompts
    The takeaway is that the "next step to the goal of ..." is really important prompt; during the exploration key decision making step, prompt it well

    Without removal - 8 to 10 steps
    Removed goal from system| removed goal and page_history and page summary from on_page - time step fail
    Removed goal from system| - 15 time steps
    Removed goal and page summary and page history from on_page| - 20 time steps
    Removed goal from next_page - time step fail

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
            "content": f"You are a wikipedia-racing AI. Your goal is to reach {self.goal_page.title} by accessing links from wikipedia pages. Your current page is {self.current_page.title}. ",
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
            "content": f"""You are currently on page: {self.current_page.title}. Make sure you start by reasoning about what steps you should take to get to the article on {self.goal_page.title}. 
            When coming up with a strategy, make sure to pay attention to the path you have already taken, and if your current strategy doesn't seem to be working out, try something else. In case you're unsure, {self.goal_page.title} has the following summary:\n\n[Begin Summary]\n{self.get_page_summary(self.goal_page)}\n[End Summary]\n
            The path you have taken so far is {" -> ".join(self.page_history)} Try not to backtrack to a previous page unless neccessary.
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
            # this is the most important step!
            "content": f"What's your next step to the goal of {self.goal_page.title}?",
        }


# %%

start_end = ("Linux", "Dana Carvey")

# %%
# Original WikiGame and WikiAgent
game = WikiGame(*start_end)
agent = WikiAgent(game, model="gpt-4o-mini", tools=wiki_game_tools)
agent_loop(agent, game, 30)

# %%
# Improved WikiGame and WikiAgent
gameBetterPrompt = WikiGamePrompting(*start_end)
agentBetter = WikiAgent(gameBetterPrompt, model="gpt-4o-mini", tools=wiki_game_tools)
agent_loop(agentBetter, gameBetterPrompt, 30)


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
        tool_string = ""
        for tool in self.tools:
            tool_string = tool_string + " " + str(tool.description)

        return {
            "role": "system",
            "content": f"""You are a wikipedia-racing AI. Your goal is to reach {self.goal_page.title} page by accessing links from wikipedia pages. Your current page is {self.current_page.title}. \n
            The description of the available tools at your disposal is:{tool_string} """,
        }


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
        self.update_history(
            apply_user_format(
                f"""Think carefully step-by-step, for how you are going to take this sequence of links. \n
                Lay out a plan for how you might possibly chart a linear path of big ideas towards the goal of {self.task.goal_page.title}"""
            )
        )
        return self.get_response(use_tool=False)

    def generate_action(self) -> ChatCompletionMessage:
        """

        Generate an action for the agent to take. This function should:
            - Get the model to generate an action for the agent to take (with tools)
            - Return the response from the model

        Returns:
            ChatCompletionMessage: The response from the model

        """
        self.update_history(
            apply_user_format(
                f"What action do you want to take to the goal of {self.task.goal.title}"
            )
        )
        return self.get_response(use_tool=True)

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
        reason = self.generate_reason()
        print("REASONING IS", reason.content)
        self.update_history(reason)
        return self.generate_action()

    def run(self):
        """
        Run one loop of the agent.

        This function should:
            - Generate a Reason and Action
            - Handle the tool calls, refusals, and no tool calls in the model response
        """
        # task_instruction = self.task.current_task_instruction
        # self.chat_history.append(apply_user_format(task_instruction))

        # get initial tool response from model
        response = self.generate_reason_and_action()

        # handle this response
        # if it has tool calls
        if response.tool_calls:
            self.handle_tool_calls(response)

        # no tool calls, try again in the next time step
        elif response.refusal:
            self.handle_refusal(response)

        # no tool calls and it did not refuse, its final
        else:
            # TODO: see if response has role as system
            self.update_history(response)


# %%
# WikiGame and WikiAgent with improved prompting
game = WikiGamePrompting("Drupe", "17th parallel north")
agent = WikiAgent(task=game, tools=wiki_game_tools)
agent_loop(agent, game, 40)

# %%
# WikiGame and WikiAgent with ReAct
game = WikiGameReAct("Drupe", "17th parallel north", tools=wiki_game_tools)
agent = WikiAgentReAct(task=game, tools=wiki_game_tools)
agent_loop(agent=agent, game=game, num_loops=40)
