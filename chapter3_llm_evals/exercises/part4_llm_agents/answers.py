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
section_dir = (exercises_dir / "part4_llm_agents").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(section_dir)
import solutions as solutions
import tests as tests
from utils import import_json, save_json, retry_with_exponential_backoff, pretty_print_questions, load_jsonl, omit
from utils import countrylist
from utils import evaluate_expression, apply_user_format, apply_assistant_format, establish_client_anthropic, establish_client_OpenAI, retry_with_exponential_backoff

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
client = OpenAI()
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
        keys = [f"{str(self.num1)} {op} {str(self.num2)}" for op in self.operations] 
        answers = {key: evaluate_expression(key) for key in keys}

        
        return answers

    @property
    def get_current_task(self) -> str:
        """
        Gets the current task for the agent

        Returns:
            str: A string containing the current task
        """
        for i, key  in enumerate(self._generate_answers().keys()):
            if i == self.current_task_number:
                return key
            
        return ""

    @property
    def instruction(self) -> str:
        """
        Gets a string containing instructions for the current task for the agent. (This will be fed to the agent as a user prompt)

        Returns:
            str: A string containing the instructions for the current task
        """
        return f"Calculate the result of the following expression: {str(self.num1)} {self.operations[self.current_task_number]} {str(self.num2)}. Give your final answer in the format: <answer>NUMBER</answer>, where NUMBER is a numerical value"

    def check_solved(self) -> bool:
        """
        Checks if all tasks have been solved

        Returns:
            bool: True if all tasks have been solved, False otherwise
        """

        for value in self.is_solved.values():
            if value != True:
                return False
        return True

    def check_answer(self, model_answer: str) -> bool:
        """
        Checks if the model's answer is correct

        Args:
            model_answer (str): The model's answer

        Returns:
            bool: True if the model's answer is correct, False otherwise
        """
        correct_answer = self.correct_answers[self.get_current_task]
        return math.isclose(
            float(model_answer), correct_answer, rel_tol=1e-5, abs_tol=1e-8
        )

    def update_current_task(self):
        """
        Sets is_solved for the current task to True and increments self.current_task_number by one
        """
        self.is_solved[self.get_current_task] = True 
        self.current_task_number = (self.current_task_number + 1) % len(self.operations)
        

# tests.ArithmeticTaskTests(ArithmeticTask)

x = ArithmeticTask(10, 15)
for problem, answer in x.correct_answers.items():
    print(f"{problem} = {answer}")

for i in range(len(x.operations)):
    x.current_task_number = i
    print(x.instruction)

print(x.check_solved())
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
        result = str(evaluate_expression(expression))
        return result

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
                "name": self.name,
                "description": 'Calculates the result of an arithmetic expression. For example, you could provide an input in the form "2+3" and the function would return 5. Or you could provide an expression like "10/3" and the function would return 3.3333333333333335.',
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The arithmetic expression that you want to be evaluated.",
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

print(response.choices[0].message.content)
print(response.choices[0].message.tool_calls)
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
        "tool_call_id": getattr(tool_call, 'id', None),
        "name": tool_call.function.name
    }
# %%
messages = [{"role": "user", "content": "Calculate 5/3. Be precise."}]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[Calculator.description],
    tool_choice="auto",
)

message = response.choices[0].message

messages.extend([
    message,
    apply_tool_call_format(
        message.tool_calls[0],
        Calculator.execute(
            json.loads(
                message.tool_calls[0].function.arguments
            )['expression']
        )
    )
])
# %%
response_to_tool_calls = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[Calculator.description],
    tool_choice="auto",
)
print(response_to_tool_calls.choices[0].message.content)
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.chat_history,
            tools = [tool.description for tool in self.tools] if use_tool else None,
            tool_choice="auto" if use_tool else None,
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
        if message.tool_calls is not None:
            for tool_call in message.tool_calls:
                tool = next((t for t in self.tools if t.name == tool_call.function.name), None)
                if tool:
                    arguments = json.loads(tool_call.function.arguments)
                    tool_response = tool.execute(**arguments, task=self.task)
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
        # tool_response = self.execute_tool_calls
        return response

tests.test_execute_tool_calls(SimpleAgent, CalculateTool, ArithmeticTask)
# %%

my_simple_agent = SimpleAgent(ArithmeticTask(10, 15), tools=[Calculator])
response = my_simple_agent.run()

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
        tools: Optional[List[Any]] = None,
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

        # Append response to chat history
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
                    print(f"\nTool call: {tool_call.function.name}, ARGS: {tool_call.function.arguments}")
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
        - Then append to chat history (and return) "Correct" if the answer is correct and "Incorrect" if the answer is incorrect.

        Args:
            None

        Returns:
            str: "Correct" or "Incorrect"
        """

        # Get the final response from the model after tool responses

        response = self.get_response(use_tool=False)
        self.chat_history.append(apply_assistant_format(response.content))

        # Check the answer
        try:
            model_answer = self.parse_answer(response)

            if self.task.check_answer(model_answer):
                self.chat_history.append(apply_user_format("Correct."))

                if self.verbose:
                    print("\nUser: Correct.")

                # Update to the next task
                self.task.update_current_task()

                return "Correct"

            else:
                self.chat_history.append(apply_user_format("Incorrect."))
                if self.verbose:
                    print("\nUser: Incorrect.")
                return "Incorrect"
                # Retry the task

        # Ends the task if there's an error parsing the model answer
        except Exception as e:
            if self.verbose:
                print(f"\nError parsing model answer: {e}")
            raise

    def run(self, with_tool: bool):
        """
        Run one loop of the agent, which involves:
        - getting a task
        - getting a response from the model
        - handling the model response, including tool calls, refusals, no tool calls, parsing and checking final answers, errors.
        - managing memory: storing the history of messages to self.chat_history
        - managing task state: staying on the same task or moving to the next task at the end of the loop
        """
        # Get the task instruction
        instruction = self.task.instruction
        if self.verbose:
            print("\nUSER:", instruction)
        self.chat_history.append(apply_user_format(instruction))

        # Get the response from the model
        response = self.get_response(use_tool=with_tool)

        if self.verbose:
            print("\nModel response:", response.content)

        # Handle the response
        ## If model makes tool calls, handle the tool calls
        if response.tool_calls:
            self.handle_tool_calls(response)

            # Then get the final answer from the model
            self.generate_and_check_final_answer()

        ## If no tool call: Handle edge cases

        ### Check if there's a refusal to answer:
        elif response.refusal:
            self.handle_refusal(response)

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
my_simple_agent = ArithmeticAgent(task=ArithmeticTask(10, 15), tools=[Calculator])
my_simple_agent.run(with_tool=True)

# %%

#Retrieve a Wikipedia page from its title
page = wikipedia.page("Large Language Model")

# Access basic page information
print("Title:", page.title)
print("\nURL", page.url)
print(f"\nSummary (word count {len( page.summary.split())}):", page.summary)
print(
    f"\nContent (word count {len( page.content.split())}):",
    page.content[:1000],
    "......",
)#Retrieve a Wikipedia page from its title
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
print(
    f"""\nLinks (link count {len(page.links)}): [{", ".join(page.links[:7])}, ......]"""
)
# %%
page = wikipedia.page("Python")
# %%
page = wikipedia.page("Animalss", auto_suggest=False)
# %%
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

    return [link for link in current_page.links if link.lower() in current_page.content.lower()]
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

        return {
            'role': 'system',
            'content': f"""You are an AI agent playing the Wikipedia game.
            Your goal is to get in as few steps as possible to the target Wikipedia
            page {self.goal_page} from the starting page {self.starting_page}.
            At each step, you choose a link from the current page to take you
            closer to the goal.."""
        }

    @property
    def on_page_instruction(self) -> dict:
        """
        Tell the agent what page they are on and give a summary of the page, formatted as a user prompt.

        Returns:
            dict: The instructions for the current page. The "role" is "user" for user messages.
        """
        return {
            'role': 'user',
            'content': f"""
                Current page title: 
                [START TITLE]
                {self.current_page.title}
                [END TITLE]\n
                Summary of current page: 
                [START SUMMARY]
                {self.current_page.summary}
                [END SUMMARY]
                \n\n
                Choose from the following links the page that will move you closest
                 to the goal page {self.goal_page.title}:
                [START LINKS]
                {self.get_permitted_links()}
                [END LINKS]
            """
        }

    @property
    def next_step_instruction(self) -> dict:
        """
        Ask the agent "What's the next step?" after making a tool call, formatted as a user prompt.

        Returns:
            dict: The instructions for the next step. The "role" is "user" for user messages.
        """
        # TODO
        return {
            'role': 'user',
            'content': """
                What is the next best step?
            """
        }

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

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": 'This tool allows you to get the content of the page, any link in the page will be wrapped in <link></link> tags.',
            },
        }


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
        new_page = new_page.replace("_"," ")
        if WikiGame.is_permitted_link(new_page):
            task.page_history.append(task.current_page.title)
            task.current_page = WikiGame.get_page(new_page)
            return f"Successfuly moved to page {task.current_page.title}"
        return f"Link provided {new_page} is not a valid link."

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
                "name": self.name,
                "description": 'Give a link as argument, and move to the page corresponding to this link.',
                "parameters": {
                    "type": "object",
                    "properties": {
                        "link": {
                            "type": "string",
                            "description": "The link you want to move to.",
                        }
                    },
                    "required": ["link"],
                    "additionalProperties": False,
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
        pass

    def reset_history(self):
        """
        Empty self.chat_history of the agent.
        """
        pass

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
        pass

    def handle_refusal(self, response: ChatCompletionMessage):
        """
        Handles refusals in the wikipedia game context:

        Args:
            response (ChatCompletionMessage): The response from the model
        """
        pass

    # ========================= Implementation logic (to implement) =========================
    def start(self):
        """
        A function to put the starting instructions in agent.chat_history when the agent starts a new page or starts the game.
        """
        pass

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
        pass