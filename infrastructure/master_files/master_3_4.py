# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
```python
[
    {"title": "Intro to LLM Agents", "icon": "1-circle-fill", "subtitle" : "5%"},
    {"title": "Simple Arithmetic Agent", "icon": "2-circle-fill", "subtitle" : "10%"},
    {"title": "More Complex Agent: WikiGame", "icon": "3-circle-fill", "subtitle" : "45%"},
    {"title": "Elicitation", "icon": "4-circle-fill", "subtitle" : "30%"},
    {"title": "Bonus", "icon": "5-circle-fill", "subtitle" : "10%"}
]
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# [3.4] - LLM Agents
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<img src = "https://raw.githubusercontent.com/info-arena/ARENA_img/refs/heads/main/img/ch3-evals-cover.jpeg" width = "600">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# Introduction
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Over the next two days, we'll be working with LLM agents. These consist of a scaffolding programs interacting with an LLM API. We'll also build two tasks, a simple and a complex one, in order to see how LLM agents act.

We'll begin by learning building a simple Arithmetic Task and Arithmetic Agent. This should teach you the basics of function calling via the OpenAI API (Anthropic's API has minor differences, but operates in essentially the same way). Then, once we're comfortable with function calling and the general setup of LLM agents and tasks, we'll move on to building a more complex agent that plays the [Wikipedia Game](https://en.wikipedia.org/wiki/Wikipedia:Wiki_Game).

Then we'll explore a variety of elicitation methods. These are methods for getting the best capabilities out of models, and is crucial for evaluating LLM agents. Elicitation tries to answer the question "Can the model do this?" Unfortunately, we'll almost never be able to prove that the model doesn't have a capability, and will only be able to say that with some effort, we couldn't get the model show this capability. This means we'll have to put a lot of effort into trying to exhibit the behavior in models (to have the highest confidence when we make a claim that the model can't exhibit this behavior). This will involve:

* Improving our prompting
* Improving our tools
* Improving the way the histories are stored
* Ensuring the model can access good information.*
Each exercise will have a difficulty and importance rating out of 5, as well as an estimated maximum time you should spend on these exercises and sometimes a short annotation. You should interpret the ratings & time estimates relatively (e.g. if you find yourself spending about 50% longer on the exercises than the time estimates, adjust accordingly). Please do skip exercises / look at solutions if you don't feel like they're important enough to be worth doing, and you'd rather get to the good stuff!
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Content & Learning Objectives

### 1️⃣ Intro to LLM Agents
> ##### Learning Objectives
> - Understand why we want to evaluate LLM agents.
> - Read resources about LLM agent evaluations to understand the current state of the field.
> - Understand the common failure modes of LLM agents.

### 2️⃣ Building a Simple Arithmetic Agent
> ##### Learning Objectives
> - Understand that a LLM agent is just a "glorified for-loop" (of the scaffolding program interacting with the LLM API).
> - Learn how to use function calling to allow LLMs to use external tools.
> - Understand the main functionalities of an LLM agent.

### 3️⃣ Building a more Complex Agent: WikiGame
> ##### Learning Objectives
> - Get comfortable building a more complex task
> - Understand how to build a more complex agent
> - Observe the failure modes of a more complex agent

### 4️⃣ Elicitation
> ##### Learning Objectives
> - Understand the importance of elicitation in evaluating LLM agents
> - Understand the different methods of elicitation
> - Understand how to improve prompting, tools, history storage, and information access in LLM agents

### 5️⃣ Bonus
> ##### Learning Objectives
>
> - Implement additional tools for elicitation
> - Explore some of your own ideas for elicitation
> - Explore some of the current research in elicitation
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Setup
'''

# ! CELL TYPE: code
# ! FILTERS: [colab]
# ! TAGS: [master-comment]

# import os
# import sys
# from importlib.metadata import distributions
# from pathlib import Path
# import warnings
# IN_COLAB = "google.colab" in sys.modules

# chapter = "chapter3_llm_evals"
# repo = "ARENA_evals"
# branch = "main"

# # Install dependencies
# if "inspect_ai" not in [dist.metadata["Name"] for dist in distributions()]:
#     %pip install openai anthropic inspect_ai tabulate wikipedia dotenv

# # Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
# root = (
#     "/content"
#     if IN_COLAB
#     else "/root"
#     if repo not in os.getcwd()
#     else str(next(p for p in Path.cwd().parents if p.name == repo))
# )

# if Path(root).exists() and not Path(f"{root}/{chapter}").exists():
#     if not IN_COLAB:
#         !sudo apt-get install unzip
#         %pip install jupyter ipython --upgrade

#     if not os.path.exists(f"{root}/{chapter}"):
#         !wget -P {root} https://github.com/chloeli-15/ARENA_evals/archive/refs/heads/{branch}.zip
#         !unzip {root}/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d {root}
#         !mv {root}/{repo}-{branch}/{chapter} {root}/{chapter}
#         !rm {root}/{branch}.zip
#         !rmdir {root}/{repo}-{branch}

# if IN_COLAB:
#     from google.colab import userdata
#     try:
#         os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
#     except:
#         warnings.warn("You don't have an OPENAI_API_KEY variable set in the secrets tab of your google colab.")


# if f"{root}/{chapter}/exercises" not in sys.path:
#     sys.path.append(f"{root}/{chapter}/exercises")

# os.chdir(f"{root}/{chapter}/exercises")

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

import json
import os
import sys
from pathlib import Path

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
import openai
from dotenv import load_dotenv

#Make sure exercises are in the path
chapter = r"chapter3_llm_evals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part4_llm_agents").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)

from utils import import_json, save_json, retry_with_exponential_backoff, pretty_print_questions, load_jsonl, omit
from utils import countrylist
from utils import evaluate_expression, apply_user_format, apply_assistant_format, establish_client_anthropic, establish_client_OpenAI, retry_with_exponential_backoff
import part4_llm_agents.tests as tests

MAIN = __name__ == "__main__"

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

client = OpenAI()

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 1️⃣ Intro to LLM Agents
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## What is an LLM agent?

An LLM agent consists of a scaffolding program interacting with an LLM API. This typically involves a loop of the following steps:

1. The scaffolding program sends instructions to the LLM, typically containing information on the task goal, the actions available to the LLM, and relevant task information.
2. The LLM processes the input and outputs an action in text (e.g. "calls" the calculate() tool in text).
3. The scaffolding program executes the action and returns the outcome (e.g. it runs the calculate() function in the background and returns the output to the agent).
4. The LLM observes the results and decides the next action.
5. Repeating the cycle until the task is complete

The two basic components of scaffolding are:

* Tool calling: This allows LLMs to use tools by providing a text description of the tool. The LLM can choose to use this tool by "calling" it in its text output. If it uses a tool, the scaffolding will execute this tool on the LLM's behalf (e.g. by running a python function, sending request to an external API etc.) and return the result of this tool call to the agent.
* Prompting: This describes the task state to the LLM, assists it in its tool use, or instruct it to use chain-of-thought to give the LLM more "thinking time" etc.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/refs/heads/main/img/ch3-llm-agent.png" width="800">

Diagram based on METR's [*Evaluating Language-Model Agents on Realistic Autonomous Tasks*], Figure 2.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Why evaluate LLM agents?

There are at least two reasons for why we might want to evaluate LLM agents.

1. **Measuring the maximum capabilities of a model**

For estimating safety risks, we want to measure the **ceiling** of dangerous capabilities. LLMs on their own often fail in easy-to-fix ways, as you will see. For example:

- They often claim to be incapable of tasks that they can actually perform.
- They can easily get stuck in loops.
- They can give up and ask the user for help
- They can hallucinate facts, or even misunderstand their own prior reasoning and hallucinate a faulty conclusion.
- They can be limited by primitive tools.
- They can be overly or underly sensitive to information in their prompts.
- They can have bugs.

This means that when a model fails to accomplish a task, it may still have the raw capability to succeed, but just require simple fixes that will unlock this capability. We want to eliminate the possibility of large capability improvements from relatively little effort, because this means our evals would have underestimated the true capability and risks associated with a model. Therefore, we want to try hard to elicit their raw capabilities via scaffolding, so that we can evaluate LLMs at their *best*.


2. **Measuring the alignment of LLMs in agentic scenarios**

We do not know if our current alignment techniques (e.g. supervised fine-tuning, RLHF) for aligning LLM chatbots will still work when LLMs are acting as agents in more complex scenarios. It is possible that these methods will not generalize well to agentic scenarios, and we want to test this.

We know today that LLMs are being used as more than just chatbots. Since the release of ChatGPT, the use of LLMs as agentic systems has grown signifcantly. These agents started off rather disappointingly initially, when they were based on GPT-3.5. However as more powerful LLMs come out and AI companies ensure their LLMs are better at tool-use, these agents are improving rapidly.

<details><summary>Further resources on LLM agent evaluations</summary>

- [Evaluating Language-Model Agents on Realistic Autonomous Tasks](https://evals.alignment.org/Evaluating_LMAs_Realistic_Tasks.pdf) (Kinniment et al., ARC Evaluations Team (now METR), 2023)
- [Large Language Models can Strategically Deceive their Users when Put Under Pressure](https://arxiv.org/pdf/2311.07590) (Scheurer et al., Apollo Research, ICLR 2024)
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) (Lilian Weng, OpenAI Safety Team, 2023)
- [AXRP Episode 34 - AI Evaluations with Beth Barnes](https://www.alignmentforum.org/posts/vACr4DExfeRMaCoo7/axrp-episode-34-ai-evaluations-with-beth-barnes) (Daniel Filan, 2024)
-[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/pdf/2303.11366) (Shinn et al., 2023)
- [Answering Questions by Meta-Reasoning over Multiple Chains of Thought](https://arxiv.org/pdf/2304.13007) (Yoran et al., 2024)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/pdf/2302.04761) (Schick et al., META AI Research, 2023)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Function Calling Guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 2️⃣ Building a Simple Arithmetic Agent
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
In general, most LLM agents share these core components:

<img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/refs/heads/main/img/ch3-sec4-agent-overview.png" width="1000">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
1. **LLM API interface**: A basic function that makes API calls (e.g. `get_response()`). <!-- (IN AGENT)-->
2. **Actions/Tools**: A set of actions the agent can take. <!-- (MOSTLY IN TASK)-->
3. **Task State Management**: Keeping track of the current state of the task and any relevant context. <!-- (IN TASK MOSTLY)-->
4. **Memory**: A way to store and retrieve information from past interactions (i.e. chat history). The simplest implemention is usually to store the list of past chat messages in a `chat_history` class attribute. <!-- (IN AGENT)-->
5. **Observation Parser**: Functions to parse and interpret the results of actions and update the state. <!-- (IN TASK MOSTLY)-->
6. **Decision/Execution Logic**: The rules or algorithms used to choose actions based on the current state and LLM output. <!-- (KIND OF IN BETWEEN)-->
7. **Task-Specific Information**: Any additional information or functions specific to the task at hand. <!-- (IN TASK)-->

These components are implemented across the `Task`, `Agent`, and `Tool` classes. However, the specific breakdown of these components in our implementation is a design choice and can vary depending on the task. While some seem more natural (e.g. LLM API interface goes into `Agent`, task state management goes into task), others can vary (e.g. Tools can be functions within the Task or the Agent, as opposed to being separate classes; observation parsing could go either way). In general, we want to maximize separability and minimize interfaces/dependencies, so that we can easily swap out the agent for the same task, or vice versa.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Task

In an LLM agent eval, there will usually be a `Task` class that interacts with the `Agent`. In general, the `Task` will:

- Prepare and provide the task instruction (and necessary files, functions etc) to the agent
- Parse and score the agent's output
- Update the task state accordingly (e.g. proceeds onto the next step of the task, ends the task).
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Build a simple arithmetic task
> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵⚪⚪⚪
> 
> You should spend up to 20-25 minutes on this exercise.
> ```

We will build a toy task called `ArithmeticTask`. This task takes in two numbers and create a list of arithmetic calculation problems with these two numbers, using arithmetic operations defined in `operations`. It should have methods to do the following:

- Get the current problem (e.g. at the start this will be "Calculate `num1` + `num2`")
- Check if a given answer is correct
- Update the current problem if the model answer was correct, (or if the model refuses to answer the question).
- Check if all problems have been solved

**How to handle calculations?** We have implemented a helper function `evaluate_expression()` to evaluate the arithmetic expressions, which you should use in your implementation of `execute()`. `evaluate_expression()` takes an arithmetic expression as a string (e.g. "3+5") and returns the result as a string (e.g. "8.0").

<details><summary>Aside: Why not use Python's in-built <code>eval()</code> function?</summary>

Python's `eval()` function evaluates an arbitrary string expression, and so allows AI models to run arbitrary code. Unless you have set up a container or sandboxed environment, it is very bad to allow LLMs to run arbitrary code on your computer!

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

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
        # EXERCISE
        # raise NotImplementedError("You need to implement the _generate_answers method")
        # END EXERCISE
        # SOLUTION
        return {
            f"{self.num1} {op} {self.num2}": evaluate_expression(f"{self.num1} {op} {self.num2}")
            for op in self.operations
        }
        # END SOLUTION

    @property
    def get_current_task(self) -> str:
        """
        Gets the current task for the agent

        Returns:
            str: A string containing the current task
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the get_current_task property")
        # END EXERCISE
        # SOLUTION
        return f"{str(self.num1)} {self.operations[self.current_task_number]} {str(self.num2)}"
        # END SOLUTION
    @property
    def instruction(self) -> dict:
        """
        Gets a string containing instructions for the current task for the agent. (This will be fed to the agent as a user prompt)

        Returns:
            dict: A dictionary containing the instructions for the current task, formatted as a user prompt.
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the instruction property")
        # END EXERCISE
        # SOLUTION
        return apply_user_format(f"Calculate the result of the following expression: {str(self.num1)} {self.operations[self.current_task_number]} {str(self.num2)}. Give your final answer in the format: <answer>NUMBER</answer>, where NUMBER is a numerical value")
        # END SOLUTION

    def check_solved(self) -> bool:
        """
        Checks if all tasks have been solved

        Returns:
            bool: True if all tasks have been solved, False otherwise
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the check_solved method")
        # END EXERCISE
        # SOLUTION
        return all(self.is_solved.values())
        # END SOLUTION

    def check_answer(self, model_answer: str | float) -> bool:
        """
        Checks if the model's answer is correct

        Args:
            model_answer (str): The model's answer

        Returns:
            bool: True if the model's answer is correct, False otherwise
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the check_answer method")
        # END EXERCISE
        # SOLUTION
        correct_answer = self.correct_answers[self.get_current_task]
        return math.isclose(
            float(model_answer), correct_answer, rel_tol=1e-5, abs_tol=1e-8
        )
        # END SOLUTION

    def update_current_task(self):
        """
        Sets is_solved for the current task to True and increments self.current_task_number by one
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the update_current_task method")
        # END EXERCISE
        # SOLUTION
        self.is_solved[self.get_current_task] = True
        self.current_task_number = (self.current_task_number + 1) % len(self.operations)
        # END SOLUTION

if MAIN:
    tests.ArithmeticTaskTests(ArithmeticTask)

    x = ArithmeticTask(10, 15)
    for problem, answer in x.correct_answers.items():
        print(f"{problem} = {answer}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Aside - What is <code>@property</code>?</summary>

The `@property` decorator in python is used to define methods that behave like they were attributes.

1. It allows you to access a method as though it were an attribute, without parentheses.
2. It allows you to perform functions when calling attributes, e.g. adding validation or performing any necessary calculations (in our case incorporating class attributes which frequently change).

For example, if we defined a `Square` class as follows:

```python
class Square:
    def __init__(self, side_length):
        self.side_length = side_length

    @property
    def perimeter(self):
        return self.side_length*4
```

Then we could access `perimeter` as if it were an attribute:

```python 
s = Square(4)
print(s.perimeter) # Output: 16
```

Using `@property` in this case helps with:
1. Making the intent of the code clearer
2. Making it slightly easier to access these "properties" of the class

</details>

<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Tool use via function calling

The simplest way for LLMs to take actions is via function calling. **Function calling** is a built-in feature of LLM Chat APIs that allows models to use external "tools" (i.e. Python functions, APIs) by simply receiving and outputing text. This involves 5 simple steps:

1. Pick a function in your codebase that the model should be able to call
2. Describe your function in the syntax of the model's API so the model knows how to call it
3. Pass your function definitions as available “tools” to the model, along with the messages
4. Receive and handle the model response
5. Provide the function call result back to the model 

**This loop of prompting the LLM with tools, executing its actions, and returning the results forms the basis of all LLM agents.** It allows LLMs to perform complex tasks like playing a game or completing a coding project "autonomously".

We will implement each step of the loop below.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Write `CalculateTool`
> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> 
> You should spend up to 10-15 minutes on this exercise.
> ```

We will define a tool class for our simple `calculate()` function with the following structure (don't need to run the code cell below, just read):
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class Tool:
    name: str # The name of the tool that models will use to call it

    @staticmethod
    def execute(task: Any, input: str) -> str: 
        """Executes the tool and returns the result as a string"""
        ...

    @property
    def description(self) -> dict: 
        """Returns the tool description in the API syntax"""
        ...

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
For the `CalculateTool`, you should implement the following methods:
- `execute()` - This should take in an arithmetical expression as string (e.g. `"3+5"`) and return the result of this expression (also as a string). The `execute()` function should take the task as a variable, as often tools will need to be able to make changes to the task state (e.g. update the current problem).
- `description` - This should return a dictionary containing the tool description in the correct API syntax. 

#### Tool Description

Models like ChatGPT and Claude are fine-tuned to interpret and respond to `tool` descriptions appropriately, just like `user` and `system` messages. Below is an example of a typical tool description for the OpenAI API (see their [function calling guide](https://platform.openai.com/docs/guides/function-calling) for more details). Note that this may differ between APIs.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
```python
{
    "type": "function",
    "function": {
        {   
        "type": "function",
        "function":{
            "name": "get_delivery_date",
            "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The customer's order ID.",
                        },
                    },
                "required": ["order_id"],
                "additionalProperties": false,
                },
            },
        },
    },
}
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary><b>Good practices for writing tool descriptions</b></summary>

Here are some good practices for writing tool descriptions for Claude according to Anthropic, which should generalize to other chat models:
- Provide extremely detailed descriptions. This is by far the most important factor in tool performance. Your descriptions should explain every aspect of the tool, including:
    - What the tool does
    - When it should be used (and when it shouldn’t)
    - What each parameter means and how it affects the tool’s behavior
    - Any important caveats or limitations, such as what information the tool does not return if the tool name is unclear. The more context you can give Claude about your tools, the better it will be at deciding when and how to use them. Aim for at least 3-4 sentences per tool description, more if the tool is complex.
- Prioritize descriptions over examples. While you can include examples of how to use a tool in its description or in the accompanying prompt, this is less important than having a clear and comprehensive explanation of the tool’s purpose and parameters. Only add examples after you’ve fully fleshed out the description.

Read Anthropic's examples of what good and bad tool calling looks like [here](https://docs.anthropic.com/en/docs/build-with-claude/tool-use#example-of-a-good-tool-description). 

</details>

Write your tool class for the `CalculateTool` below.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class CalculateTool():
    """

    A tool that calculates the result of an arithmetic expression input as a string and returns as a string.

    Attributes:
        name (str): The name of the tool

    Methods:
        - execute(task: Optional[], input: str) -> str: Executes the tool on the input and returns the result as a string.
        - description() -> str: Returns a description of the tool.

    """
    name = "calculate"

    @staticmethod
    def execute(expression: str, task: Optional[ArithmeticTask] = None) -> str:
        """
        Evaluates the string expression in Python using `evaluate_expression()` and returns the result as a string

        Args:
            expression (str): The arithmetic expression to evaluate
            task (ArithmeticTask | None): Not used in this function

        Returns:
            str: The result of the arithmetical expression as a string
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the execute method")
        # END EXERCISE
        # SOLUTION
        try:
            return str(evaluate_expression(expression))
        except (SyntaxError, NameError, ZeroDivisionError) as e:
            return f"Error: {str(e)}"
        # END SOLUTION
    @property
    def description(self):
        """
        Provides the description of the tool

        Returns:
            dict: The JSON description of the tool for the OpenAI API
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the description property")
        # END EXERCISE
        # SOLUTION
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
        # END SOLUTION

if MAIN:
    tests.run_calculate_tool_tests(CalculateTool)

Calculator = CalculateTool()

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
You can pass the tool to the model by providing the tool description to the `tools` parameter of the API call. This input has to be a list. The following code provides a standard example of this.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Why is <code>message.content = None</code>?</summary>

When LLMs use tools, they often don't generate any text output. This can be a problem later when you try to get the model to do chain-of-thought reasoning. To get around this, it can be better to make two calls to the model for more complex tool use: one call to get the model to reason about the actions it should take, and then another to get the model to use a tool to take those actions.

</details> 

<details><summary> Aside - What is <code>@staticmethod</code>?</summary>

The `@staticmethod` decorator in Python defines a "static method" within a class, which has the following properties:
1. They don't use instance- or class-specific data, thus does not require a first parameter `self` or `cls`.
2. They're often used as utility functions related to the class.

For example, if we defined a class of `MathOperations` as follows:

```python
class MathOperations:
    @staticmethod
    def add(x : int | float, y : int | float) -> int | float:
        """Evaluates the string expression and returns the result as a string."""
        return x + y
```

The `add()` method could be called on the class itself without creating an instance:

   ```python
   result = MathOperations.add(2, 3)
   ```

You can also call it on an instance of the class, but it doesn't utilize the instance in any way (it doesn't have access to `self`):
   ```python
   operation = MathOperations()
   result = operation.add(2, 3)
   ```

Typically, you would make "stand-alone" functions that do not depend on class methods or class/instance attributes a static method. Using `@staticmethod` in this case helps with the following:
1. Makes the code's intent clearer (this method doesn't need class or instance data).
2. Slightly improves performance (no `self` argument needs to be passed).
3. Allows the method to be used without creating an instance of the class.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Return tool call results
> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> ```

The model always gives its output as a `ChatCompletionMessage` object. If this contains a tool call, you will need to add **two** items to the `messages` list to return back to the model:
1. The `ChatCompletionMessage` object itself, containing the original tool call message generated by the model. 
2. The **tool response** message, describing the results of the tool call in a specific [format](https://platform.openai.com/docs/guides/function-calling/step-4-receive-and-handle-the-model-response). Similar to the system and user message, the tool response message is a dictionary with `role`, `content`, and additionally `tool_call_id` and `name` keys. (Note: This is **not** the same as the tool description we wrote above). 

If we tried to generate the next response without returning these after the model's tool call, the API will raise an error.

<details><summary>How to access the <code>ChatCompletionMessage</code> object</summary>

The `chat.completions.create()` function returns a `response` with a complicated data structure. The `ChatCompletionMessage` is nested inside and can be accessed via `response.choices[0].message` (don't need to run the code below, just read).

```python
# This is an example of the `response.choices[0]` object. `ChatCompletionMessage` is in the `message` parameter.
Choice(
    finish_reason="tool_calls",
    index=0,
    logprobs=None,
    message=chat.completionsMessage(
        content=None,
        role="assistant",
        function_call=None,
        tool_calls=[
            chat.completionsMessageToolCall(
                id="call_62136354",
                function=Function(arguments='{"expression":"2+3"}', name="calculate"),
                type="function",
            )
        ],
    ),
)
```
</details>


We have provided a function that formats the tool response correctly for the OpenAI API.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Now use the function above to return a tool call response to the model after the following message:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN: 
    messages = [{"role": "user", "content": "Calculate 5/3. Be precise."}]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=[Calculator.description],
        tool_choice="auto",
    )
    # EXERCISE
    # # TODO: Return the tool call responses back to the model
    # END EXERCISE
    # SOLUTION
    messages.extend(
        [
            response.choices[0].message,
            apply_tool_call_format(
                response.choices[0].message.tool_calls[0],
                Calculator.execute(
                    json.loads(
                        response.choices[0].message.tool_calls[0].function.arguments
                    )["expression"]
                ),
            ),
        ]
    )

    response_to_tool_calls = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=[Calculator.description],
        tool_choice="auto",
    )
    print(response_to_tool_calls.choices[0].message.content)
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Agent

We will first implement a `SimpleAgent` class that is not specific to the `ArithmeticTask`, so that we can see the key components of a generic LLM agent.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Implement `SimpleAgent`
> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
> 
> You should spend up to 20-25 minutes on this exercise.
> ```

Build out the following simple agent class by filling in `get_response()` and `execute_tool_calls()` functions.

- `get_response()`: This should make an API call and return the `ChatCompletionMessage`from the model. It should be able to either use tool calling or not, depending on the `use_tool` argument).
- `execute_tool_calls()`: This should execute the tool calls in the message and return a list of tool responses as strings (we can format them correctly in `run()`).
- `run()`: This should define the main execution logic for running 1 loop of the agent. As this is largely determined by the task, this method in `SimpleAgent` is just a dummy method and should be overridden in specific agent classes.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

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
        # EXERCISE
        # raise NotImplementedError("You need to implement the get_response method")
        # END EXERCISE
        # SOLUTION
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.chat_history,
            tools=[tool.description for tool in self.tools] if use_tool else None,
            tool_choice="auto" if use_tool else None,
        )
        return response.choices[0].message
        # END SOLUTION
    def execute_tool_calls(self, message: ChatCompletionMessage) -> List[str]:
        """
        Execute the tool calls in the message and return a list of tool_responses.

        Args:
            message (ChatCompletionMessage): The message containing the tool calls

        Returns:
            List[str]: A list of tool responses (as strings)
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the execute_tool_calls method")
        # END EXERCISE
        # SOLUTION
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
        # END SOLUTION
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
        self.chat_history.append(instruction)
        response = self.get_response(use_tool=with_tool)
        return response

if MAIN:
    tests.test_execute_tool_calls(SimpleAgent, CalculateTool, ArithmeticTask)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Then running the agent should cause the tool calls to be executed.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    my_simple_agent = SimpleAgent(task=ArithmeticTask(10, 15), tools=[Calculator])
    my_simple_agent.run()

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Build an `ArithmeticAgent`

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
> 
> You should spend up to 20-25 minutes on this exercise.
> ```

Now build our agent that will interact with the `ArithmeticTask` (with a calculator tool). Fill in the methods in the class below.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class ArithmeticAgent(SimpleAgent):
    """
    ArithmeticAgent class for doing simple arithmetic tasks.

    Inherits from SimpleAgent which includes the following attributes and methods:

    Attributes:
        model (str): The model used for generating responses (inherited)
        tool_descriptions (List[dict]): List of tool descriptions (inherited)
        client (OpenAI): OpenAI client for API calls (inherited)
        task (ArithmeticTask): The current task being executed (inherited)
        chat_history (List[dict]): History of interactions (inherited)

    Methods:
        get_response(use_tool: bool = True) -> ChatCompletionMessage:
            Get response from the model (inherited)

        execute_tool_calls(message: ChatCompletionMessage) -> List[str]:
            Execute tool calls from the model's response (inherited)

        run(with_tool: bool = True) -> bool:
            Run one loop of the Arithmetic agent
    """

    def __init__(
        self,
        model: Literal["gpt-4o-mini"] = "gpt-4o-mini",
        task: ArithmeticTask = None,
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
        # EXERCISE
        # raise NotImplementedError("You need to implement the handle_tool_calls method")
        # END EXERCISE
        # SOLUTION
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
        # END SOLUTION

    def handle_refusal(self, response: ChatCompletionMessage):
        """
        Handle the refusal from the model response. This function should only be called if the model refuses to answer and should:
        - Append the refusal to the chat history
        - Update the task state

        Args:
            response (ChatCompletionMessage): The response from the model
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the handle_refusal method")
        # END EXERCISE
        # SOLUTION
        if self.verbose:
            print("\nModel Refusal:", response.refusal)
        self.chat_history.append(apply_assistant_format(response.refusal))
        self.task.update_current_task()
        # END SOLUTION

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
        # EXERCISE
        # raise NotImplementedError("You need to implement the generate_and_check_final_answer method")
        # END EXERCISE
        # SOLUTION
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
            raise e
        # END SOLUTION

    def run(self, with_tool: bool):
        """
        Run one loop of the agent, which involves:
        - getting a task
        - getting a response from the model
        - handling the model response, including tool calls, refusals, no tool calls, parsing and checking final answers, errors.
        - managing memory: storing the history of messages to self.chat_history
        - managing task state: staying on the same task or moving to the next task at the end of the loop
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the run method")
        # END EXERCISE
        # SOLUTION
        # Get the task instruction
        instruction = self.task.instruction
        if self.verbose:
            print("\nUSER:", instruction["content"])
        self.chat_history.append(instruction)

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
        # END SOLUTION

    def parse_answer(self, message: ChatCompletionMessage) -> float:
        """
        Extract the numerical answer from the string output of the model

        Args:
            message (ChatCompletionMessage): The response from the model

        Returns:
            float: The numerical answer extracted from the model
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the parse_answer method")
        # END EXERCISE
        # SOLUTION
        response = message.content
        if response.find("<answer>") != -1:
            startpoint = response.find("<answer>") + 8
            endpoint = response.find("</answer>")
            return float(response[startpoint:endpoint])
        else:
            raise ValueError('"<answer>" not found in model response.')
        # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Note on <code>handle_refusal()</code></summary>

The `ChatCompletionMessage` object contains a `refusal` attribute that can be used to determine if the model has refused to answer. If the model has refused to answer, the `refusal` will contain this content and we can print this out. We have included this for completeness, but it is not necessary to implement this function because it almost never occurs in the Arithmetic Task.

See the [OpenAI API documentation](https://platform.openai.com/docs/guides/function-calling/edge-cases) for more information on the `refusal` attribute.

</details>

<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Run the task via an agent_loop

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> 
> You should spend up to 5-10 minutes on this exercise.
> ```

Try implementing the agent_loop below with and without tools, to see how much better the model does when we give it tools.

> **WARNING!** 
>
>When you're making API calls to LLMs to tasks, it can be tempting to use a while loop, and run the model until it finishes the task. But since every time we run a model we make an API call, this would allow us to spend arbitrarily large amounts of money on API calls. For this reason, ***always use a for loop when making API calls!!!*** It would be really unfortunate if you blew all your API budget on one mistake.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def agent_loop(agent, num_loops: int = 10):
    """
    Run the agent loop for a given number of loops

    Args:
        agent (ArithmeticAgent): The agent to run
        task (ArithmeticTask): The task to solve
        num_loops (int): The number of loops to run
    """
    # EXERCISE
    # raise NotImplementedError("You need to implement the agent_loop function")
    # END EXERCISE
    # SOLUTION
    for i in range(num_loops):
        if not agent.task.check_solved():
            agent.run(with_tool=False)
        else:
            print("\nAll tasks solved.")
            break
    # END SOLUTION

if MAIN: 
    arithmetic_task_1 = ArithmeticTask(31.1, 8)
    arithmetic_agent_1 = ArithmeticAgent(
        task=arithmetic_task_1, verbose=True, tools=[Calculator]
    )


    agent_loop(arithmetic_agent_1)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
If we want to see how the model performed at the task, then we can print all the messages from the `ChatHistory` as follows:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

for message in arithmetic_agent_1.chat_history:
    try:
        print(f"""{str(message.role)}:\n {str(message.content)}\n""")
    except:
        print(f""" {message["role"]}:\n {message["content"]}\n""")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 3️⃣ Building a more complex agent: WikiGame
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Quick Intro to the Wikipedia API

Our agent will interact with Wikipedia by making tool calls to the [Wikipedia API](https://wikipedia.readthedocs.io/en/latest/quickstart.html), which is simple to use. We will only need to learn the following key functions for the game. 

1. `wikipedia.page()` - Returns a `WikipediaPage` object, which contains various attributes and methods to access page content. (See [page docs](https://wikipedia-api.readthedocs.io/en/latest/API.html#wikipediapage) for these attributes.)
2. `wikipediaPage.title` - Returns the title of the page
3. `wikipediaPage.content` - Returns the full text content of the page (this can be very long, make sure to take snippets when possible to not use up the context window of the LLM)
4. `wikipediaPage.summary` - Returns a summary of the page (i.e. the introductory text of the Wikipage before the first section title).
5. `wikipediaPage.links` - Returns a list of all links as strings


<details><summary> Aside: Wikipedia API content can be weird!</summary>

The wikipedia API often outputs content in unintuitive ways. For example, articles that are essentially just a big list become near useless, since the content omits the list (for example, see the wikipedia API content for <a href = "https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population">List of countries and dependencies by population</a>). Another issue that you might encounter is that the API formats mathematical expressions in $\LaTeX$ pretty poorly (for example, see the wikipedia API content for <a href = "https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">Kullback-Leibler divergence</a>). This is why it's important to determine what content the wikipedia API produces when `.content` is called — and why you want to make sure you're testing a large diversity of wikipedia articles.

</details>

<details><summary> Aside: Wikipedia "summaries" can be long!</summary>

The wikipedia API accesses summaries of pages by presenting all the information before the first titled section. For certain (generally obscure) wikipedia pages, this summary itself can be extremely long, and contain lots of information that is unnecessary to determine the key information about the page the model should be trying to access. We'll handle this later when it comes up by truncating wikipedia's summary to just the first ~1000 characters

</details>

Run the following code to see how these wikipedia API functions work!
'''

# ! CELL TYPE: code
# ! FILTERS: [~py]
# ! TAGS: []

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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Now run these two lines (you should see a `DisambiguationError` for the first, and a `PageError` for the second):
'''

# ! CELL TYPE: code
# ! FILTERS: [~py]
# ! TAGS: []

page = wikipedia.page("Python")

# ! CELL TYPE: code
# ! FILTERS: [~py]
# ! TAGS: []

page = wikipedia.page("Animalss", auto_suggest=False)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
We can handle these errors using the following code:
'''

# ! CELL TYPE: code
# ! FILTERS: [~py]
# ! TAGS: []

# Fixes PageError by allowing redirects
page = wikipedia.page("Animalss", redirect=True)
print(page.title)

# Fixes DisambiguationError by selecting the first option

try:
    page = wikipedia.page("Python")
except DisambiguationError as e:
    page = wikipedia.page(e.options[0])
print(page.title)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
- `DisambiguationError`: This was raised because the title "Python" can correspond to multiple pages. 
- `PageError`: This was raised for "Animalss" as there is no Wikipedia page with that title.

We have implemented a simple function `get_page()` for you to get the page object for a particular page title with error handling.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>What do the kwargs <code>redirect</code> and <code>auto_suggest</code> in <code>wikipedia.page()</code> do?</summary>

`redirect`

- This kwarg enables redirecting when you reference an article title with **slight** differences to how it is stored in Wikipedia. For example, the Wikipedia API will generally access the correct page if there is a capitalization error on the first letter, but not for capitalization errors in the middle of the word if `redirect = False`:
```python
# This returns a WikipediaPage object for the "Human" page
page = wikipedia.page("huMan", redirect = True, auto_suggest=False)

# This raises a PageError since there is no page called "huMan"
page = wikipedia.page("huMan", redirect=False, auto_suggest=False)
```
- By default, we should set `redirect = True` in the `wikipedia.page()` function.

`auto_suggest`

- This kwarg enables the API to provide suggestions. This allows a lot more than `redirect` does, since `redirect` is only for the "obvious" cases (e.g. "huMan" → "Human", "U.S. President" → "President of the United States", etc.). When `auto_suggest` is true, it would allow something like "president of states" → "President of the United States", "gogle" → "Google"; both of which would raise an error if `redirect = True, auto_suggest = False`.
- However, `auto_suggest` can sometimes be *too* permissive and lead to errors. For example, the below code will return a `WikipediaPage` object for the "Man" page. This is clearly not what we were trying to access, and the `auto_suggest` has gotten carried away in this case:

```python
page = wikipedia.page("Human", redirect= False, auto_suggest=True)
```

- If `redirect = True` and `auto_suggest=True`, then `auto_suggest` takes priority. 
- **By default, we should set `auto_suggest` to `False` unless it is used as a last resort to resolve an error!**

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Implement `get_permitted_links()`
> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵⚪⚪⚪
> 
> You should spend up to ~10 mins on this exercise.
> ```

This is a quick exercise to familarize you with the Wikipedia API.

When you get the links from a page using `page.links`, this will include every possible Wikipedia link that is accessible from the HTML on that page, including those that are not in the main page content (e.g. links in sidebars, links in footnotes etc.), which are irrelevant or not permitted by the rules of the Wiki game. 

Write a simple `get_permitted_links()` function. This should only return the links that can be found inside the main content. The resulting list of permitted links should be about a third as long as the list of links from `page.links`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def get_permitted_links(current_page: WikipediaPage) -> list[str]:
    """
    Get "permitted" links (i.e. links that are in the content of the page) from a Wikipedia page.

    Args:
        current_page (WikipediaPage): The current Wikipedia page

    Returns:
        list[str]: A list of permitted links from current_page

    """
    # EXERCISE
    # raise NotImplementedError("You need to implement the get_permitted_links function")
    # END EXERCISE
    # SOLUTION
    all_links = current_page.links
    content_lower = current_page.content.lower()
    permitted_links = [link for link in all_links if link.lower() in content_lower]
    return permitted_links
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## LLM Agent for WikiGame

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/refs/heads/main/img/ch3-wiki-task-overview.png" width="1000">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Build the WikiGame task
> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵⚪
> 
> You should spend up to 10-15 mins on this exercise.
> ```

Build the `WikiGame` class that instantiates the wikipedia game. This should contain the following functionalities:
1. Keep track of task states
2. Give task-specific instructions
3. Task-specific helper functions for calling the Wikipedia API. These less interesting methods have been provided for you, but you should read and understand what they do.

#### Providing information to the agent

While models are trained on most of the Wikipedia content, a particular page may still be confused with something else, or be an article that was added after the training cutoff. Models also can't always accurately recall information in their training data if they only come up once or twice. So you should use the game's `get_summary()` function to provide details of the goal page to the agent in its initial message.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

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
        Generate the starting instructions for the game, formatted as a system prompt.

        Returns:
            dict: The starting instructions.
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the system_instruction property")
        # END EXERCISE
        # SOLUTION
        return apply_system_format("You are a wikipedia-racing AI. Your aim is to reach the goal page by accessing links from a series of wikipedia pages.")
        # END SOLUTION
    @property
    def on_page_instruction(self) -> dict:
        """
        Tell the agent what page they are on and give a summary of the page, formatted as a user prompt.

        Returns:
            dict: The instructions for the current page.
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the on_page_instruction property")
        # END EXERCISE
        # SOLUTION
        return apply_user_format(f"""You are currently on page: {self.current_page.title}. Your goal page is {self.goal_page.title}.""")
        # END SOLUTION
    @property
    def next_step_instruction(self) -> dict:
        """
        Ask the agent "What's the next step?" after making a tool call, formatted as a user prompt.

        Returns:
            dict: The instructions for the next step.
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the next_step_instruction property")
        # END EXERCISE
        # SOLUTION
        return apply_user_format(f"What's your next step?")
        # END SOLUTION

    def check_win(self) -> bool:
        # EXERCISE
        # raise NotImplementedError("You need to implement the check_win method
        # END EXERCISE
        # SOLUTION
        return self.current_page == self.goal_page
        # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Build tools for the WikiGame
> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
> 
> You should spend up to 15-20 mins on this exercise.
> ```

The basic WikiAgent will need these two tools at minimum to play the game:
1. `GetContentTool`: This returns the full content of the current page, with all the wiki-links wrapped in `<link></link>` tags (as otherwise they are presented as strings and indistinguishable from normal text). As implementing this involves annoying regex, we have done this for you, but you should fill in the `description()` property.
2. `MovePageTool`: This executes moving to a new given page when called and updates the `WikiGame` task state if successful. You should implement both the `execute()` function and the `description()` property.

When formatting this tool list, refer back to your code for the arithmetic game, or the OpenAI function-calling docs [here](https://platform.openai.com/docs/guides/function-calling).

<details><summary>Why not just use <code>WikipediaPage.links()</code> to get a list of links directly?</summary>

We don't just present a list of the accessible links, as this is not very faithful to the wikipedia game. The agent does perform somewhat better if we just give it a list of links, but the task of parsing the content of wikipedia pages and isolating the most important links is big part of the challenge of the wikipedia game.

</details>

<details><summary>Caveat for the <code>GetContentTool</code></summary>

The `GetContentTool` wraps all the texts that correspond to links in `<link></link>` tags. However, since we identify links in the text via their names on wikipedia pages, there are certain articles that will never (or only very rarely) get flagged as links. For example, the page "Python (programming language)" is almost never referenced by its title, instead its almost always referenced by just "Python"; the same is true for towns, which are usually referenced on Wikipedia as e.g. "Juneau, Alaska", but these are almost always referred to as just "Juneau" in the articles where they appear. For this reason, you should avoid having goal pages which are not referenced by their title (or else implement a better version of the function, but beware of simply extracting the HTML source from pages, `wikipediaPage.html` can take a very long time to run, and HTML formatting can vary significantly across Wikipedia).

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class GetContentTool():
    name = "get_content"

    @staticmethod
    def execute(task: WikiGame) -> str:
        """
        Get all the content for the wikipedia page you are currently on. Anything which corresponds to a link is wrapped in <link></link> tags.

        Args:
            task (WikiGame): The current task object.

        Returns:
            str: The content of the page with links wrapped
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the execute method for the GetContentTool")
        # END EXERCISE
        # SOLUTION
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
        # END SOLUTION

    @property
    def description(self):
        """
        Provides the description of the getContent tool

        Returns:
            dict: The description of the tool for the API
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the description property for the GetContentTool")
        # END EXERCISE
        # SOLUTION
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Get all the content for the wikipedia page you are currently on. Anything which corresponds to a link you can select to move to will be wrapped in <link></link> tags.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
        # END SOLUTION


class MovePageTool():
    name = "move_page"

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
        # EXERCISE
        # raise NotImplementedError("You need to implement the execute method for the MovePageTool")
        # END EXERCISE
        # SOLUTION
        new_page_normalized = new_page.replace("_", " ")
        if task.is_permitted_link(new_page_normalized):
            task.current_page = task.get_page(new_page_normalized)
            task.page_history.append(task.current_page.title)
            return f"Moving page to {task.current_page.title}"
        else:
            return f"Couldn't move page to {new_page}. This is not a valid link."
        # END SOLUTION

    @property
    def description(self):
        """
        Provides the description of the move_page tool

        Returns:
            dict: The description of the move_page tool for the API
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the description property for the MovePageTool")
        # END EXERCISE
        # SOLUTION
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Changes your current page to a specified new page which is accessible via a link from the current page. You can only call this function once at a time, as it will take you to a different page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_page": {
                            "type": "string",
                            "description": 'The title of the new page you want to move to. This should be formatted the way the title appears on wikipedia (e.g. to move to the wikipedia page for the United States of America, you should enter "United States"). Underscores are not necessary.',
                        }
                    },
                    "required": ["new_page"],
                },
            },
        }  
        # END SOLUTION

GetContentTool_inst = GetContentTool()
MovePageTool_inst = MovePageTool()
wiki_game_tools = [GetContentTool_inst, MovePageTool_inst]

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Build a WikiAgent
> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
> 
> You should spend up to 30-60 mins on this exercise.
> ```

We will now build a `WikiAgent` that can use these tools to solve the `WikiGame`. Build the agent so that it can be called via an agent loop, similar to the one we had for the arithmetic game. 

There are a few further considerations in this case that we didn't have for the arithmetic game. 

#### Context window constraint

Since Wikipedia articles could be very long, the length of the LLM's context window becomes a constraint. GPT-4o and GPT-4o-mini both have context windows of 128k tokens (which corresponds to ~96k words). For reference, the wikipedia page for the United States has around 10k words alone and the agent will often need to visit more than 10 articles in one run of the game, not counting its own output, which eventually adds up to be significant. 

We'll solve this for now by simply resetting the messages of the agent every time it reaches a new wikipedia page, and providing an updated set of instructions, so the agent can locate itself in the game. We'll address different methods for solving this issue later, you can probably already think of some. So be careful to include the current page and goal page for the agent in the instruction.

Since we'll reset the `chat_history` attribute of the agent class each time it reaches a new page, we'll also store a `full_chat_history` property that won't get reset, so we can access the entire run of the game.


#### Printing output

The `WikiGame` is a lot longer than the `ArithmeticTask`, with a much larger volume of agent, task and tool messages. If there's some chance you might not want to see this output, you should use the `verbose` parameter to set whether to print the output or not.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class WikiAgent(SimpleAgent):
    """
    Inherits from SimpleAgent and adds the ability to handle tool calls and refusals in the Wikipedia game context.

    Attributes:
        model (str): The model used for generating responses (inherited)
        tools (List[Any]): List of tools (inherited)
        client (OpenAI): OpenAI client for API calls (inherited)
        task (WikiGame): The current task being executed
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
        task: WikiGame,
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
        self, message: dict[str,str] | ChatCompletionMessage | List[dict[str,str] | ChatCompletionMessage]
    ):
        """
        Update self.chat_history and self.full_chat_history with a message or list of messages.

        Args:
            message (dict[str, str] | ChatCompletionMessage | List[dict[str,str] | ChatCompletionMessage]): The message to add to the chat history
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the update_history method")
        # END EXERCISE
        # SOLUTION
        if isinstance(message, list):
            self.chat_history.extend(message)
            self.full_chat_history.extend(message)
        else:
            self.chat_history.append(message)
            self.full_chat_history.append(message)
        # END SOLUTION
    def reset_history(self):
        """
        Empty self.chat_history of the agent.
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the reset_history method")
        # END EXERCISE
        # SOLUTION
        self.chat_history = []
        # END SOLUTION

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
        # EXERCISE
        # raise NotImplementedError("You need to implement the handle_tool_calls method")
        # END EXERCISE
        # SOLUTION
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
        # END SOLUTION

    def handle_refusal(self, response: ChatCompletionMessage):
        """
        Handles refusals in the wikipedia game context:

        Args:
            response (ChatCompletionMessage): The response from the model
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the handle_refusal method")
        # END EXERCISE
        # SOLUTION
        self.update_history(apply_assistant_format(response.refusal))
        if self.verbose:
            print(f"\nMODEL REFUSAL: {response.refusal}")
        # END SOLUTION

    def start(self):
        """
        A function to put the starting instructions in agent.chat_history when the agent starts a new page or starts the game.
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the start method")
        # END EXERCISE
        # SOLUTION
        instruction_messages = [
            self.task.system_instruction,
            self.task.on_page_instruction,
        ]
        self.update_history(instruction_messages)
        if self.verbose:
            print(
                f"\nSYSTEM: \n{instruction_messages[0]['content']} \n\nUSER: \n{instruction_messages[1]['content']}"
            )
        # END SOLUTION

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
        # EXERCISE
        # raise NotImplementedError("You need to implement the run method")
        # END EXERCISE
        # SOLUTION
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
        # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Run the task
> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵⚪⚪⚪
> 
> You should spend up to 10-15 mins on this exercise.
> ```

Similar to the `ArithmeticAgent`, write an agent loop for the `WikiAgent`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def agent_loop(agent, num_loops=10):
    """
    Run the agent loop for a given number of loops

    Args:
        agent (WikiAgent): The agent to run
        game (WikiGame): The game to play
        num_loops (int): The number of loops to run
    """
    # EXERCISE
    # raise NotImplementedError("You need to implement the agent_loop function")
    # END EXERCISE
    # SOLUTION
    for i in range(num_loops):
        if agent.task.check_win():
            print("Success!")
            return
        agent.run()
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Your agent should be able to accomplish the following tasks:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    game_1 = WikiGame("Barack Obama", "India")
    agent = WikiAgent(task=game_1, tools=wiki_game_tools)
    agent_loop(agent, game_1, 30)

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    game_2 = WikiGame("Albert Einstein", "Aristotle")
    agent = WikiAgent(task=game_2, tools=wiki_game_tools)
    agent_loop(agent, game_2, 30)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Once you've seen that the agent can accomplish the above, try out some different articles and see where the agent fails.

Check the messages in the chat history to see the full conversation between the agent and the user, to ensure that the messages that are printed above are faithful to the actual chat history (it can be easy to make minor mistakes that mess up the agent's `chat_history`).
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    for message in agent.chat_history:
        try:
            print(f"{str(message.role)}:\n {str(message.content)}")
        except:
            print(f"""{message["role"]}:\n {message["content"]}""")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 4️⃣ Elicitation
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
You may have observed that while the above implementation of `WikiAgent` succeeds at Albert Einstein → Aristotle, it fails at more difficult tasks. However, this doesn't mean that GPT-4o-mini does not have the capability to perform better on this task, but this capability might be blocked because we:

- Prompted the model poorly
- Stored the history poorly
- Didn't give the model sufficient tools to accomplish the task.
- ...

In general, it is hard to show that a model does not have a capability, even if we failed to demonstrate this capability. For example, it took 3.5 years after the release of GPT-2 (and 2.5 years after the release of GPT-3) for people to discover that [chain-of-thought reasoning](https://arxiv.org/abs/2201.11903) massively improves model performance, which enabled the same models to complete significantly harder tasks. LLM agent evals aim to elicit the best capability we possibly can, until we feel we've managed to gain [**evidence of absence**](https://en.wikipedia.org/wiki/Evidence_of_absence), **not** just **absence of evidence**.


Broadly speaking, there are two categories of elicitation:

1. **Narrow elicitation**: Task-specific methods that improve model performance on a particular task or small class of tasks, but likely won't impact model performance in general across many tasks. 
    - E.g. A tool that gives the model access to the content of arbitrary wikipedia articles. This will improve performance on this task significantly, but wouldn't generalize to other tasks.
2. **General elicitation**: Task-agnostic methods that improve model performance on a wide array of possible tasks. 
    - E.g. Chain-of-thought prompting: This tends to improve model performance on a wide array of tasks. These sorts of elicitation methods are the ones we're most interested in, as if researchers find an improvement to models that is roughly as easy and effective as chain-of-thought prompting, then we would see a very rapid increase in risk from AI.


We will try the following elicitation methods in this section:
1. Prompt engineering, including:
    - Chain-of-thought prompting
    - The ReAct framework
2. Reflexion, which allows the model to cheaply explore future paths
3. Improved message histories

Then you will be able to try further elicitation methods, including any of your own, as a bonus.

<details><summary>Tip - How to find wikipedia pages to test on</summary>

You might start having a hard time coming up with wikipedia pages to test on. Luckily, Wikipedia offers a random page link, which is accessible via: https://en.wikipedia.org/wiki/special:random. 

If the pages are *too* random and not connected by links, try the "special:random" link to a different language's wikipedia (which is sparser and contains more popular/well-connected pages). **Make sure to check this page exist in English!**

To test whether two pages are connect via links, use this free online tool to see the possible paths between pages: https://www.sixdegreesofwikipedia.com/ (be somewhat careful with this though, as the paths that this website believes are accessible may not be accessible to our agent). 

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Prompting

As you should already know, prompting can have a large impact on model performance. There are many changes you could make for prompts in this task. You should experiment first with more general elicitation methods such as getting the agent to think more deeply, and output plans in different ways. After this, you might try more narrow elicitation methods, such as:

- Telling the agent how many pages it's visited.
- Telling the agent if it's already visited the page it's on (and how many times).
- Schedule different prompts and planning methods for the "zoom out" and "zoom in" sections of the game, since we know that the general strategy for the wikipedia game looks like:

   `Narrow article (with few links) -> General article (with many links) -> Narrow article (with few links)`
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Engineer prompts
> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
> 
> You should spend up to 20-35 mins on this exercise.
> ```
Try and design prompts that improve the performance of the wikipedia agent. You may have to do a decent amount of experimentation here. Remember that your prompts will have to be robust to: 

* Different tasks within the wikipedia game, 
* Different states within those tasks,
* Different failure-modes the agent could encounter.

See if you can significantly improve performance. There's a test task below that you should aim to be able to solve with improved prompting.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

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
        # EXERCISE
        # raise NotImplementedError("You need to implement a new system_instruction property")
        # END EXERCISE
        # SOLUTION
        return {
            "role": "system",
            "content": f"You are a wikipedia-racing AI. Your goal is to reach {self.goal_page.title} by accessing links from wikipedia pages. Your current page is {self.current_page.title}.",
        }
        # END SOLUTION

    @property
    def on_page_instruction(self):
        """
        Provide improved instructions for the current page.

        Returns:
            dict: The instructions for the current page. "role" is "user" for user messages.
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement a new on_page_instruction property")
        # END EXERCISE
        # SOLUTION
        return {
            "role": "user",
            "content": f"""You are currently on page: {self.current_page.title}. Make sure you start by reasoning about what steps you should take to get to the article on {self.goal_page.title}. When coming up with a strategy, make sure to pay attention to the path you have already taken, and if your current strategy doesn't seem to be working out, try something else. In case you're unsure, {self.goal_page.title} has the following summary:\n\n[Begin Summary]\n{self.get_page_summary(self.goal_page)}\n[End Summary]\n\nThe path you have taken so far is {" -> ".join(self.page_history)}.
            """,
        }
        # END SOLUTION

    @property
    def next_step_instruction(self):
        """
        Provide improved instructions for the next step.

        Returns:
            dict: The instructions for the next step. "role" is "user" for user messages.
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement a new next_step_instruction property")
        # END EXERCISE
        # SOLUTION
        return {
            "role": "user",
            "content": f"""What's your next step?""",
        }
        # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

This isn't a *perfect* solution, but is an example of improved prompting compared to that in the `WikiGame` solution code.
```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Your original `WikiGame` and `WikiAgent` may not work on the example path "Linux" -> "Dana Carvey". But with sufficiently improved prompting, you should be able to get the agent to solve this task!
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    # Original WikiGame and WikiAgent
    game = WikiGame("Linux", "Dana Carvey")
    agent = WikiAgent(game, model="gpt-4o-mini", tools=wiki_game_tools)
    agent_loop(agent, game, 30)

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    #Improved WikiGame and WikiAgent
    game = WikiGamePrompting("Linux", "Dana Carvey")
    agent = WikiAgent(game, model="gpt-4o-mini", tools=wiki_game_tools)
    agent_loop(agent, game, 30)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Implement the ReAct framework
> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
> 
> You should spend up to 15-20 mins on this exercise.
> ```
The [**ReAct** framework](https://arxiv.org/abs/2210.03629) is an extension of chain-of-thought reasoning. In addition to prompting the model to simply think step-by-step, it separates this into two steps:

- **Re**asoning: The model is asked to reason about its current situation, and what sort of actions it should consider taking.
- **Act**ion: Then, the model is asked to perform an action based on its outputted reasoning.

Note that during the reasoning step, when you're calling the model without tools, OpenAI won't provide the model with a description of the tools. However, we still want the model to have information on its available tools when it's reasoning about what actions to take. Thus, we'll have to ensure that the tool descriptions are in the `system_instruction` we provide. (This will lead to some redundancy when the model takes an action, but this seems to be okay). This means that from now on we will have to pass the tools to both the *task* and the *agent*.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class WikiAgentReAct(WikiAgent):
    """
    Inherits from WikiAgent and adds the ReAct framework.

    Attributes:
        model (str): The model used for generating responses (inherited)
        client (OpenAI): OpenAI client for API calls (inherited)
        task (WikiGame): The current task being executed (inherited)
        chat_history (List[dict]): History of interactions (inherited)
        tools (List[Any]): List of tools (implemented below)

    Methods:
        get_response(use_tool: bool = True) -> ChatCompletionMessage: Get response from the model (inherited)

        execute_tool_calls(message: ChatCompletionMessage) -> List[str]: Execute tool calls from the model's response (inherited)

        run(with_tool: bool = True) -> bool: Run one loop of the Wikipedia agent (inherited)

        update_history(message : dict[str, str] | ChatCompletionMessage | List[dict[str, str] | ChatCompletionMessage]): Update self.chat_history and self.full_chat_history with a message or list of messages. (inherited)

        reset_history(): Empty self.chat_history of the agent. (inherited)

        handle_tool_calls(response: ChatCompletionMessage): Handles tool_calls in the wikipedia game context. (inherited)

        handle_refusal(response: ChatCompletionMessage): Handles refusals in the wikipedia game context. (inherited)

        start(): A function to put the starting instructions in agent.chat_history when the agent starts a new page or starts the game. (inherited)

        run(): This function runs the agent in the wikipedia game context. (inherited)


    """
    def __init__(self, starting_page: str, goal_page: str, tools=None):
        self.tools = tools
        super().__init__(starting_page, goal_page)

    @property
    def system_instruction(self):
        """
        Provided a description of the tools in the system message. When generate is called with tools this is redundant, but when generate is called without tools, this is useful.

        Returns:
            dict: The starting instructions. "role" is "system" for system messages.
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the new system_instruction property")
        # END EXERCISE
        # SOLUTION
        tool_descriptions = "\n".join([tool.description["function"]["name"] + ":" + tool.description["function"]["description"] for tool in self.tools])
        return {
            "role": "system",
            "content": f"""You are a wikipedia-racing AI. Your goal is to reach {self.goal_page.title} by accessing links from wikipedia pages. Your current page is {self.current_page.title}. You have access to {str(len(self.tools))} tools, which are:\n{tool_descriptions}""",
        }
        # END SOLUTION


    def generate_reason(self) -> ChatCompletionMessage:
        """
        Generate a reason for the agent to take an action. This function should:
            - Get the model to reason about the current state of the game (without tools)
            - Return the response from the model

        Returns:
            message (ChatCompletionMessage): The response from the model
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the generate_reason method")
        # END EXERCISE
        # SOLUTION
        # Get the model to reason about the current state of the game and add the response to the messages (you may not want to give it tools for this)
        self.chat_history.append(
            apply_user_format(
                "Think carefully about your current situation and what actions you want to take to get closer to"
                + self.task.goal_page.title
                + "."
            )
        )
        response = self.get_response(use_tool=False)
        return response
        # END SOLUTION

    def generate_action(self) -> ChatCompletionMessage:
        """

        Generate an action for the agent to take. This function should:
            - Get the model to generate an action for the agent to take (with tools)
            - Return the response from the model

        Returns:
            message (ChatCompletionMessage): The response from the model

        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the generate_action method")
        # END EXERCISE
        # SOLUTION
        # Get the model to generate an action based on the reasoning and add the response to the messages
        self.chat_history.append(apply_user_format("What action do you want to take?"))
        response = self.get_response(use_tool=True)
        return response
        # END SOLUTION

    def generate_reason_and_action(self) -> ChatCompletionMessage:
        """

        Generate a Reason and Action for the agent to take. This function should:
            - Generate a Reason
            - Add the Reason to the chat history
            - Generate an Action
            - Return the Action so that tool calls can be handled

        Returns:
            message (ChatCompletionMessage): The action from the model
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the generate_reason_and_action method")
        # END EXERCISE
        # SOLUTION
        reason = self.generate_reason()
        self.update_history(apply_assistant_format(reason.content))
        print("\nModel response ('Reason'):", reason.content)

        action = self.generate_action()

        return action
        # END SOLUTION

    def run(self):
        """
        Run one loop of the agent.

        This function should:
            - Generate a Reason and Action
            - Handle the tool calls, refusals, and no tool calls in the model response
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the new run method")
        # END EXERCISE
        # SOLUTION
        response = self.generate_reason_and_action()

        if response.tool_calls:
            self.handle_tool_calls(response)
        elif response.refusal:
            self.handle_refusal(response)
        # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
You may have to rewrite your `agent_loop` function (depending on how you implemented it originally).
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def agent_loop_ReAct(agent, num_loops = 10):
    """
    Run the agent loop for a given number of loops with the ReAct framework.

    Args:
        agent (WikiAgentReAct): The agent to run
        game (WikiGameReAct): The game to play
        num_loops (int): The number of loops to run
    """
    for i in range(num_loops):
        if agent.task.check_win():
            print("Success")
            return
        agent.run()

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Your `WikiAgent` and `WikiGamePrompting` with only improved prompting might not be able to solve "Drupe" → "17th parallel north" (or might not be able to solve it very effectively or reliably). However, your ReAct agent should be able to solve this path.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    # WikiGame and WikiAgent with only improved prompting
    game = WikiGamePrompting("Drupe", "17th parallel north")
    agent = WikiAgent(task=game, tools=wiki_game_tools)
    agent_loop(agent, game, 40)

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN: 
    # WikiGame and WikiAgent with ReAct
    game = WikiGameReAct("Drupe", "17th parallel north", tools=wiki_game_tools)
    agent = WikiAgentReAct(game, model="gpt-4o-mini", tools = wiki_game_tools)
    agent_loop_ReAct(game, agent,40)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Implement a reflexion tool
> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵⚪⚪
> 
> You should spend up to 25-35 mins on this exercise.
> ```

The [reflexion paper](https://arxiv.org/abs/2303.11366) builds on ReAct and proposes a method that improves performance by getting LLMs to do self-reflection. The original paper looks at LLM agents in a RL set-up, where getting a reward signal on the agent's signal is slow and expensive. The key idea is to get **quick cheap feedback** from an evaluator on every proposed action, then to **reflect** on this feedback before taking the next action, as opposed to waiting for the final outcome. In their case, the evaluator was a heuristic function that estimated the reward function. 

We will borrow this idea and build a tool that gives feedback on our ReAct model's proposed action by performing a look-ahead. We allow the agent to suggest candidate paths, then the tool will check if these paths work and inform the model where these paths go wrong (if they do). You'll need to add this tool to the list of tools.

We don't want to provide the agent the links or content of every page when it does this lookahead, as then we'd just be reimplementing a smaller version of the game *inside the game*. Instead, we'll let the agent suggest paths without seeing any content or links, and then let it know if this path works. It's very likely that a suggested link will — at some point — not be accessible from one of the pages, but this tool will still be useful to help the agent plan.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class TestPathTool():
    """
    Implements a tool that allows the agent to test paths from the current state of the game.

    Attributes:
        name (str): The name of the tool

    Methods:
        execute(task: WikiGame, path: str) -> str: Test if a given path is valid.

        description -> dict: Provides the description of the test_path tool for the API
    """

    name = "test_path"

    def execute(self, task: WikiGame, path: str) -> str:
        """
        Test if a given path is valid.

        Args:
            path (str): A string representing a path, e.g., "Barack Obama -> Indonesia -> India"
            task (WikiGame): The current task being run.

        Returns:
            str: A message indicating whether the path is valid or where it fails.
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the execute method for the TestPathTool")
        # END EXERCISE
        # SOLUTION
        path_nodes = [node.strip() for node in path.split("->")]

        if not path_nodes:
            return "ERROR: Empty path provided."

        if path_nodes[0] != task.current_page.title:
            return f"ERROR: The path should start with the current page: {task.current_page.title}"

        for i in range(len(path_nodes) - 1):
            current_node = path_nodes[i]
            next_node = path_nodes[i + 1]

            permitted_links = (link.lower() for link in task.get_permitted_links(current_node))

            if next_node.lower() not in permitted_links:
                return f"This path works until {next_node}, which is not accessible from {current_node}"

        return "This path is valid."
        # END SOLUTION

    @property
    def description(self):
        # EXERCISE
        # raise NotImplementedError("You need to implement the description property for the TestPathTool")
        # END EXERCISE
        # SOLUTION
        return {
            "type" : "function",
            "function" : {
                "name" : "test_path",
                "description" : "Accepts a test path string in the form \"current_page -> page1 -> page2 -> ... -> pageN\" and if the path does not work, then it returns where the path goes wrong, if the path does work it returns \"success.\" Be careful that path titles can be sensitive to plurals or rephrasings. This tool is especially useful to check longer plans.",
                "parameters" : {
                    "type" : "object",
                    "properties": {
                        "path" : {
                            "type" : "string",
                            "description" : "The path you want to test, formatted as \" current_page -> page1 -> page2 -> ... -> pageN\"."
                        },
                    },
                    "required" : ["path"]
                }
            }
        }
        # END SOLUTION
if MAIN:
    TestPathTool_inst = TestPathTool()
    wiki_game_tools = [get_content_tool_inst, move_page_tool_inst, TestPathTool_inst]

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Now come up with your own paths to run your agent with the TestPathTool tool, to see if it has improved the agent's performance.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Help! My agent isn't using the <code>TestPathTool</code></summary>

If your agent isn't using the test path tool, you may want to go back and modify your prompting. One way you could do this is to schedule a prompt to tell the agent to use the `TestPathTool` tool if it hasn't used it in its last few tool calls. Alternatively, you could just include in every `on_page_instruction` a strong indication that the agent should use this tool.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Let the LLM see its entire chat history
> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵⚪⚪⚪
> 
> You should spend up to 10-15 mins on this exercise.
> ```

You may have noticed that the agent performs significantly worse as a result of the fact that we decided to reset the chat history every time the agent encounters a new page. It often comes up with plans and doesn't follow through on them. We can fix this issue by letting the agent see the entirety of its chat history.

What we have to overcome is the context window considerations, specifically with regards to the length of wikipedia pages. However, we can fix these issues by resetting **only** the outputs of the `get_content()` function each time the agent moves to a new page, instead of resetting the entire chat history.

We'll modify the reset function in the `WikiAgentReAct` class to accomplish this.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class WikiAgentChatHistory(WikiAgentReAct):
    """
    Inherits from WikiAgentReAct and adds the ability to store and retrieve chat history.

    Attributes:
        model (str): The model used for generating responses (inherited)
        tools (List[Any]): List of tools (inherited)
        client (OpenAI): OpenAI client for API calls (inherited)
        task (WikiGame): The current task being executed (inherited)
        chat_history (List[dict]): History of interactions (inherited)
        full_chat_history (List[dict]): Full history of interactions

    Methods:
        get_response(use_tool: bool = True) -> ChatCompletionMessage: Get response from the model (inherited)

        execute_tool_calls(message: ChatCompletionMessage) -> List[str]: Execute tool calls from the model's response (inherited)

        run(with_tool: bool = True) -> bool: Run one loop of the Wikipedia agent (inherited)

        update_history(message : dict[str, str] | ChatCompletionMessage | List[dict[str, str] | ChatCompletionMessage]): Update self.chat_history and self.full_chat_history with a message or list of messages. (inherited)

        reset_history(): Empty self.chat_history of the agent. (modified below)

        handle_tool_calls(response: ChatCompletionMessage): Handles tool_calls in the wikipedia game context. (inherited)

        handle_refusal(response: ChatCompletionMessage): Handles refusals in the wikipedia game context. (inherited)

        start(): A function to put the starting instructions in agent.chat_history when the agent starts a new page or starts the game. (inherited)

        run(): This function runs the agent in the wikipedia game context. (inherited)

        store_chat_history(): Store the current chat history in the full chat history.

        retrieve_chat_history(): Retrieve the full chat history.
    """
    def reset_history(self):
        """
        Replace the output of get_content tool with an indication that wikipedia content was output when the agent moves to a new page
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement a new reset_history method")
        # END EXERCISE
        # SOLUTION
        for message in self.chat_history:
            if isinstance(message, dict):
                if message["role"] == "tool" and message["name"] == "get_content" and message["content"] != "Wikipedia content was output here.":
                    message["content"] = "Wikipedia content was output here."
                else:
                    pass
            else:
                pass
        # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Now see how your agent performs:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

game = WikiGameReAct("Drupe", "17th parallel north", tools=wiki_game_tools)
agent = WikiAgentChatHistory(game, model="gpt-4o-mini", tools = wiki_game_tools)
agent_loop_ReAct(game, agent, 40)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 5️⃣ Bonus
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Additional Tool use
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Implement a page summary tool
> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵⚪⚪⚪
> 
> You should spend up to 10-15 mins on this exercise.
> ```

Implement a tool that allows an agent to get a summary of any page that is accessible from its current page. This imitates a feature on wikipedia where you can see a short summary of a page when you hover over the link to it. You could either implement this tool so that the agent can just read the summary, or you can modify the `move_page` tool, so that the agent sees a summary of the page it wants to move to, and can then make a decision whether to ultimately move page.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class GetAccessiblePageSummaryTool():
    """
    Implements a tool that allows the agent to get the summary of a Wikipedia page (you should use the get_page_summary function from the agent class)
    """

    name = "get_accessible_page_summary"

    @staticmethod
    def get_page_summary(task: WikiGame, page_title: str) -> str:
        """
        Get summary of a wikipedia page, to the last full stop within the first 500 characters. This is used to give a brief overview of the page to the agent.

        Args:
            page (str): The Wikipedia page title.
            task (WikiGame): The current task object.

        Returns:
            str: The summary of the Wikipedia page.
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the get_page_summary method for the GetAccessiblePageSummaryTool")
        # END EXERCISE
        # SOLUTION
        page = task.get_page(page_title)
        if page in task.get_permitted_links():
            summary = page.content[:500]
            last_period_index = summary.rfind(".")
            return (
                summary[: last_period_index + 1] if last_period_index != -1 else summary
            )
        else:
            return "This page is not accessible from the current page."
        # END SOLUTION

    @property
    def description(self):
        """
        Provides the description of the get_page_summary tool

        Returns:
            dict: The description of the tool for the API
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the description property for the GetAccessiblePageSummaryTool")
        # END EXERCISE
        # SOLUTION
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Get the summary of a wikipedia page you are considering moving to, to the last full stop within the first 500 characters. The page needs to be accessible via a link from the current page. Anything which corresponds to a link you can select will be wrapped in <link></link> tags.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page_title": {
                            "type": "string",
                            "description": "The wikipedia page you want to get the summary of.",
                        }
                    },
                    "required": ["page_title"],
                },
            },
        }
        # END SOLUTION

GetAccessiblePageSummaryTool_inst = GetAccessiblePageSummaryTool()
wiki_game_tools = [GetContentTool_inst, MovePageTool_inst, TestPathTool_inst, GetAccessiblePageSummaryTool_inst]

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Implement an arbitrary page summary/content tool
> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵⚪⚪⚪
> 
> You should spend up to 5-10 mins on this exercise.
> ```

Now implement a tool that allows the agent to suggest dany wikipedia page, and get a brief summary of it. This may be helpful for the agent to formulate plans into the future.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class GetAnyPageContent():
    """
    Implements a tool that allows the agent to get the content of any Wikipedia page (not wrapped in link tags).
    """

    name = "get_any_page_content"

    @staticmethod
    def execute(task: WikiGame, page_title: str | None = None) -> str:
        """
        Get the content of any wikipedia page

        Also provides current page content if no page_title is provided.

        Args:
            page_title (str): The title of the Wikipedia page
            task (WikiGame): The current task being run.

        Returns:
            str: The content of the page (not wrapped in link tags).
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the execute method for the GetAnyPageContent tool")
        # END EXERCISE
        # SOLUTION
        if page_title:
            page = task.get_page(page_title)
            content = page.content
            return content
        else:
            content = task.current_page.content
            permitted_links = get_permitted_links(task.current_page)
            for word in sorted(permitted_links, key=len, reverse=True):
                content = re.sub(
                    r"""(\s|[,.)!?;:'"])(""" + re.escape(word) + r""")(\s|[,.)!?;:'"s])""",
                    r"""\1<link>\2</link>\3""",
                    content,
                    count=1,
                    flags=re.IGNORECASE,
                )
            return content
        # END SOLUTION

    @property
    def description(self):
        """
        Provides the description of the get_any_page_content tool

        Returns:
            dict: The description of the tool for the API
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the description property for the GetAnyPageContentTool")
        # END EXERCISE
        # SOLUTION
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Get all the content for any wikipedia page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page_title": {
                            "type": "string",
                            "description": "The wikipedia page you want to get the content of.",
                        }
                    },
                    "required": ["page_title"],
                },
            },
        }
        # END SOLUTION

# HIDE
GetAnyPageContentTool_inst = GetAnyPageContent()
wiki_game_tools = [GetContentTool_inst, MovePageTool_inst, TestPathTool_inst, GetAnyPageContentTool_inst]
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Implement additional rules
> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵⚪⚪⚪⚪
> ```

Allow the game to have additional rules. Some suggestions are a "No country" rule, and a "No articles above a given length" rule, but feel free to add more if you think of any others. With all of our elicitation methods, the agent generally only fails if the path is impossible or unreasonably hard. To implement a no country rule, you may want to use the wikipedia API's "categories" attribute for `WikipediaPage` objects.

First, let's modify the prompts in the Wikigame class so that we inform the agent about the additional rules it will have to abide by.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class WikiGameRules(WikiGameReAct):
    """
    Inherits from WikiGameReAct and adds the ability to store and display the rules of the game.

    Attributes:
        starting_page (str): The title of the starting page (inherited)
        goal_page (str): The title of the goal page (inherited)
        current_page (WikipediaPage): The current Wikipedia page (inherited)
        page_history (List[str]): The history of pages visited (inherited)
        full_chat_history (List[dict]): The full history of messages sent (inherited)

    Methods:
        get_page(title: str) -> WikipediaPage: Get a Wikipedia page object given a title (inherited)

        get_page_summary(page: WikipediaPage | None = None) -> str: Get the summary of a Wikipedia page (inherited)

        get_permitted_links(title: Optional[str] = None) -> list[str]: Get permitted links for the current page (inherited)

        is_permitted_link(link: str) -> bool: Check if a link is permitted (inherited)

        system_instruction -> dict: Generate the starting instructions for the game (inherited)

        on_page_instruction -> dict: Generate instructions for the current page (inherited)

        next_step_instruction -> dict: Generate instructions for the next step (inherited)

        check_win() -> bool: Check if the game has been won (inherited)
    """

    def __init__(self, starting_page: str, goal_page: str, rules: List[Literal["no countries", "no pages above length 30000"]], tools : list = None):
        super().__init__(starting_page, goal_page, tools)
        self.rules = rules if rules else None

    @property
    def system_instruction(self):
        """
        Provide improved starting instructions for the game.

        Returns:
            dict: The starting instructions. "role" is "system" for system messages.
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the system_instruction property")
        # END EXERCISE
        # SOLUTION
        tool_descriptions = "\n".join([tool.description["function"]["name"] + ":" + tool.description["function"]["description"] for tool in self.tools])
        if self.rules:
            return {
            "role" : "system",
            "content" : f"""You are a wikipedia-racing AI. Your goal is to reach {self.goal_page.title} by accessing links from wikipedia pages. Your current page is {self.current_page.title}. You have access to {str(len(self.tools))} tools, which are:\n{tool_descriptions}\n\nThe additional rules of the game are: {",".join(self.rules)}"""
        }
        else:
            return {
            "role" : "system",
            "content" : f"""You are a wikipedia-racing AI. Your goal is to reach {self.goal_page.title} by accessing links from wikipedia pages. Your current page is {self.current_page.title}. You have access to {str(len(self.tools))} tools, which are:\n{tool_descriptions}"""
        }
        # END SOLUTION

    @property
    def on_page_instruction(self):
        """
        Provide improved instructions for the current page.

        Returns:
            dict: The instructions for the current page. "role" is "user" for user messages.
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the on_page_instruction property")
        # END EXERCISE
        # SOLUTION
        return {
            "role" : "user",
            "content" : f"""You are currently on page: {self.current_page.title}. Make sure you start by reasoning about what steps you should take to get to the article on {self.goal_page.title}. When coming up with a strategy, make sure to pay attention to the path you have already taken, and if your current strategy doesn't seem to be working out, try something else. In case you're unsure, {self.goal_page.title} has the following summary:\n[Begin Summary]\n{self.get_page_summary(self.goal_page)}\n[End Summary]\n\nThe pages you've visited so far are: {" -> ".join(self.page_history)}\n\nThe rules of the game are: {",".join(self.rules)}"""
        }
        # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Now let's implement these rules by modifying the `MovePageTool` class, so that the agent can only move page if it's within the rules of the game.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class MovePageTool_rules(MovePageTool):
    """
    Inherits from move_page_tool and adds the ability to check the rules of the game.
    """

    @staticmethod
    def execute(new_page: str, task: WikiGame) -> str:
        """
        Changes your current page to a specified new page which is accessible via a link from the current page. You can only call this function once at a time, as it will take you to a different page.

        Only allow the agent to move if it is permitted by the rules.

        Args:
            task (BaseWikiGame): The current task object.
            new_page (str): The title of the new page to move to.

        Returns:
            str: A message indicating the result of the move
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the execute method for the MovePageTool_rules")
        # END EXERCISE
        # SOLUTION
        new_page_normalized = new_page.replace("_", " ")
        if task.is_permitted_link(new_page_normalized):
            if "no countries" in task.rules and any("countries in" in category for category in task.get_page(new_page_normalized).categories.lower()):
                return f"Couldn't move page to {new_page}. This page is in the category of countries."
            if "no pages above length 30000" in task.rules and len(task.get_page(new_page_normalized).content) > 30000:
                return f"Couldn't move page to {new_page}. This page is above the maximum length of 30000 characters."
            task.current_page = task.get_page(new_page_normalized)
            task.page_history.append(task.current_page.title)
            return f"Moving page to {task.current_page.title}"
        else:
            return f"Couldn't move page to {new_page}. This is not a valid link."
        # END SOLUTION

    @property
    def description(self):
        """
        Provides the description of the modified move_page tool

        Returns:
            dict: The description of the move_page tool for the API
        """
        # EXERCISE
        # raise NotImplementedError("You need to implement the description property for the MovePageTool_rules")
        # END EXERCISE
        # SOLUTION
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Changes your current page to a specified new page which is accessible via a link from the current page. You can only call this function once at a time, as it will take you to a different page. If any pages violate the rules of the game",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_page": {
                            "type": "string",
                            "description": 'The title of the new page you want to move to. This should be formatted the way the title appears on wikipedia (e.g. to move to the wikipedia page for the United States of America, you should enter "United States"). Underscores are not necessary.',
                        }
                    },
                    "required": ["new_page"],
                },
            },
        }
        # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Try further elicitation methods
> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵⚪⚪⚪
> ```
Read some of the resources we linked on the Intro to LLM agents page, and explore some of your own methods to elicit improved performance on the task. If you start seeing diminishing returns from elicitation (due to saturating performance on the task), come up with ways to make the task harder; or better yet, construct your own, more difficult, task.
'''

