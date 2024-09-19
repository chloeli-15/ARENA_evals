import streamlit as st


def section():
    st.sidebar.markdown(r"""
## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#introduction'>Introduction</a></li>
    <li class='margtop'><a class='contents-el' href='#content-learning-objectives'>Content & Learning Objectives</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#1-intro-to-llm-agents'>Intro to LLM Agents</a></li>
        <li><a class='contents-el' href='#2-simple-arithmetic-agent'>Simple Arithmetic Agent</a></li>
        <li><a class='contents-el' href='#3-wikiagent'>WikiAgent</a></li>
        <li><a class='contents-el' href='#4-elicitation'>Elicitation</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
</ul>""",
        unsafe_allow_html=True,
    )



    st.markdown(
        r'''
# Building a Simple Arithemtic Agent

> ### Learning Objectives
> 
> - Understand that a LLM agent is just a "glorified for-loop" (of the scaffolding program interacting with the LLM API).
> - Learn how to use function calling to allow LLMs to use external tools.
> - Understand the main functionalities of an LLM agent:  


We will start by building a simple LLM agent that solves arithmetic problems. LLMs struggle with arithmetic, but we can drastically improve their performance by providing a simple calculation tool. We'll try the model with and without tools on this task, and see how significantly performance improves.

To build this, we will implement 4 things:
- The `ArithmeticTask` class handles arithmetic problem generation and solution verification.
- The `CalculateTool`, a tool that LLM agents can use to solve the task.
- The `ArithmeticAgent` class handles interacting with the LLM API, doing the calculation, and keeping track of the overall task progress.
- The `agent_loop()` function defines the interaction loop between the task and the agent to execute the task.


In general, most LLM agents share these core components:

<img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/refs/heads/main/img/ch3-sec4-agent-overview.png" width="1000">


1. **LLM API interface**: A basic function that makes API calls (e.g. `get_response()`). <!-- (IN AGENT)-->
2. **Actions/Tools**: A set of actions the agent can take. <!-- (MOSTLY IN TASK)-->
3. **Task State Management**: Keeping track of the current state of the task and any relevant context. <!-- (IN TASK MOSTLY)-->
4. **Memory**: A way to store and retrieve information from past interactions (i.e. chat history). The simplest implemention is usually to store the list of past chat messages in a `chat_history` class attribute. <!-- (IN AGENT)-->
5. **Observation Parser**: Functions to parse and interpret the results of actions and update the state. <!-- (IN TASK MOSTLY)-->
6. **Decision/Execution Logic**: The rules or algorithms used to choose actions based on the current state and LLM output. <!-- (KIND OF IN BETWEEN)-->
7. **Task-Specific Information**: Any additional information or functions specific to the task at hand. <!-- (IN TASK)-->


These components are implemented across the `Task`, `Agent`, and `Tool` classes. However, the specific breakdown of these components in our implementation is a design choice and can vary depending on the task. While some seem more natural (e.g. LLM API interface goes into `Agent`, task state management goes into task), others can vary (e.g. Tools can be functions within the Task or the Agent, as opposed to being separate classes; observation parsing could go either way). In general, we want to maximize separability and minimize interfaces/dependencies, so that we can easily swap out the agent for the same task, or vice versa. 


## Task

In an LLM agent eval, there will usually be a `Task` class that interacts with the `Agent`. In general, the `Task` will:

- Prepare and provide the task instruction (and necessary files, functions etc) to the agent
- Parse and score the agent's output
- Update the task state accordingly (e.g. proceeds onto the next step of the task, ends the task).


### Exercise - Build a simple arithmetic task
```c
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵⚪⚪⚪

You should spend up to 20-25 minutes on this exercise.
```
We will build a toy task called `ArithmeticTask`. This task takes in two numbers and create a list of arithmetic calculation problems with these two numbers, using arithmetic operations defined in `operations`. It should have methods to do the following:

- Get the current problem (e.g. at the start this will be "Calculate `num1` + `num2`")
- Check if a given answer is correct
- Update the current problem if the model answer was correct, or after a certain number of attempts
- Check if all problems have been solved

</details>

```python
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
        return {}

    @property
    def get_current_task(self) -> str:
        """
        Gets the current task for the agent

        Returns:
            str: A string containing the current task
        """
        return ""

    @property
    def current_task_instruction(self) -> str:
        """
        Gets a string containing instructions for the current task for the agent. This will be fed to the agent as a user prompt.

        Returns:
            str: A string containing the instructions for the current task
        """
        return ""

    def check_solved(self) -> bool:
        """
        Checks if all tasks have been solved

        Returns:
            bool: True if all tasks have been solved, False otherwise
        """
        return True

    def check_answer(self, model_answer: str) -> bool:
        """
        Checks if the model's answer is correct

        Args:
            model_answer (str): The model's answer

        Returns:
            bool: True if the model's answer is correct, False otherwise
        """
        return True

    def update_current_task(self):
        """
        Sets is_solved for the current task to True and increments self.current_task_number by one
        """
        pass


x = ArithmeticTask(10, 15)
for problem, answer in x.correct_answers.items():
    print(f"{problem} = {answer}")
```
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

## Tool use via function calling

The simplest way for LLMs to take actions is via function calling. **Function calling** is an built-in feature of LLM Chat APIs that allows models to use external "tools" (i.e. Python functions, APIs) by simply receiving and outputing text. This involves 5 simple steps:

1. Pick a function in your codebase that the model should be able to call
2. Describe your function in the syntax of the model's API so the model knows how to call it
3. Pass your function definitions as available “tools” to the model, along with the messages
4. Receive and handle the model response
5. Provide the function call result back to the model 

**This loop of prompting the LLM with tools, executing its actions, and returning the results forms the basis of all LLM agents.** It allows LLMs to perform complex tasks like playing a game or completing a coding project "autonomously".

We will implement each step of the loop below.

<!--[DIAGRAM]
The workflow for allowing models to use tools in this way will look something like:

- We generate a response where the model is allowed to use a tool.
- The model generates a response and makes a tool call.
- We implement the function that the model has called in our code.
- We get the output of this function (i.e. the answer to the tool call).
- We format the tool response using the `apply_tool_call_format()` below (we'll need to include the original tool call message that the model made, so that we can provide the correct ID for the tool call)
- We add this formatted tool response to the message history, and get the agent to generate a new response, now that it has the answer to the tool it called in its context.

-->


### Exercise - Write `CalculateTool`
```c
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 10-15 minutes on this exercise.
```

We will define a tool class for our simple `calculate()` function with the following structure (don't need to run the code cell below, just read):

```python
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
```

For the `CalculateTool`, it should have the following methods:
- `execute()` - This should take in an arithmetical expression as string (e.g. `"3+5"`) and return the result of this expression (also as a string). The `execute()` function should take the task as a variable, as often tools will need to be able to make changes to the task state (e.g. update the current problem).
- `description` - This should return a dictionary containing the tool description in the correct API syntax. 

**How to handle calculations?** We have implemented a helper function `evaluate_expression()` to evaluate the arithmetic expressions, which you should use in your implementation of `execute()`. <!-- insert function docstring--> 

<details><summary>Aside: Why not use Python's in-built <code>eval()</code> function?</summary>

Python's `eval()` function evaluates an arbitrary string expression, and so allows AI models to run arbitrary code. Unless you have set up a container or sandboxed environment, it is very bad to allow LLMs to run arbitrary code on your computer!

</details>

#### Tool Description
Models like ChatGPT and Claude are fine-tuned to interpret and respond to `tool` descriptions appropriately, just like `user` and `system` messages. Below is an example of a typical tool description for the OpenAI API (see their [function calling guide](https://platform.openai.com/docs/guides/function-calling) for more details). Note that this may differ between APIs.

```python
{
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
    }
}
```

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

```python
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
        return ""

    @property
    def description(self):
        """
        Provides the description of the tool

        Returns:
            str: The description of the tool
        """

        return {}


Calculator = CalculateTool()
```

You can pass the tool to the model by providing the tool description to the `tools` parameter of the API call. This input has to be a list. The following code provides a standard example of this.

```python
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
```

<details><summary>Why is <code>message.content = None</code>?</summary>

When LLMs use tools, they often don't generate any text output. This can be a problem later when you try to get the model to do chain-of-thought reasoning. To get around this, it can be better to make two calls to the model for more complex tool use: one call to get the model to reason about the actions it should take, and then another to get the model to use a tool to take those actions.

</details> 

<details><summary> Aside - What is a <code>@staticmethod</code>?</summary>

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

### Exercise - Return tool call results
```c
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 10-15 minutes on this exercise.
```

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

```python
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
```
Now use the function above to return a tool call response to the model after the following message:

```python
messages = [{"role": "user", "content": "Calculate 5/3. Be precise."}]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[Calculator.description],
    tool_choice="auto",
)

# TODO: Return the tool call responses back to the model

```
## Agent

We will first implement a `SimpleAgent` class that is not specific to the `ArithmeticTask`, so that we can see the key components of an generic LLM agent.

### Exercise - Implement `SimpleAgent`
```c
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵🔵

You should spend up to 20-25 minutes on this exercise.
```

Build out the following simple agent class by filling in `get_response()` and `execute_tool_calls()` functions.

- `get_response()`: This should make an API call and return the `ChatCompletionMessage`from the model. It should be able to either use tool calling or not, depending on the `use_tool` argument).
- `execute_tool_calls()`: This should execute the tool calls in the message and return a list of tool responses as strings (we can format them correctly in `run()`).
- `run()`: This should define the main execution logic for running 1 loop of the agent. As this is largely determined by the task, this method in `SimpleAgent` is just a dummy method and should be overridden in specific agent classes.

```python
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

        return response.choices[0].message

    def execute_tool_calls(self, message: ChatCompletionMessage) -> List[str]:
        """
        Execute the tool calls in the message and return a list of tool_responses.

        Args:
            message (ChatCompletionMessage): The message containing the tool calls

        Returns:
            List[str]: A list of tool responses (as strings, we'll format them correctly in run())
        """

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
```

Try to execute the tool calls

```python
my_simple_agent = SimpleAgent(ArithmeticTask(10, 15), tools=[Calculator])
my_simple_agent.run()
```

### Exercise - Build an `ArithmeticAgent`
```c
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵🔵

You should spend up to 20-25 minutes on this exercise.
```

Now build our agent that will interact with the `ArithmeticTask` (with a calculator tool). Fill in the methods in the class below.

```python
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
        pass

    def handle_refusal(self, response: ChatCompletionMessage):
        """
        Handle the refusal from the model response. This function should only be called if the model refuses to answer and should:
        - Append the refusal to the chat history
        - Update the task state

        Args:
            response (ChatCompletionMessage): The response from the model
        """
        pass

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

        pass

    def run(self, with_tool: bool):
        """
        Run one loop of the agent, which involves:
        - getting a task
        - getting a response from the model
        - handling the model response, including tool calls, refusals, no tool calls, parsing and checking final answers, errors.
        - managing memory: storing the history of messages to self.chat_history
        - managing task state: staying on the same task or moving to the next task at the end of the loop
        """
        pass

    def parse_answer(self, message: ChatCompletionMessage) -> float:
        """
        Extract the numerical answer from the string output of the model

        Args:
            message (ChatCompletionMessage): The response from the model

        Returns:
            float: The numerical answer extracted from the model
        """
        return float(response[startpoint:endpoint])
```
<details><summary>Note on <code>handle_refusal()</code></summary>

The `ChatCompletionMessage` object contains a `refusal` attribute that can be used to determine if the model has refused to answer. If the model has refused to answer, the `refusal` will contain this content and we can print this out. We have included this for completeness, but it is not necessary to implement this function because it almost never occurs in the WikiGame.

See the [OpenAI API documentation](https://platform.openai.com/docs/guides/function-calling/edge-cases) for more information on the `refusal` attribute.

</details>

### Exercise - Run the task via an agent_loop 
```c
Difficulty: 🔴⚪⚪⚪⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 5-10 minutes on this exercise.
```

Try implementing the agent_loop below with and without tools, to see how much better the model does when we give it tools.

> WARNING! 
>
>When you're making API calls to LLMs to tasks, it can be tempting to use a while loop, and run the model until it finishes the task. But since every time we run a model we make an API call, this would allow us to spend arbitrarily large amounts of money on API calls. For this reason, ***always use a for loop when making API calls!!!*** It would be really unfortunate if you blew all your API budget on one mistake!


```python
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
    pass


agent_loop(arithmetic_agent_1, arithmetic_task_1)
```

If we want to see how the model performed at the task, then we can print all the messages from the `ChatHistory` as follows:

```python
for message in arithmetic_agent_1.chat_history:
    try:
        print(f"""{str(message.role)}:\n {str(message.content)}\n""")
    except:
        print(f""" {message["role"]}:\n {message["content"]}\n""")
```
                
''',
        unsafe_allow_html=True,
    )
