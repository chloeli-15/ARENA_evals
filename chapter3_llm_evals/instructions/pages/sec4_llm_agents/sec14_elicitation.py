import streamlit as st


def section():
    st.sidebar.markdown(
        r"""
## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#introduction'>Introduction</a></li>
    <li class='margtop'><a class='contents-el' href='#content-learning-objectives'>Content & Learning Objectives</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#1-Intro to Inspect'>Intro to Inspect</a></li>
        <li><a class='contents-el' href='#2-mcq-benchmarks-exploration'>MCQ Benchmarks: Exploration</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
</ul>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        r'''
# Elicitation

> ### Learning objectives
>- Understand the importance of elicitation in evaluating LLM agents
>- Understand the different methods of elicitation
>- Understand how to improve prompting, tools, history storage, and information access in LLM agents


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


## Prompting

As you should already know, prompting can have a large impact on model performance. There are many changes you could make for prompts in this task. You should experiment first with more general elicitation methods such as getting the agent to think more deeply, and output plans in different ways. After this, you might try more narrow elicitation methods, such as:

- Telling the agent how many pages it's visited.
- Telling the agent if it's already visited the page it's on (and how many times).
- Schedule different prompts and planning methods for the "zoom out" and "zoom in" sections of the game, since we know that the general strategy for the wikipedia game looks like:

   `Narrow article (with few links) -> General article (with many links) -> Narrow article (with few links)`


### Exercise - Engineer prompts
```c
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 20-35 mins on this exercise.
```
Try and design prompts that improve the performance of the wikipedia agent. You may have to do a decent amount of experimentation here. Remember that your prompts will have to be robust to: 

* Different tasks within the wikipedia game, 
* Different states within those tasks,
* Different failure-modes the agent could encounter.

See if you can significantly improve performance. There's a test task below that you should aim to be able to solve with improved prompting.

```python
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
        # TODO
        return {}

    @property
    def on_page_instruction(self):
        """
        Provide improved instructions for the current page.

        Returns:
            dict: The instructions for the current page. "role" is "user" for user messages.
        """
        # TODO
        return {}

    @property
    def next_step_instruction(self):
        """
        Provide improved instructions for the next step.

        Returns:
            dict: The instructions for the next step. "role" is "user" for user messages.
        """
        # TODO
        return {}
```

<details><summary>Solution:</summary>

This isn't a perfect solution, but is an example of improved prompting to that in the solution in the basic `WikiGame` class.

```python
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
```

</details>




Your `WikiGame` and `WikiAgent` may not work on the example path "Linux" -> "Dana Carvey". But with sufficiently improved prompting, you should be able to get the agent to solve this task!

```python
# Original WikiGame and WikiAgent
game = WikiGame("Linux", "Dana Carvey")
agent = WikiAgent(game, model="gpt-4o-mini", tools=wiki_game_tools)
agent_loop(agent, game, 30)
```

```python
#Improved WikiGame and WikiAgent
game = WikiGamePrompting("Linux", "Dana Carvey")
agent = WikiAgent(game, model="gpt-4o-mini", tools=wiki_game_tools)
agent_loop(agent, game, 30)
```

### Exercise - Implement the ReAct framework
```c
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 15-20 mins on this exercise.
```
The [**ReAct** framework](https://arxiv.org/abs/2210.03629) is an extension of chain-of-thought reasoning. In addition to prompting the model to simply think step-by-step, it separates this into two steps:

- **Re**asoning: The model is asked to reason about its current situation, and what sort of actions it should consider taking.
- **Act**ion: Then, the model is asked to perform an action based on its outputted reasoning.

Note that during the reasoning step, when you're calling the model without tools, OpenAI won't provide the model with a description of the tools. However, we still want the model to have information on its available tools when it's reasoning about what actions to take. Thus, we'll have to ensure that the tool descriptions are in the `system_instruction` we provide. (This will lead to some redundancy when the model takes an action, but this seems to be okay). This means that from now on we will have to pass the tools to both the *task* and the *agent*.

```python
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
        # TODO

        pass

class WikiAgentReAct(WikiAgent):
    """
    Inherits from WikiAgent and adds the ReAct framework.

    Attributes:
        model (str): The model used for generating responses (inherited)
        tools (List[Any]): List of tools (inherited)
        client (OpenAI): OpenAI client for API calls (inherited)
        task (WikiGame): The current task being executed (inherited)
        chat_history (List[dict]): History of interactions (inherited)

    Methods:
        - get_response(use_tool: bool = True) -> ChatCompletionMessage: Get response from the model (inherited)
        - execute_tool_calls(message: ChatCompletionMessage) -> List[str]: Execute tool calls from the model's response (inherited)
        - update_history(message : dict[str, str] | ChatCompletionMessage | List[dict[str, str] | ChatCompletionMessage]): Update self.chat_history and self.full_chat_history with a message or list of messages. (inherited)
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
            message (ChatCompletionMessage): The response from the model
        """
        # TODO
        pass

    def generate_action(self) -> ChatCompletionMessage:
        """
        
        Generate an action for the agent to take. This function should:
            - Get the model to generate an action for the agent to take (with tools)
            - Return the response from the model
        
        Returns:
            message (ChatCompletionMessage): The response from the model
        
        """
        # TODO
        pass

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
        # TODO
        pass

    def run(self):
        """
        Run one loop of the agent.

        This function should:
            - Generate a Reason and Action
            - Handle the tool calls, refusals, and no tool calls in the model response
        """
        # TODO
        pass
```

<details><summary>Solution:</summary>

```python
class WikiGameReAct(WikiGamePrompting):
    """
    Inherits from WikiGame and adds the ReAct framework.

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

        system_instruction -> dict: Generate the starting instructions for the game (inherited)

        on_page_instruction -> dict: Generate instructions for the current page (inherited)

        next_step_instruction -> dict: Generate instructions for the next step (inherited)

        check_win() -> bool: Check if the game has been won (inherited)

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
        tool_descriptions = "\n".join([tool.description["function"]["name"] + ":" + tool.description["function"]["description"] for tool in self.tools])
        return {
            "role": "system",
            "content": f"""You are a wikipedia-racing AI. Your goal is to reach {self.goal_page.title} by accessing links from wikipedia pages. Your current page is {self.current_page.title}. You have access to {str(len(self.tools))} tools, which are:\n{tool_descriptions}""",
        }


class WikiAgentReAct(WikiAgent):
    """
    Inherits from WikiAgent and adds the ReAct framework.

    Attributes:
        model (str): The model used for generating responses (inherited)
        tools (List[Any]): List of tools (inherited)
        client (OpenAI): OpenAI client for API calls (inherited)
        task (WikiGame): The current task being executed (inherited)
        chat_history (List[dict]): History of interactions (inherited)

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

    def generate_reason(self) -> ChatCompletionMessage:
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

    def generate_action(self) -> ChatCompletionMessage:
        # Get the model to generate an action based on the reasoning and add the response to the messages
        self.chat_history.append(apply_user_format("What action do you want to take?"))
        response = self.get_response(use_tool=True)
        return response

    def generate_reason_and_action(self) -> ChatCompletionMessage:
        """
        Generate a reason, store this in history, then generate and return an action.
        """
        reason = self.generate_reason()
        self.update_history(apply_assistant_format(reason.content))
        print("\nModel response ('Reason'):", reason.content)

        action = self.generate_action()

        return action

    def run(self):
        """
        Run one loop of the agent.

        This function should:
            - Generate a Reason and Action
            - Handle the tool calls, refusals, and no tool calls in the model response
        """
        response = self.generate_reason_and_action()

        if response.tool_calls:
            self.handle_tool_calls(response)
        elif response.refusal:
            self.handle_refusal(response)
```

</details>


You may have to rewrite your `agent_loop` (depending on how you implemented it originally).

```python
def agent_loop_ReAct(game, agent, num_loops = 10):
    """
    Run the agent loop for a given number of loops with the ReAct framework.

    Args:
        agent (WikiAgentReAct): The agent to run
        game (WikiGameReAct): The game to play
        num_loops (int): The number of loops to run
    """
    pass
```

<details><summary>Solution:</summary>

```python
def agent_loop_ReAct(game, agent, num_loops = 10):
    """
    Run the agent loop for a given number of loops with the ReAct framework.

    Args:
        agent (WikiAgentReAct): The agent to run
        game (WikiGameReAct): The game to play
        num_loops (int): The number of loops to run
    """
    agent.start()
    for i in range(num_loops):
        if game.check_win():
            print("Success")
            return
        agent.run()
```
</details>

Your `WikiAgent` and `WikiGamePrompting` with only improved prompting might not be able to solve "Drupe" → "17th parallel north" (or might not be able to solve it very effectively or reliably). However, your ReAct agent should be able to solve this path.

```python
# WikiGame and WikiAgent with improved prompting
game = WikiGamePrompting("Drupe", "17th parallel north")
agent = WikiAgent(task=game, tools=wiki_game_tools)
agent_loop(agent, game, 40)
```

```python
# WikiGame and WikiAgent with ReAct
game = WikiGameReAct("Drupe", "17th parallel north", tools=wiki_game_tools)
agent = WikiAgentReAct(game, model="gpt-4o-mini", tools = wiki_game_tools)
agent_loop_ReAct(game, agent,40)
```

### Exercise - Implement a reflexion tool
```c
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 25-35 mins on this exercise.
```

The [reflexion paper](https://arxiv.org/abs/2303.11366) builds on ReAct and proposes a method that improves performance by getting LLMs to do self-reflection. The original paper looks at LLM agents in a RL set-up, where getting a reward signal on the agent's signal is slow and expensive. The key idea is to get **quick cheap feedback** from an evaluator on every proposed action, then to **reflect** on this feedback before taking the next action, as opposed to waiting for the final outcome. In their case, the evaluator was a heuristic function that estimated the reward function. 

We will borrow this idea and build a tool that gives feedback on our ReAct model's proposed action by performing a look-ahead. We allow the agent to suggest candidate paths, then the tool will check if these paths work and inform the model where these paths go wrong (if they do). You'll need to add this tool to the list of tools.

We don't want to provide the agent the links or content of every page when it does this lookahead, as then we'd just be reimplementing a smaller version of the game *inside the game*. Instead, we'll let the agent suggest paths without seeing any content or links, and then let it know if this path works. It's very likely that a suggested link will — at some point — not be accessible from one of the pages, but this tool will still be useful to help the agent plan.

```python
class TestPathTool():
    """
    Implements a tool that allows the agent to test paths from the current state of the game.

    Attributes:
        name (str): The name of the tool

    Methods:
        execute(task: WikiGame, path: str) -> str: Test if a given path is valid.

        description -> dict: Provides the description of the TestPathTool tool for the API
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
        # TODO
        pass
    
    @property
    def description(self) -> dict:
        # TODO
        return {}

TestPathTool_inst = TestPathTool()
wiki_game_tools = [get_content_tool_inst, move_page_tool_inst, TestPathTool_inst]
```

<details><summary>Solution:</summary>

```python
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
        path_nodes = [node.strip() for node in path.split("->")]
        
        if not path_nodes:
            return "ERROR: Empty path provided."
        
        if path_nodes[0] != task.current_page.title:
            return f"ERROR: The path should start with the current page: {task.current_page.title}"
        
        for i in range(len(path_nodes) - 1):
            current_node = path_nodes[i]
            next_node = path_nodes[i + 1]
            
            permitted_links = set(link.lower() for link in task.get_permitted_links(current_node))
            
            if next_node.lower() not in permitted_links:
                return f"This path works until {next_node}, which is not accessible from {current_node}"
        
        return "This path is valid."
    
    @property
    def description(self):
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
```

</details>


Now run your own tests on your different agents on path to see if the `TestPathTool` tool has improved the agent's performance.

```python
game = WikiGameReAct("", "", tools = wiki_game_tools)
agent = WikiAgentReAct(game, model="gpt-4o-mini", tools = wiki_game_tools)
agent_loop_ReAct(game,agent, 40)
```

<details><summary>Help: My agent isn't using the <code>TestPathTool</code> tool</summary>

If your agent isn't using the test path tool, you may want to go back and modify your prompting. One way you could do this is to schedule a prompt to tell the agent to use the `TestPathTool` tool if it hasn't used it in its last N tool calls. Alternatively, you could just include in every `on_page_instruction` an indication that the agent should use this tool.

</details>

### Exercise - Let the LLM see its entire chat history
```c
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵⚪⚪⚪

You should spend up to 10-15 mins on this exercise.
```

You may have noticed that the agent performs significantly worse as a result of the fact that we decided to reset the chat history every time the agent encounters a new page. It often comes up with plans and doesn't follow through on them. We can fix this issue by letting the agent see the entirety of its chat history.

What we have to overcome is the context window considerations, specifically with regards to the length of wikipedia pages. However, we can fix these issues by resetting **only** the outputs of the `get_content()` function each time the agent moves to a new page, instead of resetting the entire chat history.

We'll modify the reset function in the `WikiAgentReAct` class to accomplish this.

```python
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
        - get_response(use_tool: bool = True) -> ChatCompletionMessage: Get response from the model (inherited)
        - execute_tool_calls(message: ChatCompletionMessage) -> List[str]: Execute tool calls from the model's response (inherited)
        - update_history(message : dict[str, str] | ChatCompletionMessage | List[dict[str, str] | ChatCompletionMessage]): Update self.chat_history and self.full_chat_history with a message or list of messages. (inherited)
        - reset_history(): Empty self.chat_history of the agent. (modified below)
        - handle_tool_calls(response: ChatCompletionMessage): Handles tool_calls in the wikipedia game context. (inherited)
        - handle_refusal(response: ChatCompletionMessage): Handles refusals in the wikipedia game context. (inherited)
        - start(): A function to put the starting instructions in agent.chat_history when the agent starts a new page or starts the game. (inherited)
        - run(): This function runs 1 loop of the agent in the wikipedia game. (inherited)
        - store_chat_history(): Store the current chat history in the full chat history.
        - retrieve_chat_history(): Retrieve the full chat history.
    """
    def reset_history(self):
        """
        Replace the output of the get_content tool responses with "Wikipedia Content was output here" when the agent moves to a new page.

        This function should only be called if the agent moved to a new page. It should:
            - Look through the messages in the chat history
            - Determine if a message is a get_content tool call response
            - Replace the output of the get_content tool response with "Wikipedia Content was output here"
        """
        # TODO
        pass
```

<details><summary>Solution:</summary>

```python
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
        for message in self.chat_history:
            if isinstance(message, dict):
                if message["role"] == "tool" and message["name"] == "get_content" and message["content"] != "Wikipedia content was output here.":
                    message["content"] = "Wikipedia content was output here."
                else:
                    pass
            else:
                pass
```

</details>

See how your agent performs now:

```python
game = WikiGameReAct("Drupe", "17th parallel north", tools=wiki_game_tools)
agent = WikiAgentChatHistory(game, model="gpt-4o-mini", tools = wiki_game_tools)
agent_loop_ReAct(game, agent, 40)
```

''',
        unsafe_allow_html=True,
    )