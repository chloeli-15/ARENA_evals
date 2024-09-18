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


We will try:
1. Prompt engineering, including:
    - Chain-of-thought prompting
    
    - The ReAct framework
     
2. Reflexion (allowing the model to cheaply explore future paths)

3. Improved message histories

Then you will be able to try further elicitation methods, including any of your own, as a bonus.

<details><summary>Some notes on Wikipedia and Implementation</summary>

You might start having a hard time coming up with wikipedia pages to test on. Luckily, Wikipedia offers a random page link, which is accessible via: https://en.wikipedia.org/wiki/special:random. Sometimes pages are *too* random, and won't be accessible by links between each other. One suggestion for coming up with "random but sensible" wikipedia pages is to go to a different language's wikipedia (which is generally much smaller than English wikipedia), and then use the "special:random" link (it works in every language). The majority of pages in other languages will also exist in English, but you'll want to switch languages back to English to check (and get the English title).

If you want to test whether two pages are accessible via links from each other, then use this free online tool to see the possible paths between pages: https://www.sixdegreesofwikipedia.com/ (be somewhat careful with this though, as the paths that this website believes are accessible may not be accessible to our agent). 

</details>


## Prompting

You should already be aware that prompting can have a large impact on model performance. There are a wide variety of possible changes you could make for prompts in this task. You should experiment first with more general elicitation methods such as getting the agent to think more deeply, and output plans in different ways. After this, you might try a wide array of narrow elicitation methods including:

- Telling the agent how many pages it's visited.

- Telling the agent if it's already visited the page it's on (and how many times).

- Schedule different prompts and planning methods for the "zoom out" and "zoom in" sections of the game, since we know that the general strategy for the wikipedia game looks like:

    Specific article (with few links) -> General article (with many links) -> Specific article (with few links)


### Exercise - Engineer prompts
```c
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 20-25 mins on this exercise.
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
        return {}

    @property
    def on_page_instruction(self):
        """
        Provide improved instructions for the current page.

        Returns:
            dict: The instructions for the current page. "role" is "user" for user messages.
        """
        return {}

    @property
    def next_step_instruction(self):
        """
        Provide improved instructions for the next step.

        Returns:
            dict: The instructions for the next step. "role" is "user" for user messages.
        """

        return {}
```

Your `WikiGame` and `WikiAgent` may not work on the example path "Linux" -> "Dana Carvey". But with improved prompting, you should be able to get the agent to solve this task!

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

Chain-of-thought prompting confers significant benefits to model performance, and you probably tried it when you messed around with prompting above. But when we're using LLMs as agents, we may want to provide a different structure to elicit reasoning. This is called the [**ReAct** framework](https://arxiv.org/abs/2210.03629); it consists of:

- Getting the model to generate **Re**asoning about its current situation, and what sort of actions it should consider taking.

- Then getting the model to perform an **Act**ion based on its outputted reasoning.

Remember that if you're calling the model without tools, OpenAI won't automatically provide the model with a description of the tools in its system message, so we'll have to ensure that the tool descriptions are in the `system_instruction` we provide (this will lead to some redundancy when the model takes an action, but this seems to be okay). This means that from now on we will have to feed both the *task* and the *agent* the list of tools the agent can use.

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

        get_page(title: str) -> WikipediaPage: Get a Wikipedia page object given a title (inherited)

        get_page_summary(page: WikipediaPage | None = None) -> str: Get the summary of a Wikipedia page (inherited)

        get_permitted_links(title: Optional[str] = None) -> list[str]: Get permitted links for the current page (inherited)

        is_permitted_link(link: str) -> bool: Check if a link is permitted (inherited)

        system_instruction -> dict: Generate the starting instructions for the game (inherited)

        on_page_instruction -> dict: Generate instructions for the current page (inherited)

        next_step_instruction -> dict: Generate instructions for the next step (inherited)

        get_instructions(system: bool, on_page: bool, next_step: bool) -> str: Generate instruction messages based on the current game state (inherited)

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
        return {}


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
        get_response(use_tool: bool = True) -> ChatCompletionMessage: Get response from the model (inherited)

        execute_tool_calls(message: ChatCompletionMessage) -> List[str]: Execute tool calls from the model's response (inherited)

        run(with_tool: bool = True) -> bool: Run one loop of the Wikipedia agent (inherited)

        update_history(message : str | ChatCompletionMessage | List[str | ChatCompletionMessage]): Update self.chat_history and self.full_chat_history with a message or list of messages. (inherited)

        reset_history(): Empty self.chat_history of the agent. (inherited)

        handle_tool_calls(response: ChatCompletionMessage): Handles tool_calls in the wikipedia game context. (inherited)

        handle_refusal(response: ChatCompletionMessage): Handles refusals in the wikipedia game context. (inherited)

        start(): A function to put the starting instructions in agent.chat_history when the agent starts a new page or starts the game. (inherited)

        run(): This function runs the agent in the wikipedia game context. (inherited)


    """

    def generate_reason(self) -> ChatCompletionMessage:
        """
        
        Generate a reason for the agent to take an action. This function should:
            - Get the model to reason about the current state of the game (without tools)
            - Return the response from the model
        
        Returns:
            ChatCompletionMessage: The response from the model
        """
        pass

    def generate_action(self) -> ChatCompletionMessage:
        """
        
        Generate an action for the agent to take. This function should:
            - Get the model to generate an action for the agent to take (with tools)
            - Return the response from the model
        
        Returns:
            ChatCompletionMessage: The response from the model
        
        """
        pass

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
        pass

    def run(self):
        """
        Run one loop of the agent.

        This function should:
            - Generate a Reason and Action
            - Handle the tool calls, refusals, and no tool calls in the model response
        """
        pass
```

You may have to rewrite your `agent_loop`.

```python
def agent_loop_ReAct(game, agent, num_loops = 10):
    """
    Run the agent loop for a given number of loops with the ReAct framework.

    Args:
        agent (WikiReActAgent): The agent to run
        game (WikiGameReAct): The game to play
        num_loops (int): The number of loops to run
    """
    pass
```
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

You should spend up to 10-15 mins on this exercise.
```
[Add a diagram for Reflexion]

[This paper](https://arxiv.org/abs/2303.11366) finds better performance by LLMs on tasks when they can perform "lookahead" and get feedback on their plans. We will imitate this by allowing the agent to suggest candidate paths, and informing it where these paths go wrong (if they do). You'll need to add this tool to the list of tools.

We don't want to provide the agent the links/content of every page when it does this lookahead, as then we'd just be reimplementing a smaller version of the game *inside the game*. Instead, we'll let the agent suggest paths without seeing any content or links, and then let it know if this path works. It's very likely that a suggested link will — at some point — not be accessible from one of the pages, but this tool will still be useful to help the agent plan.

```python
class test_path_tool():
    """
    Implements a tool that allows the agent to test paths from the current state of the game.

    Attributes:
        name (str): The name of the tool

    Methods:
        execute(task: Any, path: str) -> str: Test if a given path is valid.

        description -> dict: Provides the description of the test_path tool for the API
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
        pass
    
    @property
    def description(self) -> dict:
        return {}

test_path_tool_inst = test_path_tool()
wiki_game_tools = [get_content_tool_inst, move_page_tool_inst, test_path_tool_inst]
```

Now run your own tests on your different agents on path to see if the `test_path` tool has improved the agent's performance.

```python
game = WikiGameReAct("", "", tools = wiki_game_tools)
agent = WikiAgentReAct(game, model="gpt-4o-mini", tools = wiki_game_tools)
agent_loop_ReAct(game,agent, 40)
```

<details><summary>Help: My agent isn't using the <code>test_path</code> tool</summary>

If your agent isn't using the test path tool, you may want to go back and modify your prompting. One way you could do this is to schedule a prompt to tell the agent to use the `test_path` tool if it hasn't used it in its last N tool calls. Alternatively, you could just include in every `on_page_instruction` an indication that the agent should use this tool.

</details>

### Exercise - Let the LLM see its entire chat history
```c
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵⚪⚪⚪
```

You may have noticed that the agent performs significantly worse as a result of the fact that we decided to reset the chat history every time the agent encounters a new page. It often comes up with plans and doesn't follow through on them. We can fix this issue by letting the agent see the entirety of its chat history.

What we have to overcome is the context window considerations, specifically with regards to the length of wikipedia pages. However, we can fix these issues by resetting the outputs of the `get_content` function each time the agent moves to a new page, instead of resetting the entire chat history.

We'll modify the reset function in the `WikiAgentReAct` class to accomplish this.

```python
class WikiAgentChatHistory(WikiAgentReAct):
    """
    Inherits from WikiAgentReAct and adds the ability to store and retrieve chat history.

    Attributes:
        model (str): The model used for generating responses (inherited)
        tools (List[Any]): List of tools (inherited)
        client (OpenAI): OpenAI client for API calls (inherited)
        task (Any): The current task being executed (inherited)
        chat_history (List[dict]): History of interactions (inherited)
        full_chat_history (List[dict]): Full history of interactions

    Methods:
        get_response(use_tool: bool = True) -> ChatCompletionMessage: Get response from the model (inherited)

        execute_tool_calls(message: ChatCompletionMessage) -> List[str]: Execute tool calls from the model's response (inherited)

        run(with_tool: bool = True) -> bool: Run one loop of the Wikipedia agent (inherited)

        update_history(message : str | ChatCompletionMessage | List[str | ChatCompletionMessage]): Update self.chat_history and self.full_chat_history with a message or list of messages. (inherited)

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
        Replace the output of the get_content tool responses with "Wikipedia Content was output here" when the agent moves to a new page.

        This function should only be called if the agent moved to a new page. It should:
            - Look through the messages in the chat history
            - Determine if a message is a get_content tool call response
            - Replace the output of the get_content tool response with "Wikipedia Content was output here"
        """
        pass
```                     

See how your agent performs now:

```python
game = WikiGameReAct("Drupe", "17th parallel north", tools=wiki_game_tools)
agent = WikiAgentChatHistory(game, model="gpt-4o-mini", tools = wiki_game_tools)
agent_loop_ReAct(game, agent, 40)
```

''',
        unsafe_allow_html=True,
    )