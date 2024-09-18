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
# Building a More Complex Task: WikiGame

We will now build a more complicated task. This won't be instantly solvable by LLMs with simple tool calls and will require us to elicit better capabilities from models.

The task we will build and elicit behavior for will be the [Wikipedia Game](https://en.wikipedia.org/wiki/Wikipedia:Wiki_Game): Players use wiki-links to travel from one Wikipedia page to another and the first person who reaches the destination page wins the race. This is not directly related to any dangerous capabilities, and if GPT-N+1 could do this task, but GPT-N couldn't, we wouldn't tell OpenAI to be particularly careful about the release of GPT-N+1 as a result. However, it makes a useful test case for elicitation methods, since there are many strategies for deciding what path to take and we can create a scale of difficulty by choosing different articles to navigate to/from.

We'll build a rudimentary agent that can play the wikipedia Game. Then, in section [3.4.4], we'll learn both general and specific methods to improve our LLM agent until it becomes adept at the Wikipedia Game (so don't worry if your agent starts out relatively poorly). 

## Quick Intro to the Wikipedia API


Our agent will interact with Wikipedia by making tool calls to the [Wikipedia API](https://wikipedia.readthedocs.io/en/latest/quickstart.html), which is simple to use. We will only need to learn the following key functions for the game. 

1. `wikipedia.page()` - Returns a `WikipediaPage` object, which contains various attributes and methods to access page content. (See [page docs](https://wikipedia-api.readthedocs.io/en/latest/API.html#wikipediapage) for these attributes.)
2. `wikipediaPage.title()` - Returns the title of the page
3. `wikipediaPage.contents()` - Returns the full text content of the page (this can be very long, make sure to take snippets when possible to not use up the context window of the LLM)
4. `wikipediaPage.summary()` - Returns a summary of the page (i.e. the introductory text of the Wikipage before the first section title).
5. `wikipediaPage.links()` - Returns a list of all links as strings


<details><summary> Aside: Wikipedia API content can be weird!</summary>

The wikipedia API often outputs content in unintuitive ways. For example, articles that are essentially just a big list become near useless, since the content omits the list (for example, see the wikipedia API content for <a href = "https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population">List of countries and dependencies by population</a>). Another issue that you might encounter is that the API formats mathematical expressions in $\LaTeX$ pretty poorly (for example, see the wikipedia API content for <a href = "https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">Kullback-Leibler divergence</a>). This is why it's important to determine what content the wikipedia API produces when `.content` is called — and why you want to make sure you're testing a large diversity of wikipedia articles.

</details>

<details><summary> Aside: Wikipedia "summaries" can be long!</summary>

The wikipedia API accesses summaries of pages by presenting all the information before the first titled section. For certain (generally obscure) wikipedia pages, this summary itself can be extremely long, and contain lots of information that is unnecessary to determine the key information about the page the model should be trying to access. We'll handle this later when it comes up by truncating wikipedia's summary to just the first ~1000 characters

</details>

Run the following code to see how these wikipedia API functions work!

```python
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
```

Now run these two lines (you should see a `DisambiguationError` for the first, and a `PageError` for the second):

```python
page = wikipedia.page("Python")
```

```python
page = wikipedia.page("Animalss", auto_suggest=False)
```

We can handle these errors using the following code:

```python
# Fixes PageError by allowing redirects
page = wikipedia.page("Animalss", redirect=True)
print(page.title)

# Fixes DisambiguationError by selecting the first option

try:
    page = wikipedia.page("Python")
except DisambiguationError as e:
    page = wikipedia.page(e.options[0])
print(page.title)
```

The above code gives a `DisambiguationError` because the title "Python" can correspond to multiple pages. Then there is a `PageError` for "Animalss" as there is no Wikipedia name with that title.

To handle these errors, we have implemented a simple function `get_page` for you to get the page object for a particular page title. This handles `RedirectError` (and in some cases `PageError`) by setting `redirect=True`. The function also handles `DisambiguationError` by choosing the first option in the list of potential pages we could be referring to.

We handle any `PageError` not fixed by the above by setting `auto_suggest=True`, and letting wikipedia guess at the page we mean (this is a last resort, and hopefully won't be necessary).

<details><summary>What do <code>redirect</code> and <code>auto_suggest</code> do?</summary>

**Redirect**

The keyword `redirect` tells the API to allow Wikipedia to provide redirections. This happens when you reference an article in a manner which is slightly different than how it is stored in Wikipedia. This rarely happens when we will use the wikipedia API, as we will access pages based on how they are stored in Wikipedia, but as an example:
```python
page = wikipedia.page("huMan", redirect = True, auto_suggest=False)
```
will return a `WikipediaPage` object for the "Human" page. However,
```python
page = wikipedia.page("huMan", redirect=False, auto_suggest=False)
```
will return a `PageError` (since there is a page called "Human" but not "huMan"). The Wikipedia API will generally access the correct page if there is a capitalization issue on the first letter, but a capitalization error in the middle of the word will raise an error (unless `redirect=True`).

<br>

**Auto suggest**

The keyword `auto_suggest` tells the API to allow Wikipedia to provide suggestions. This allows a lot more than `redirect` does, since `redirect` is only for the "obvious" cases (e.g. "huMan" → "Human", "U.S. President" → "President of the United States", etc.). When `auto_suggest` is true, it would allow something like "president of states" → "President of the United States", "gogle" → "Google"; both of which would raise an error if `redirect = True, auto_suggest = False`.

However, `auto_suggest` can sometimes be *too* permissive and lead to errors, for example:

```python
page = wikipedia.page("Human", redirect= False, auto_suggest=True)
```
will return a `WikipediaPage` object for the "Man" page. This is clearly not what we were trying to access, and the `auto_suggest` has gotten carried away in this case.

If `redirect = True` and `auto_suggest=True`, then `auto_suggest` takes priority.
</details>



```python
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
```

### Exercise - Get permitted links from a wikipedia page
```c
Difficulty: 🔴⚪⚪⚪⚪
Importance: 🔵🔵⚪⚪⚪

You should spend up to 5-10 mins on this exercise.
```

When you get the links from a page using `page.links`, this will include every possible Wikipedia link that is accessible from the HTML on that page, including those that are not in the main page content (e.g. links in sidebars, links in footnotes etc.), which are either irrelevant or not permitted by the rules of the Wiki game. Write a simple `get_permitted_links` function, that only returns the links that can be found inside the main content. The resulting list of permitted links should be about a third as long as the list of links from the wikipedia API (with more variance for shorter articles as you would expect). 
<!-- When writing this function, if you manage to get the links in a very effective way, then do that. But remember that Wikipedia is written by a large number of different contributors, often adhering to inconsistent stylings (especially for smaller articles). We just need to get something that **works well enough**. Put more time into doing this effectively if you want at the end, but as soon as something plausibly works, you should move on.

<img src="https://imgs.xkcd.com/comics/code_lifespan_2x.png" width="400px" style = "margin-left: auto; margin-right: auto;display:block"></img> -->

```python
def get_permitted_links(current_page: WikipediaPage) -> list[str]:
    """
    Get "permitted" links (i.e. links that are in the content of the page) from a Wikipedia page.

    Args:
        current_page (WikipediaPage): The current Wikipedia page

    Returns:
        list[str]: A list of permitted links from current_page

    """
    return []
```
## Build WikiGame

### Exercise - Build the WikiGame task
```c
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 25-30 mins on this exercise.
```

Implement the following class that instantiates the wikipedia game. When the model uses tools it will be making calls to this class, so make sure that the functions return messages you're happy for the model to see as a tool response, such as error messages if the tool doesn't work.

Reuse your code from the `get_permitted_links()` function above in this class. Also some of the more uninteresting methods in this class have been implemented for you. You should still try to understand these methods to see how we're implementing the `WikiGame` class (this will help when we build the agent later).

```python
class BaseWikiGame:
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
        Generate the starting instructions for the game.

        Returns:
            dict: The starting instructions. "role" is "system" for system messages.
        """
        return {}

    @property
    def on_page_instruction(self) -> dict:
        """
        Generate instructions for the current page.

        Returns:
            dict: The instructions for the current page. "role" is "user" for user messages.
        """
        return {}

    @property
    def next_step_instruction(self) -> dict:
        """
        Generate instructions for the next step.

        Returns:
            dict: The instructions for the next step. "role" is "user" for user messages.
        """
        return {}

    def get_instructions(self, system: bool, on_page: bool, next_step: bool) -> str:
        """
        Generate instruction messages based on the current game state.

        Args:
            system (bool): Whether to include system instructions.
            on_page (bool): Whether to include on-page instructions.
            next_step (bool): Whether to include next-step instructions.

        Returns:
            list[str]: A list of instruction messages.
        """
        return []

    def check_win(self) -> bool:
        """
        Check if the agent has won the game.

        Returns:
            bool: True if the agent has won, False otherwise.
        """
        return False
```

### Exercise - Build Tools for the Wiki Game
```c
Difficulty: 🔴⚪⚪⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 10-15 mins on this exercise.
```

Fill in the following tool classes that the agent will need to use to accomplish this game.
- For the `get_content_tool`, you should just fill in the description, since we've implemented the functionality for you.
- For the `move_page_tool`, you should implement both the `execute()` function and the `description()` property.

When formatting this tool list, refer back to the solution for you wrote for the arithmetic game, or else the docs are [here](https://platform.openai.com/docs/guides/function-calling).

<details><summary>Why not just use <code>page.links</code> to get a list of links directly?</summary>

We don't just present a list of the accessible links, as this is not very faithful to the wikipedia game. The agent does perform somewhat better if we just give it a list of links, but the task of parsing the content of wikipedia pages and isolating the most important links is where the majority of the challenge of the wikipedia game lies.

</details>
<br>
<details><summary>Notes on the <code>get_content()</code> tool</summary>

The `get_content` function wraps all the texts that correspond to links in `<link></link>` tags (since otherwise they are presented as strings and indistinguishable from normal text, so the agent doesn't know what links to choose). However, since we identify links in the text via their names on wikipedia pages, there are certain articles that will never (or only very rarely) get flagged as links. For example, the page "Python (programming language)" is almost never referenced by its title, instead its almost always referenced by just "Python"; the same is true for towns, which are usually referenced on Wikipedia as e.g. "Juneau, Alaska", but these are almost always referred to as just "Juneau". For this reason, you should avoid having goal pages which are not referenced by their title (or else implement a better version of the function, but beware of simply extracting the HTML source from pages, `wikipediaPage.html` can take a very long time to run, and HTML formatting varies significantly on Wikipedia).

</details>

```python
class get_content_tool():
    name = "get_content"

    @staticmethod
    def execute(task: BaseWikiGame | Any) -> str:
        """
        Get all the content for the wikipedia page you are currently on. Anything which corresponds to a link is wrapped in <link></link> tags.

        Args:
            task (BaseWikiGame | Any): The current task object.

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
        Provides the description of the get_content tool

        Returns:
            dict: The description of the tool for the API
        """
        return {}


class move_page_tool():
    name = "move_page"

    @staticmethod
    def execute(new_page: str, task: Any) -> str:
        """
        Changes your current page to a specified new page which is accessible via a link from the current page. You can only call this function once at a time, as it will take you to a different page.

        Args:
            task (BaseWikiGame): The current task object.
            new_page (str): The title of the new page to move to.

        Returns:
            str: A message indicating the result of the move
        """
        return ""
    @property
    def description(self):
        """
        Provides the description of the move_page tool

        Returns:
            dict: The description of the move_page tool for the API
        """
        return {}


get_content_tool_inst = get_content_tool()
move_page_tool_inst = move_page_tool()
wiki_game_tools = [get_content_tool_inst, move_page_tool_inst]
```
### Exercise - Build a WikiAgent
```c
🔴🔴🔴🔴⚪
🔵🔵🔵🔵🔵

You should spend up to 30-40 mins on this exercise.
```

Now that you have the `WikiGame` class and tools set up, build out a `WikiAgent` that can access these tools and solve the Wikipedia game. Build the agent so that it can be thrown into an agent loop (similar to the one we had for the arithmetic game) without much additional scaffolding. 

There are a few further considerations in this case that we didn't have for the arithmetic game. 

#### Context window considerations

Since the agent will need to read (potentially very long) Wikipedia articles to interact with the game, the length of the context window becomes relevant. GPT-4o and GPT-4o-mini both have context windows of 128k tokens (which corresponds to ~96k words). For reference, the wikipedia page for the United States has around 10k words alone and the agent will often need to visit more than 10 articles in one run of the game, not counting its own output, which eventually adds up to be significant. We'll solve this for now by resetting the messages of the agent every time it reaches a new wikipedia page, and providing an updated `user_message` (and possibly `system_message`) so that the agent can locate itself, and then proceed with the game. We'll address different methods for solving this issue later, you can probably already think of some. So be careful to include the current page and goal page for the agent in the `user_message`.

Since we'll reset the `chat_history` attribute of the agent class each time it reaches a new page, we'll also store a `full_chat_history` property that won't get reset, so we can access the entire run of the game.

#### Providing information to the agent

There shouldn't be much on Wikipedia that the agent is completely unfamiliar with (AI companies *will* have scraped wikipedia), but it may be easily confused with something else, or be an article that was added before the training cutoff, and models can't always accurately recall information in their training data if they only come up once or twice. So you should use the game's get_summary function to provide details of the goal page to the agent in its initial message.


#### Getting output from the agent

In this case we'll have a lot more moving pieces than the `arithmeticGame` agent. In that case it was technically possible to print output directly from the agent loop. In this case, you should print output as it comes up in the agent class. If there's some chance you might not want to see this output, you should use the `verbose` flag to determine whether to print the output or not.

```python
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

    def start(self):
        """
        A function to put the starting instructions in agent.chat_history when the agent starts a new page or starts the game.
        """
        pass
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
        pass
```

### Exercise - Run the task
```c
Difficulty: 🔴⚪⚪⚪⚪
Importance: 🔵🔵⚪⚪⚪

You should spend up to 5 mins on this exercise.
```

Just like we did for the arithmetic agent, you should write an agent loop for the wikipedia agent (in this case you won't need to print output, as we handled it in the agent class so this function should be a very simple loop).

```python
def agent_loop(agent, game, num_loops=10):
    """
    Run the agent loop for a given number of loops

    Args:
        agent (WikiAgent): The agent to run
        game (BaseWikiGame): The game to play
        num_loops (int): The number of loops to run
    """

    pass
```

Your agent should be able to accomplish the following task:


```python
game = BaseWikiGame("Albert Einstein", "Aristotle")
agent = WikiAgent(task=game, tools=wiki_game_tools)
agent_loop(agent, game, 10)
```


Once you've seen that the agent can accomplish the above, try out some different articles.

Make sure to check the messages in the chat history to see the full conversation between the agent and the user, to ensure that the messages that are printed above are faithful to the actual chat history (it can be easy to make minor mistakes that mess up the agent's chat_history).

```python
for message in agent.chat_history:
    try:
        print(f"{str(message.role)}:\n {str(message.content)}")
    except:
        print(f"""{message["role"]}:\n {message["content"]}""")
```

                
''',
        unsafe_allow_html=True,
    )
