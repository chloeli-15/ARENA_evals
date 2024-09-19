import streamlit as st


def section():
    st.sidebar.markdown(
        r"""
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
# Building a More Complex Agent: WikiGame

> ### Learning objectives
>- Get comfortable building a more complex task
>- Understand how to build a more complex agent
>- Observe the failure modes of a more complex agent


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

- `DisambiguationError`: This was raised because the title "Python" can correspond to multiple pages. 
- `PageError`: This was raised for "Animalss" as there is no Wikipedia page with that title.

We have implemented a simple function `get_page()` for you to get the page object for a particular page title with error handling.

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


<details><summary>What do kwargs <code>redirect</code> and <code>auto_suggest</code> in <code>wikipedia.page()</code> do?</summary>

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

### Exercise - Implement `get_permitted_links()`
```c
Difficulty: 🔴⚪⚪⚪⚪
Importance: 🔵🔵⚪⚪⚪

You should spend up to ~10 mins on this exercise.
```
This is a quick exercise to familarize you with the Wikipedia API.

When you get the links from a page using `page.links`, this will include every possible Wikipedia link that is accessible from the HTML on that page, including those that are not in the main page content (e.g. links in sidebars, links in footnotes etc.), which are irrelevant or not permitted by the rules of the Wiki game. 

Write a simple `get_permitted_links()` function. This should only return the links that can be found inside the main content. The resulting list of permitted links should be about a third as long as the list of links from `page.links`. 

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

<details><summary>Solution:</summary>

```python
def get_permitted_links(current_page: WikipediaPage) -> list[str]:
    """
    Get "permitted" links (i.e. links that are in the content of the page) from a Wikipedia page.

    Args:
        current_page (WikipediaPage): The current Wikipedia page

    Returns:
        list[str]: A list of permitted links from current_page

    """
    all_links = current_page.links
    content = current_page.content
    permitted_links = [link for link in all_links if link in content]
    return permitted_links
```

</details>


## LLM Agent for WikiGame

<img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/refs/heads/main/img/ch3-wiki-task-overview.png" width="1000">

### Exercise - Build the WikiGame task
```c
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 30-35 mins on this exercise.
```

Build the `WikiGame` class that instantiates the wikipedia game. This should contain the following functionalities:
1. Keep track of task states
2. Give task-specifc instructions
3. Task-specific helper functions for calling the Wikipedia API. These less interesting methods have been provided for you, but you should read and understand what they do.


#### Providing information to the agent

While models are trained on most of the Wikipedia content, a particular page may still be confused with something else, or be an article that was added after the training cutoff. Models also can't always accurately recall information in their training data if they only come up once or twice. So you should use the game's `get_summary()` function to provide details of the goal page to the agent in its initial message.


```python
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
        # TODO
        return {}

    @property
    def on_page_instruction(self) -> dict:
        """
        Tell the agent what page they are on and give a summary of the page, formatted as a user prompt.

        Returns:
            dict: The instructions for the current page. The "role" is "user" for user messages.
        """
        # TODO
        return {}

    @property
    def next_step_instruction(self) -> dict:
        """
        Ask the agent "What's the next step?" after making a tool call, formatted as a user prompt.

        Returns:
            dict: The instructions for the next step. The "role" is "user" for user messages.
        """
        # TODO
        return {}

    # ========================= Task State management (to implement) =========================

    def check_win(self) -> bool:
        """
        Check if the agent has won the game.

        Returns:
            bool: True if the agent has won, False otherwise.
        """
        # TODO
        pass

tests.run_wiki_game_tests(WikiGame)

```

<details><summary>Solution:</summary>

```python
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
```

</details>



### Exercise - Build Tools for the WikiGame
```c
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 15-20 mins on this exercise.
```

The basic WikiAgent will need these two tools at minimum to play the game:
1. `GetContentTool`: This returns the full content of the current page, with all the wiki-links wrapped in `<link></link>` tags (as otherwise they are presented as strings and indistinguishable from normal text). As implementing this involves annoying regex, we have done this for you, but you should fill in the `description()` property.
2. `MovePageTool`: This executes moving to a new given page when called and updates the `WikiGame` task state if successful. You should implement both the `execute()` function and the `description()` property.

When formatting this tool list, refer back to the solution for you wrote for the arithmetic game, or the OpenAI docs [here](https://platform.openai.com/docs/guides/function-calling).

<details><summary>Why not just use <code>WikipediaPage.links()</code> to get a list of links directly?</summary>

We don't just present a list of the accessible links, as this is not very faithful to the wikipedia game. The agent does perform somewhat better if we just give it a list of links, but the task of parsing the content of wikipedia pages and isolating the most important links is big part of the challenge of the wikipedia game.

</details>

<details><summary>Caveat for the <code>GetContentTool</code></summary>

The `GetContentTool` wraps all the texts that correspond to links in `<link></link>` tags. However, since we identify links in the text via their names on wikipedia pages, there are certain articles that will never (or only very rarely) get flagged as links. For example, the page "Python (programming language)" is almost never referenced by its title, instead its almost always referenced by just "Python"; the same is true for towns, which are usually referenced on Wikipedia as e.g. "Juneau, Alaska", but these are almost always referred to as just "Juneau". For this reason, you should avoid having goal pages which are not referenced by their title (or else implement a better version of the function, but beware of simply extracting the HTML source from pages, `wikipediaPage.html` can take a very long time to run, and HTML formatting varies significantly on Wikipedia).

</details>

```python
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
        # TODO
        return {}


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
        # TODO
        pass

    @property
    def description(self):
        """
        Provides the description of the MovePageTool

        Returns:
            dict: The description of the MovePageTool for the API
        """
        # TODO
        return {}


get_content_tool_inst = GetContentTool()
move_page_tool_inst = MovePageTool()
wiki_game_tools = [get_content_tool_inst, move_page_tool_inst]
```

<details><summary>Solution:</summary>

```python
class GetContentTool():
    name = "get_content"

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
        Provides the description of the get_content tool

        Returns:
            dict: The description of the tool for the API
        """
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


class MovePageTool():
    name = "move_page"

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
        new_page_normalized = new_page.replace("_", " ")
        if task.is_permitted_link(new_page_normalized):
            task.current_page = task.get_page(new_page_normalized)
            task.page_history.append(task.current_page.title)
            return f"Moving page to {task.current_page.title}"
        else:
            return f"Couldn't move page to {new_page}. This is not a valid link."

    @property
    def description(self):
        """
        Provides the description of the move_page tool

        Returns:
            dict: The description of the move_page tool for the API
        """
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
```

</details>



### Exercise - Build a WikiAgent
```c
🔴🔴🔴🔴⚪
🔵🔵🔵🔵🔵

You should spend up to 40-60 mins on this exercise.
```

We will now build a `WikiAgent` that can use these tools to solve the `WikiGame`. Build the agent so that it can be called via an agent loop, similar to the one we had for the arithmetic game. 

There are a few further considerations in this case that we didn't have for the arithmetic game. 

#### Context window constraint

Since Wikipedia articles could be very long, the length of the LLM's context window becomes a constraint. GPT-4o and GPT-4o-mini both have context windows of 128k tokens (which corresponds to ~96k words). For reference, the wikipedia page for the United States has around 10k words alone and the agent will often need to visit more than 10 articles in one run of the game, not counting its own output, which eventually adds up to be significant. 

We'll solve this for now by simply resetting the messages of the agent every time it reaches a new wikipedia page, and providing an updated set of instructions, so the agent can locate itself in the game. We'll address different methods for solving this issue later, you can probably already think of some. So be careful to include the current page and goal page for the agent in the instruction.

Since we'll reset the `chat_history` attribute of the agent class each time it reaches a new page, we'll also store a `full_chat_history` property that won't get reset, so we can access the entire run of the game.


#### Printing output

The `WikiGame` is a lot longer than the `ArithmeticTask`, with a much larger volume of agent, task and tool messages. If there's some chance you might not want to see this output, you should use the `verbose` parameter to set whether to print the output or not.

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
```

<details><summary>Solution:</summary>

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
```
</details>


### Exercise - Run the task
```c
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵⚪⚪⚪

You should spend up to 10-15 mins on this exercise.
```

Similar to the `ArithmeticAgent`, write an agent loop for the `WikiAgent`.

```python
def agent_loop(agent, game, num_loops=10):
    """
    Run the agent loop for a given number of loops

    Args:
        agent (WikiAgent): The agent to run
        game (WikiGame): The game to play
        num_loops (int): The number of loops to run
    """

    pass
```

<details><summary>Solution:</summary>

```python
def agent_loop(agent, game, num_loops=10):
    """
    Run the agent loop for a given number of loops

    Args:
        agent (WikiAgent): The agent to run
        game (WikiGame): The game to play
        num_loops (int): The number of loops to run
    """
    #agent.start()

    for i in range(num_loops):
        if game.check_win():
            print("Success!")
            return
        agent.run()
```
</details>

Your agent should be able to accomplish the following tasks:


```python
game_1 = WikiGame("Barack Obama", "India")
agent = WikiAgent(task=game_1, tools=wiki_game_tools)
agent_loop(agent, game_1, 30)
```

```python
game_2 = WikiGame("Albert Einstein", "Aristotle")
agent = WikiAgent(task=game_2, tools=wiki_game_tools)
agent_loop(agent, game_2, 30)
```


Once you've seen that the agent can accomplish the above, try out some different articles and see where the agent fails.

Check the messages in the chat history to see the full conversation between the agent and the user, to ensure that the messages that are printed above are faithful to the actual chat history (it can be easy to make minor mistakes that mess up the agent's chat_history).

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
