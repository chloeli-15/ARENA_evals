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
# 5️⃣ Bonus

> ### Learning objectives
>- Implement additional tools for elicitation
>- Explore some of your own ideas for elicitation
>- Explore some of the current research in elicitation and implement it

## Additional Tool use

### Exercise - Implement a page summary tool
```c
Difficulty:🔴🔴⚪⚪⚪
Importance:🔵🔵⚪⚪⚪

You should spend up to 10-15 mins on this exercise.
```

Implement a tool that allows an agent to get a summary of any page that is accessible from its current page. This imitates a feature on wikipedia where you can see a short summary of a page when you hover over the link to it. You could either implement this tool so that the agent can just read the summary, or you can modify the `move_page` tool, so that the agent sees a summary of the page it wants to move to, and can then make a decision whether to ultimately move page.


```python
class GetAccessiblePageSummaryTool():
    """
    Implements a tool that allows the agent to get the summary of a Wikipedia page (you should use the get_page_summary function from the agent class)
    """

    name = "get_accessible_page_summary"

    @staticmethod
    def get_page_summary(task: Any, page_title: str) -> str:
        """
        Get summary of a wikipedia page, to the last full stop within the first 500 characters. This is used to give a brief overview of the page to the agent.

        Args:
            page (str): The Wikipedia page title.
            task (Any): The current task object.

        Returns:
            str: The summary of the Wikipedia page.
        """

        return ""

    @property
    def description(self):
        """
        Provides the description of the get_page_summary tool

        Returns:
            dict: The description of the tool for the API
        """
        return {}


get_accessible_page_summary_tool_inst = GetAccessiblePageSummaryTool()
wiki_game_tools = [get_content_tool_inst, move_page_tool_inst, test_path_tool_inst, get_accessible_page_summary_tool_inst]
```

<details><summary>Solution:</summary>

```python
class GetAccessiblePageSummaryTool():
    """
    Implements a tool that allows the agent to get the summary of a Wikipedia page (you should use the get_page_summary function from the agent class)
    """

    name = "get_accessible_page_summary"

    @staticmethod
    def get_page_summary(task: Any, page_title: str) -> str:
        """
        Get summary of a wikipedia page, to the last full stop within the first 500 characters. This is used to give a brief overview of the page to the agent.

        Args:
            page (str): The Wikipedia page title.
            task (Any): The current task object.

        Returns:
            str: The summary of the Wikipedia page.
        """

        page = task.get_page(page_title)
        if page in task.get_permitted_links():
            summary = page.content[:500]
            last_period_index = summary.rfind(".")
            return (
                summary[: last_period_index + 1] if last_period_index != -1 else summary
            )
        else:
            return "This page is not accessible from the current page."

    @property
    def description(self):
        """
        Provides the description of the get_page_summary tool

        Returns:
            dict: The description of the tool for the API
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Get the summary of a wikipedia page you are considering moving to, to the last full stop within the first 500 characters. The page needs to be accessible via a link from the current page. Anything which corresponds to a link you can select will be wrapped in <link></link> tags.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page_title": {
                            "type": "object",
                            "description": "The wikipedia page you want to get the summary of.",
                        }
                    },
                    "required": ["page_title"],
                },
            },
        }
```

</details>

Test your agent on the following game (or come up with your own):

```python
wiki_game_tools = [get_content_tool_inst, move_page_tool_inst, test_path_tool_inst, get_accessible_page_summary_tool_inst]
agent = WikiAgentReAct(game, model="gpt-4o-mini", tools = wiki_game_tools)
game = WikiGameHistory("William Pitt the Younger", "Central Vietnam", tools = wiki_game_tools)
```

### Exercise - Implement an arbitrary page summary/content tool
```c
Difficulty:🔴⚪⚪⚪⚪
Importance:🔵🔵⚪⚪⚪

You should spend up to 5-10 mins on this exercise.
```

Now implement a tool that allows the agent to suggest any wikipedia page, and get a brief summary of it. This may be helpful for the agent to formulate plans into the future.

```python
class GetAnyPageContent():
    """
    Implements a tool that allows the agent to get a summary of any Wikipedia page (with no links wrapped in link tags).
    """

    name = "get_any_page_content"

    @staticmethod
    def execute(task: Any, page_title: str | None = None) -> str:
        """
        Get the content of any wikipedia page

        Also provides current page content if no page_title is provided.

        Args:
            page_title (str): The title of the Wikipedia page

        Returns:
            str: The content of the page (not wrapped in link tags).
        """
        return ""

    @property
    def description(self):
        """
        Provides the description of the get_any_page_content tool

        Returns:
            dict: The description of the tool for the API
        """
        return {}

get_any_page_content_tool_inst = GetAnyPageContent()
wiki_game_tools = [get_content_tool_inst, move_page_tool_inst, test_path_tool_inst, get_any_page_content_tool_inst]
```

<details><summary>Solution:</summary>

```python
class GetAnyPageContent():
    """
    Implements a tool that allows the agent to get the content of any Wikipedia page (not wrapped in link tags).
    """

    name = "get_any_page_content"

    @staticmethod
    def execute(task: Any, page_title: str | None = None) -> str:
        """
        Get the content of any wikipedia page

        Also provides current page content if no page_title is provided.

        Args:
            page_title (str): The title of the Wikipedia page

        Returns:
            str: The content of the page (not wrapped in link tags).
        """
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

    @property
    def description(self):
        """
        Provides the description of the get_any_page_content tool

        Returns:
            dict: The description of the tool for the API
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Get all the content for any wikipedia page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page_title": {
                            "type": "object",
                            "description": "The wikipedia page you want to get the content of.",
                        }
                    },
                    "required": ["page_title"],
                },
            },
        }
```

</details>

Test your agent on the following game:

```python
agent = WikiAgentReAct(game, model="gpt-4o-mini", tools = wiki_game_tools)
game = WikiGameHistory("49th Fighter Training Squadron", "Rosenhan experiment", tools = wiki_game_tools)
agent_loop_ReAct(game, agent, 30)
```

### Exercise - Implement additional rules
```c
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵⚪⚪⚪⚪
```

Allow the game to have additional rules. Some suggestions are a "No country" rule, and a "No articles above a given length" rule, but feel free to add more if you think of any others. With all of our elicitation methods, the agent generally only fails if the path is impossible or unreasonably hard. To implement a no country rule, you may want to use the wikipedia API's "categories" attribute for `WikipediaPage` objects.

```python
class WikiGameRules(WikiGameHistory):
    """
    Inherits from WikiGameHistory and adds the ability to store and display the rules of the game.

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

        get_instructions(system: bool, on_page: bool, next_step: bool) -> str: Generate instruction messages based on the current game state (inherited)

        check_win() -> bool: Check if the game has been won (inherited)
    """

    def __init__(self, starting_page: str, goal_page: str, rules: List[Literal["no countries", "no pages above length 30000"]], tools=None):
        super().__init__(starting_page, goal_page, tools)
        self.rules = rules if rules else None

    def system_instruction(self):
        """
        Provide improved starting instructions for the game. Includes the rules.

        Returns:
            dict: The starting instructions. "role" is "system" for system messages.
        """
        return {}

    def on_page_instruction(self):
        """
        Provide improved instructions for the current page.

        Returns:
            dict: The instructions for the current page. "role" is "user" for user messages.
        """
        return {}
```

<details><summary>Solution:</summary>

```python
class WikiGameRules(WikiGameReAct):
    """
    Inherits from WikiGameHistory and adds the ability to store and display the rules of the game.

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

    @property
    def on_page_instruction(self):
        """
        Provide improved instructions for the current page.

        Returns:
            dict: The instructions for the current page. "role" is "user" for user messages.
        """
        return {
            "role" : "user",
            "content" : f"""You are currently on page: {self.current_page.title}. Make sure you start by reasoning about what steps you should take to get to the article on {self.goal_page.title}. When coming up with a strategy, make sure to pay attention to the path you have already taken, and if your current strategy doesn't seem to be working out, try something else. In case you're unsure, {self.goal_page.title} has the following summary:\n[Begin Summary]\n{self.get_page_summary(self.goal_page)}\n[End Summary]\n\nThe pages you've visited so far are: {" -> ".join(self.page_history)}\n\nThe rules of the game are: {",".join(self.rules)}"""
        }
```
</details>

Now modify the move page tool to apply the rules of the game before moving to a new page, and allowing the move only if it's within the rules.

```python
class MovePageTool_rules(MovePageTool):
    """
    Inherits from move_page_tool and adds the ability to check the rules of the game.
    """

    @staticmethod
    def execute(new_page: str, task: Any) -> str:
        """
        Changes your current page to a specified new page which is accessible via a link from the current page. You can only call this function once at a time, as it will take you to a different page.

        Only allow the agent to move if it is permitted by the rules.

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
        Provides the description of the modified move_page tool

        Returns:
            dict: The description of the move_page tool for the API
        """
        return {}
```

<details><summary>Solution:</summary>

```python
class MovePageTool_rules(MovePageTool):
    """
    Inherits from MovePageTool and adds the ability to check the rules of the game.
    """

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
            if "no countries" in task.rules and any("countries in" in category for category in task.get_page(new_page_normalized).categories.lower()):
                return f"Couldn't move page to {new_page}. This page is in the category of countries."
            if "no pages above length 30000" in task.rules and len(task.get_page(new_page_normalized).content) > 30000:
                return f"Couldn't move page to {new_page}. This page is above the maximum length of 30000 characters."
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
```

</details>

### Exercise - Try further elicitation methods
```c
Difficulty:🔴🔴🔴⚪⚪
Importance:🔵🔵🔵⚪⚪
```

Read some of the resources we linked on the Intro to LLM Agents page, and explore some of your own methods to elicit improved performance on the task. If you start seeing diminishing returns from elicitation, come up with ways to make the task harder (or construct your own task).
 



''',
        unsafe_allow_html=True,
    )
