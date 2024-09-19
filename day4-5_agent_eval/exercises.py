# %%

# Import the IPython module for automatic reloading
from IPython import get_ipython

# Enable automatic reloading of modules
ipython = get_ipython()
if ipython is not None:
    # Enable autoreload extension
    ipython.magic("load_ext autoreload")
    # Set autoreload to reload all modules (except those excluded) before executing the Python code
    ipython.magic("autoreload 2")

# Inline comment explaining the motivation
# This setup allows for automatic reloading of imported modules,
# which is particularly useful during development and debugging
# as it ensures that the latest version of the code is always used


# %%

import json
import os
import pathlib
import sys

import wikipedia
from wikipedia import WikipediaPage
from wikipedia import DisambiguationError, PageError

# from openai.types.chat.chat_completion_message import ChatCompletionMessage
# from anthropic import Anthropic
from typing import Literal, Optional, Dict, List, Any
from abc import abstractmethod
import math
import re

# %%

import dataclasses
import functools
from typing import Protocol, Callable, TypeVar
import os
import sys
import subprocess
import json

import rich

# Get the resolved absolute filepath of the current file
current_file_path = pathlib.Path(__file__).resolve()

# Add the parent directory to the Python system path
parent_directory = current_file_path.parent
sys.path.append(str(parent_directory))

# Inline comment explaining the motivation
# This allows importing modules from the parent directory,
# which can be useful for organizing related scripts and modules

# %%

import openai
import openai.types
import openai.types.chat

# from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

openai_client = openai.OpenAI()
# %%

import xml.dom.minidom


def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2))


def print_xml(xml_string: str) -> None:
    """
    Pretty prints an XML string.

    Args:
        xml_string (str): The XML string to be pretty printed.

    Returns:
        str: The pretty printed XML string.
    """
    # Parse the XML string into a DOM object
    dom = xml.dom.minidom.parseString(xml_string)

    # Convert the DOM object back to a string with pretty printing
    pretty_xml_as_string = dom.toprettyxml(indent="  ")

    # truncate if insanely long
    max_length_to_show = 256
    if len(pretty_xml_as_string) > max_length_to_show:
        pretty_xml_as_string = pretty_xml_as_string[:max_length_to_show] + "..."

    print(pretty_xml_as_string)


# %%


from typing import Any, Optional, Protocol, TypeVar, ParamSpec
from types import TracebackType

P = ParamSpec("P")
R = TypeVar("R")


class ToolProtocol(Protocol[P, R]):
    """
    Tool protocol is a context manager that provides a callable.

    This allows the tool to have setup / teardown logic.

    The expected usage if needing to wrap this in a higher level like an agent, is:

        # ex: agent or agent manager
        class Agent:

            def __init__(self, tool: ToolProtocol[P, R]):
                self.tool = tool

            def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
                with self.tool as tool:
                    return tool(*args, **kwargs)

    A base class means you then need outside functions to convert to tool protocol etc, that
    should be internal to framework

    """

    # arbitrary name for the tool, needed only because the agent needs some way to
    # disambiguate tools, and likely does better than if this was a hash
    @property
    def name(self) -> str: ...

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

    def __enter__(self) -> "ToolProtocol[P, R]":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        # Implement any cleanup logic here if needed
        return None


# %%

import wikipedia_utils
import tool_use_via_xml_tags as tool_utils

# %%


# %%


@dataclasses.dataclass(frozen=True)
class ToolDefinitionWithCallable:
    tool_definition: tool_utils.ToolDefinition
    callable_fn: Callable


class FunctionCallManager:
    def __init__(self, tool_definitions: List[ToolDefinitionWithCallable]) -> None:

        # Store a mapping of function names to their ToolDefinitionWithCallable objects
        self._tool_map: Dict[str, ToolDefinitionWithCallable] = {
            tool_def.tool_definition.function: tool_def for tool_def in tool_definitions
        }

    def get_tool_definition_list(self) -> tool_utils.ToolDefinitionList:
        # available for easy access by outside when defining everything together with it's corresponding callable
        return tool_utils.ToolDefinitionList(
            tools=[x.tool_definition for x in self._tool_map.values()]
        )

    # TODO(bschoen): Ignore type conversion for now (could try ast literal then fall back to string if can't parse)
    # TODO(bschoen): Ignoring errors for now
    def execute_tool(self, request: tool_utils.ToolUseRequest) -> tool_utils.ToolUseResponse:

        rich.print(f"Executing tool: {request}")

        # Lookup the ToolDefinitionWithCallable for the requested function
        if request.function not in self._tool_map:
            raise ValueError(f"Unknown function: {request.function}")

        tool_def_with_callable = self._tool_map[request.function]

        # Call the function with the provided arguments

        # note: we just assume values are in order instead of enforcing kwargs,
        #       since we want to be able to easily use partial even if we
        #       don't know argument names

        result = tool_def_with_callable.callable_fn(*request.arguments.values())

        # Return the result wrapped in a ToolUseResponse
        response = tool_utils.ToolUseResponse(return_value=str(result))

        rich.print(f"Executed request. Tool response: {response}")

        return response


# %%
import uuid


def test_function_call_manager() -> None:

    def example_callable(a: str, b: str, c: str) -> str:
        return f"{a=} {b=} {c=}"

    func_call_manager = FunctionCallManager(
        [
            ToolDefinitionWithCallable(
                tool_definition=tool_utils.ToolDefinition(
                    function="concatenate",
                    description="Concatenate two strings",
                    arguments=["a", "b"],
                    return_description="returns concatenated string",
                ),
                callable_fn=functools.partial(example_callable, c="example-c"),
            ),
            # test no arguments
            ToolDefinitionWithCallable(
                tool_definition=tool_utils.ToolDefinition(
                    function="generate_uuid",
                    description="Generate a UUID",
                    arguments=[],
                    return_description="returns new UUID",
                ),
                callable_fn=lambda: str(uuid.uuid4()),
            ),
        ],
    )

    request = tool_utils.ToolUseRequest(
        function="concatenate",
        arguments={"a": "example-a", "b": "example-b"},
    )

    # round trip it
    unparsed_request = tool_utils.unparse_tool_use_request(request)
    parsed_request = tool_utils.parse_tool_use_request(unparsed_request)

    assert parsed_request == request

    response = func_call_manager.execute_tool(request)

    assert response.return_value == "a='example-a' b='example-b' c='example-c'"

    request = tool_utils.ToolUseRequest(
        function="generate_uuid",
        arguments={},
    )

    # round trip it
    unparsed_request = tool_utils.unparse_tool_use_request(request)
    parsed_request = tool_utils.parse_tool_use_request(unparsed_request)

    assert parsed_request == request

    response = func_call_manager.execute_tool(request)

    # Attempt to create a UUID object from the return_value
    # (this will throw if it's invalid)
    uuid_obj = uuid.UUID(response.return_value)


test_function_call_manager()

# %%


def test_function_call_manager_replace_value() -> None:

    @dataclasses.dataclass
    class MockBashCommand:
        command: str | None = None

        def set_command(self, command: str) -> None:
            assert command, "command was empty"
            self.command = command

    # here we essentially create a mock executor, whenever the model requests to use
    # the bash command, we store it, can do whatever we want with it, then manufacture
    # the response
    #
    # alternatively could parse it out
    #
    # we just use this to demonstrate something besides `echo`
    #
    mock_bash_command = MockBashCommand()

    func_call_manager = FunctionCallManager(
        [
            ToolDefinitionWithCallable(
                tool_definition=tool_utils.ToolDefinition(
                    function="execute_bash_command",
                    description="Execute bash command",
                    arguments=["bash_command_to_execute_as_string"],
                    return_description="returns the output of the bash command",
                ),
                callable_fn=lambda x: mock_bash_command.set_command(x),
            ),
        ],
    )

    # use model to call the tool
    tool_request = tool_utils.ToolUseRequest(
        function="execute_bash_command",
        arguments={"bash_command_to_execute_as_string": "ls -l"},
    )

    tool_response = func_call_manager.execute_tool(tool_request)

    tool_response = dataclasses.replace(tool_response, return_value="replaced value")

    rich.print(f"tool_response: {tool_response}")


test_function_call_manager_replace_value()

# %%

import ast

func_call_manager = FunctionCallManager(
    [
        ToolDefinitionWithCallable(
            tool_definition=tool_utils.ToolDefinition(
                function="execute_bash_command",
                description="Execute bash command",
                arguments=["bash_command_to_execute_as_string"],
                return_description="returns the output of the bash command",
            ),
            # here we just echo so we can mock the response by overwriting it
            callable_fn=lambda x: x,
        ),
        # TODO(bschoen): Move this to tool_utils
        # this is the tool we use in the example given to the model about
        # how function calling works, and we ask the model to try it out,
        # so we provide the actual function here for convenience (and
        # it also makes sure our own code works)
        ToolDefinitionWithCallable(
            tool_definition=tool_utils.ToolDefinition(
                function="add",
                description="Add two numbers together",
                arguments=["first_number", "second_number"],
                return_description="returns the sum of the two numbers",
            ),
            # we convert from strings before adding
            callable_fn=lambda a, b: ast.literal_eval(a) + ast.literal_eval(b),
        ),
    ],
)

# %%


def parse_message_content_text_from_response(response: openai.types.chat.ChatCompletion) -> str:

    if not len(response.choices) == 1:
        raise ValueError(f"Expected 1 choice, got {len(response.choices)}")

    choice: openai.types.chat.chat_completion.Choice = response.choices[0]

    rich.print(f"{choice.finish_reason=}")

    message: openai.types.chat.ChatCompletionMessage = choice.message

    if not message.content:
        raise ValueError(f"No content in response: {response.model_dump_json(indent=2)}")

    message_content_text: str = message.content

    return message_content_text


# %%


def call_model_and_update_messages(
    client: openai.OpenAI,
    messages: list[openai.types.chat.ChatCompletionMessageParam],
) -> str:

    response = client.chat.completions.create(
        model="o1-mini",
        messages=messages,
        max_completion_tokens=16384,
    )

    rich.print(f"Response: {response.model_dump_json(indent=2)}")

    message_content_text = parse_message_content_text_from_response(response)

    rich.print(f"message_content_text: {message_content_text}")

    # add the message to history
    messages.append(
        {
            "role": "assistant",
            "content": message_content_text,
        },
    )

    return message_content_text


def generate_fresh_message_history() -> list[openai.types.chat.ChatCompletionMessageParam]:

    return [
        {"role": "user", "content": tool_utils.TOOL_USE_EXPLANATION},
    ]


# %%

# TODO(bschoen): I still like this cache store idea

# reset messages
messages = generate_fresh_message_history()

# %%

message_content_text = call_model_and_update_messages(openai_client, messages)

# %%

tool_use_request = tool_utils.parse_tool_use_request(message_content_text)

rich.print(f"tool_use_request: {tool_use_request}")

# %%

tool_use_response = func_call_manager.execute_tool(tool_use_request)

rich.print(f"Tool Use Response: {tool_use_response}")

# %%

tool_use_response_text = tool_utils.unparse_tool_use_response(tool_use_response)

rich.print(f"tool_use_response_text: {tool_use_response_text}")

# %%

# add tool use response to messages (which we'll then send to model)
messages.append(
    {
        "role": "user",
        "content": tool_use_response_text,
    },
)

# %%

# give the result back to the model (which it should use in it's response,
# sending something like `The result of adding 7 and -12 is -5.`, this
# function automatically prints message content so can manually verify)
message_content_text = call_model_and_update_messages(openai_client, messages)


# %%

# now that it's successfully used an example, let's see what it can do
messages.append(
    {
        "role": "user",
        "content": "",
    },
)

# %%

# note: can quickly reset the last two messages and try again
messages = messages[:-2]

# %%

print(json.dumps(messages, indent=2))

# %%

message_content_text = call_model_and_update_messages(openai_client, messages)

# %%


# %%

# first we'll test this independently


@dataclasses.dataclass
class WikiGameState:
    start_page_title: wikipedia_utils.WikiPageTitleStr
    goal_page_title: wikipedia_utils.WikiPageTitleStr
    page_history: list[wikipedia_utils.WikiPageTitleStr]
    current_page: wikipedia.WikipediaPage

    @classmethod
    def from_start_and_goal_titles(
        cls,
        start_page_title: wikipedia_utils.WikiPageTitleStr,
        goal_page_title: wikipedia_utils.WikiPageTitleStr,
    ) -> "WikiGameState":
        return cls(
            start_page_title=start_page_title,
            goal_page_title=goal_page_title,
            page_history=[],
            current_page=wikipedia_utils.get_page(start_page_title),
        )


def summarize_game_state_for_model(state: WikiGameState) -> str:
    # note: model doesn't need page history
    output_str = "<game-state>"

    for key, value in dataclasses.asdict(state).items():

        if key in ["current_page", "page_history"]:
            continue

        output_str += f"<{key}>{value}</{key}>"

    output_str += "</game-state>"

    return output_str


# note: to start we'll just use the agent directly
#
# need to be able to iterate with this step by step to debug anyway
class WikiGamePlayer:

    def __init__(
        self,
        start_page_title: str,
        goal_page_title: str,
    ) -> None:

        self.state = WikiGameState.from_start_and_goal_titles(
            start_page_title,
            goal_page_title,
        )

    def has_reached_goal_page(self) -> bool:

        return self.state.current_page.title == self.state.goal_page_title

    def move_to_page(self, new_page_title: str) -> str:
        """
        Changes your current page to a specified new page which is accessible via a link from the current page. You can only call this function once at a time, as it will take you to a different page.

        The title of the new page you want to move to. This should be formatted the way the title appears on wikipedia (e.g. to move to the wikipedia page for the United States of America, you should enter "United States"). Underscores are not necessary.

        Returns:
            str: A message indicating the result of the move. (note: this is opaque to humans, only used by the model)
        """

        # TODO(bschoen): Check for win condition
        new_page_title_normalized = new_page_title.replace("_", " ")

        rich.print(f"Moving to page: {new_page_title_normalized}")

        if not wikipedia_utils.is_permitted_link(
            current_page=self.state.current_page,
            link=new_page_title_normalized,
        ):

            permitted_links = wikipedia_utils.get_permitted_links(self.state.current_page)

            move_result = f"Failed to move to page `{new_page_title}`, as it was not accessible from the current page. The current page is still: `{self.state.current_page.title}`. The permitted links from the current page are: {permitted_links}"

            rich.print(f"move_result: {move_result}")

            return move_result

        else:

            # update the current page
            self.state.current_page = wikipedia_utils.get_page(new_page_title_normalized)

            # update history
            self.state.page_history.append(self.state.current_page.title)

            # note: we don't repeat the previous page, as we don't want to confuse the model
            move_result = f"Successfully moved to page `{new_page_title}`! Current page is now: `{self.state.current_page.title}`"

            rich.print(f"move_result: {move_result}")

            return move_result

    def get_current_page_content(self) -> str:

        return wikipedia_utils.get_page_content(self.state.current_page)


def test_wiki_game_player() -> None:

    wiki_game_player = WikiGamePlayer(
        start_page_title="Barrack Obama",
        goal_page_title="India",
    )

    # TODO(bschoen): For example, could guess united states here, but not permitted,
    #                so want to allow retry
    #
    # note: we can now essentially start a new game, throwing away message history.
    #
    move_result = wiki_game_player.move_to_page("President of the United States")

    assert not wiki_game_player.has_reached_goal_page()

    # %%

    move_result = wiki_game_player.move_to_page("2004 Indian Ocean earthquake and tsunami")

    assert not wiki_game_player.has_reached_goal_page()

    # check invalid move doesn't work, and doesn't mess up the state
    move_result = wiki_game_player.move_to_page("Isomorphism")

    assert not wiki_game_player.has_reached_goal_page()

    # %%

    move_result = wiki_game_player.move_to_page("India")

    assert wiki_game_player.has_reached_goal_page()


# %%


wiki_game_player = WikiGamePlayer(
    start_page_title="Barrack Obama",
    goal_page_title="India",
)

func_call_manager = FunctionCallManager(
    [
        ToolDefinitionWithCallable(
            tool_definition=tool_utils.ToolDefinition(
                function="get_current_wikipedia_page_content",
                description=r"Get all the content for the wikipedia page you are currently on. Anything which corresponds to a link is wrapped in <link></link> tags.",
                arguments=[],
                return_description="returns full content of a given wikipedia page with links wrapped in <link></link> tags",
            ),
            callable_fn=lambda: wiki_game_player.get_current_page_content(),
        ),
        ToolDefinitionWithCallable(
            tool_definition=tool_utils.ToolDefinition(
                function="move_to_wikipedia_page_from_current_page",
                description=r"""
Changes your current page to a specified new page which is accessible via a link from the current page.

The title of the new page you want to move to should be formatted the way the title appears on wikipedia (e.g. to move to the wikipedia page for the United States of America, you should enter "United States"). Underscores are not necessary.
""",
                arguments=["title_of_page_to_move_to"],
                return_description="returns a message indicating success or failure of the move",
            ),
            callable_fn=lambda x: wiki_game_player.move_to_page(x),
        ),
    ]
)


# to avoid getting insane message length, whenever we successfully move to a new page
# we essentially pop the messages of progress so far, and start the model on the
# new page (this is likely a bit less successful, but massively saves on tokens
# and having to re setup the game)
#
# TODO(bschoen): Can the model do it with just links?


# %%

print(json.dumps(messages, indent=2))
# %%


def generate_wikipedia_race_instructions(
    wiki_game_player: WikiGamePlayer,
    func_call_manager: FunctionCallManager,
) -> str:

    tool_definitions_list = func_call_manager.get_tool_definition_list()
    tool_definitions_list_str = tool_utils.unparse_tool_definitions(tool_definitions_list)

    game_state_str = summarize_game_state_for_model(wiki_game_player.state)

    # TODO(bschoen): avoid loops
    wikipedia_race_instructions = rf"""
    Great! Now let's try it out! We'll be playing "Wikipedia Race" where you try to navigate from the current page to the goal page using the functions provided.

    The current game state is:

    {game_state_str}

    Available tools are:

    {tool_definitions_list_str}

    Remember that you can _only_ move to the current page's <link></link> tags.

    Let's begin!
    """

    return wikipedia_race_instructions


wikipedia_race_instructions = generate_wikipedia_race_instructions(
    wiki_game_player,
    func_call_manager,
)

print(wikipedia_race_instructions)
# %%

messages.append(
    {
        "role": "user",
        "content": wikipedia_race_instructions,
    },
)

# %%

print(json.dumps(messages, indent=2))

# %%

message_content_text = call_model_and_update_messages(openai_client, messages)

# %%

# ???
assert tool_utils.has_tool_use_request(message_content_text)

# %%

tool_use_request = tool_utils.parse_tool_use_request(message_content_text)

rich.print(f"tool_use_request: {tool_use_request}")

# %%

tool_use_response = func_call_manager.execute_tool(tool_use_request)

rich.print(f"Tool Use Response: {tool_use_response}")

# %%

tool_use_response_text = tool_utils.unparse_tool_use_response(tool_use_response)

rich.print(f"tool_use_response_text: {tool_use_response_text}")

# %%

# add tool use response to messages (which we'll then send to model)
messages.append(
    {
        "role": "user",
        "content": tool_use_response_text,
    },
)

# %%

message_content_text = call_model_and_update_messages(openai_client, messages)

# %%


def parse_tool_use_request_and_update_messages(
    message_content_text: str,
    messages: list[openai.types.chat.ChatCompletionMessageParam],
) -> None:

    print("Message Content Text:")
    print_xml(message_content_text)

    tool_use_request = tool_utils.parse_tool_use_request(message_content_text)

    print(f"tool_use_request: {repr(tool_use_request)[:256]}")

    tool_use_response = func_call_manager.execute_tool(tool_use_request)

    print(f"Tool Use Response: {repr(tool_use_response)[:256]}")

    tool_use_response_text = tool_utils.unparse_tool_use_response(tool_use_response)

    print("Tool Use Response Text:")
    print_xml(tool_use_response_text)

    # add tool use response to messages (which we'll then send to model)
    messages.append(
        {
            "role": "user",
            "content": tool_use_response_text,
        },
    )


def reset_player_state_and_messages_from_new_page_if_moved(
    wiki_game_player: WikiGamePlayer,
    func_call_manager: FunctionCallManager,
    messages: list[openai.types.chat.ChatCompletionMessageParam],
) -> None:

    # actually, whenever state is updated, we can reset messages
    if len(wiki_game_player.state.page_history) == 0:
        rich.print("No new page history found, not resetting messages and player state")
        return

    if len(wiki_game_player.state.page_history) != 1:
        raise ValueError(
            f"We expect that this function is called immediately after player moves to a new page (which would give us a page history of length 1), but it's current length is {len(wiki_game_player.state.page_history)}"
        )

    # reset all state, essentially starting a new game from `current_page`
    wiki_game_player.state = WikiGameState.from_start_and_goal_titles(
        start_page_title=wiki_game_player.state.current_page.title,
        goal_page_title=wiki_game_player.state.goal_page_title,
    )

    # now generate new instructions
    new_instructions = generate_wikipedia_race_instructions(
        wiki_game_player=wiki_game_player,
        func_call_manager=func_call_manager,
    )

    # remove messages until we're back to the message before the game started
    target_string = """We'll be playing "Wikipedia Race" """
    while target_string not in messages[-1]["content"]:
        messages.pop()

    # pop the actual one with the target string
    messages.pop()

    # rich.print("Reset messages to before the game started")

    # rich.print("Adding new instructions...")

    messages.append(
        {
            "role": "user",
            "content": new_instructions,
        },
    )

    # print("Updated messages:")
    # print(json.dumps(messages, indent=2))

    # we can now continue playing from here, without sending massive content block for previous pages

    return


# %%


# %%


def step(
    openai_client: openai.OpenAI,
    wiki_game_player: WikiGamePlayer,
    func_call_manager: FunctionCallManager,
    messages: list[openai.types.chat.ChatCompletionMessageParam],
) -> None:

    message_content_text = call_model_and_update_messages(openai_client, messages)

    parse_tool_use_request_and_update_messages(
        message_content_text,
        messages,
    )

    reset_player_state_and_messages_from_new_page_if_moved(
        wiki_game_player=wiki_game_player,
        func_call_manager=func_call_manager,
        messages=messages,
    )


print("Stepping...")
step(
    openai_client=openai_client,
    wiki_game_player=wiki_game_player,
    func_call_manager=func_call_manager,
    messages=messages,
)

if wiki_game_player.has_reached_goal_page():
    rich.print("Player has reached the goal page!")
else:
    rich.print("Player has not reached the goal page yet.")

# %%

print_json(messages)


# %%

parse_tool_use_request_and_update_messages(
    message_content_text,
    messages,
)


reset_player_state_and_messages_from_new_page_if_moved(
    wiki_game_player=wiki_game_player,
    func_call_manager=func_call_manager,
    messages=messages,
)
# %%
