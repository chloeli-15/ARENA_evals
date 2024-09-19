# %%


import re
import xml.etree.ElementTree as ET
from typing import List
import dataclasses


# can be provided to text only model
#
# we stick to the common terms used to align closest to training example
#
# TODO(bschoen): Multiple tools.
#
# note: making the user / assistant messages in tags results in the model not getting the example and responding with the code blocks as well
#
# An example response to the below from o1-mini is:
#
#   <tool-use-request> <function>add</function> <argument>first_number=7</argument> <argument>second_number=-12</argument> </tool-use-request>
#
#   User `<tool-use-response><return>-5</return></tool-use-response>`
#
#   The result of adding 7 and -12 is -5.
#
# o1-preview responds the same way, token for token.
#
# Note: naive XML parsing is explicitly and easily vulnerable, if set up an obstacle
#       or even told it to cleverly access a file with only some arbitrary tool, it probably
#       could
#
TOOL_USE_EXPLANATION = """
You are provided the following protocol:

You will be given a list of tools, the following exchange between a user and an assistant is an example of how they can be used to assist the user:

---

# Example User Message: 

Here are the available tools, please use them to add 14 and 21.

<tool-definition-list>

<tool-definition>
<function>get_wikipedia_page</function>
<description>Returns full content of a given wikipedia page</description>
<argument>title_of_wikipedia_page</argument>
<return>returns full content of a given wikipedia page as a json string</return>
</tool-definition>

<tool-definition>
<function>add</function>
<description>Add two numbers together</description>
<argument>first_number</argument>
<argument>second_number</argument>
<return>returns the numeric value of adding together `first_number` and `second_number`</return>
</tool-definition>

</tool-definition-list>

# Example Assistant Message Requesting Tool Use:

<tool-use-request>
<function>add</function>
<argument>first_number=14</argument>
<argument>second_number=21</argument>
</tool-use-request>

# Example User Response To Tool Use Request:

<tool-use-response>
<return>35</return>
</tool-use-response>

# Example Assistant Response To Tool Use Response:

The result of adding 14 and 21 is 35.

---

To demonstrate you understand the protocol, please use the above tools in the example to add 7 and -12. (hint: your answer should literally be <tool-use-request><function>add</function><argument>first_number=7</argument><argument>second_number=-12</argument></tool-use-request>)
"""


class Tags:
    TOOL_DEFINITION_LIST = "tool-definition-list"
    TOOL_DEFINITION = "tool-definition"
    TOOL_USE_REQUEST = "tool-use-request"
    TOOL_USE_RESPONSE = "tool-use-response"
    FUNCTION = "function"
    DESCRIPTION = "description"
    ARGUMENT = "argument"
    RETURN = "return"


@dataclasses.dataclass(frozen=True)
class ToolDefinition:
    function: str
    description: str
    arguments: List[str]
    return_description: str


@dataclasses.dataclass(frozen=True)
class ToolDefinitionList:
    tools: List[ToolDefinition]


@dataclasses.dataclass(frozen=True)
class ToolUseRequest:
    function: str
    arguments: dict[str, str]


@dataclasses.dataclass(frozen=True)
class ToolUseResponse:
    return_value: str


def parse_tool_definitions(xml_string: str) -> ToolDefinitionList:
    root = ET.fromstring(xml_string)
    tools = []
    for tool in root.findall(Tags.TOOL_DEFINITION):
        tool_dict = ToolDefinition(
            function=tool.find(Tags.FUNCTION).text or "",
            description=tool.find(Tags.DESCRIPTION).text or "",
            arguments=[arg.text or "" for arg in tool.findall(Tags.ARGUMENT)],
            return_description=tool.find(Tags.RETURN).text or "",
        )
        tools.append(tool_dict)
    return ToolDefinitionList(tools)


def add_newlines_to_xml(xml_string: str) -> str:
    """Add a newline before each XML element in a string."""
    # This regex adds a newline before each opening XML tag, except for closing tags and <link> tags
    #
    # Explanation:
    #  - r"<(?!/|link)" - Matches '<' that is not followed by '/' (closing tag) or 'link'
    #  - r"\n<" - Replaces the match with a newline followed by '<'
    #  - .strip() - Removes leading/trailing whitespace from the result
    return re.sub(r"<(?!/|link)", r"\n<", xml_string).strip()


def unparse_tool_definitions(tools: ToolDefinitionList) -> str:
    root = ET.Element(Tags.TOOL_DEFINITION_LIST)
    for tool in tools.tools:
        tool_elem = ET.SubElement(root, Tags.TOOL_DEFINITION)
        ET.SubElement(tool_elem, Tags.FUNCTION).text = tool.function
        ET.SubElement(tool_elem, Tags.DESCRIPTION).text = tool.description
        for arg in tool.arguments:
            ET.SubElement(tool_elem, Tags.ARGUMENT).text = arg
        ET.SubElement(tool_elem, Tags.RETURN).text = tool.return_description

    # Use a custom function to convert to string without modifying <link> tags
    def custom_tostring(elem):
        xml_str = ET.tostring(elem, encoding="unicode")
        # Replace escaped < and > characters within <link> tags
        return re.sub(r"&lt;link&gt;(.*?)&lt;/link&gt;", r"<link>\1</link>", xml_str)

    return add_newlines_to_xml(custom_tostring(root))


def parse_tool_use_request(xml_string: str) -> ToolUseRequest:
    root = ET.fromstring(xml_string)
    arguments = {}
    for arg in root.findall(Tags.ARGUMENT):

        # support either `=` or just the arg, since model passes either sometimes

        key, value = arg.text.split("=", 1) if "=" in arg.text else (arg.text, arg.text)
        arguments[key] = value

    return ToolUseRequest(
        function=root.find(Tags.FUNCTION).text,
        arguments=arguments,
    )


def unparse_tool_use_request(request: ToolUseRequest) -> str:
    root = ET.Element(Tags.TOOL_USE_REQUEST)
    ET.SubElement(root, Tags.FUNCTION).text = request.function
    for key, value in request.arguments.items():
        ET.SubElement(root, Tags.ARGUMENT).text = f"{key}={value}"
    return add_newlines_to_xml(ET.tostring(root, encoding="unicode"))


def has_tool_use_request(xml_string: str) -> bool:
    root = ET.fromstring(xml_string)
    return root.find(Tags.TOOL_USE_REQUEST).text is not None


def parse_tool_use_response(xml_string: str) -> ToolUseResponse:
    root = ET.fromstring(xml_string)
    return ToolUseResponse(return_value=root.find(Tags.RETURN).text or "")


def unparse_tool_use_response(response: ToolUseResponse) -> str:
    root = ET.Element(Tags.TOOL_USE_RESPONSE)
    ET.SubElement(root, Tags.RETURN).text = response.return_value
    return add_newlines_to_xml(ET.tostring(root, encoding="unicode"))


def test_tool_use_parsing_str() -> None:
    """

    Example:

        Parsed tool definitions: ToolDefinitionList(tools=[ToolDefinition(function='greet', description='Greet a person with a custom message', arguments=['name', 'message'], return_description='returns a greeting string'), ToolDefinition(function='search', description='Search for information on a topic', arguments=['query'], return_description='returns search results as a string')])

        Unparsed tool definitions:
        <tool-definition-list><tool-definition><function>greet</function><description>Greet a person with a custom message</description><argument>name</argument><argument>message</argument><return>returns a greeting string</return></tool-definition><tool-definition><function>search</function><description>Search for information on a topic</description><argument>query</argument><return>returns search results as a string</return></tool-definition></tool-definition-list>

        --- Request 1 ---
        Parsed tool use request: ToolUseRequest(function='greet', arguments={'name': 'John Doe', 'message': 'Welcome to our company!'})

        Unparsed tool use request:
        <tool-use-request><function>greet</function><argument>name=John Doe</argument><argument>message=Welcome to our company!</argument></tool-use-request>

        --- Request 2 ---
        Parsed tool use request: ToolUseRequest(function='search', arguments={'query': 'Python programming best practices'})

        Unparsed tool use request:
        <tool-use-request><function>search</function><argument>query=Python programming best practices</argument></tool-use-request>

        --- Response 1 ---
        Parsed tool use response: ToolUseResponse(return_value='Hello, John Doe! Welcome to our company!')

        Unparsed tool use response:
        <tool-use-response><return>Hello, John Doe! Welcome to our company!</return></tool-use-response>

        --- Response 2 ---
        Parsed tool use response: ToolUseResponse(return_value='Search results for "Python programming best practices": 1. Write readable code. 2. Use virtual environments. 3. Follow PEP 8 guidelines.')

        Unparsed tool use response:
        <tool-use-response><return>Search results for "Python programming best practices": 1. Write readable code. 2. Use virtual environments. 3. Follow PEP 8 guidelines.</return></tool-use-response>

    """
    tool_defs_xml = """
    <tool-definition-list>
        <tool-definition>
            <function>greet</function>
            <description>Greet a person with a custom message</description>
            <argument>name</argument>
            <argument>message</argument>
            <return>returns a greeting string</return>
        </tool-definition>
        <tool-definition>
            <function>search</function>
            <description>Search for information on a topic</description>
            <argument>query</argument>
            <return>returns search results as a string</return>
        </tool-definition>
    </tool-definition-list>
    """

    parsed_tools = parse_tool_definitions(tool_defs_xml)
    print("Parsed tool definitions:", parsed_tools)

    unparsed_tools = unparse_tool_definitions(parsed_tools)
    print("\nUnparsed tool definitions:\n", unparsed_tools)

    # Example tool use requests with strings and spaces
    tool_use_requests = [
        """
        <tool-use-request>
            <function>greet</function>
            <argument>name=John Doe</argument>
            <argument>message=Welcome to our company!</argument>
        </tool-use-request>
        """,
        """
        <tool-use-request>
            <function>search</function>
            <argument>query=Python programming best practices</argument>
        </tool-use-request>
        """,
    ]

    for i, request_xml in enumerate(tool_use_requests, 1):
        print(f"\n--- Request {i} ---")
        parsed_request = parse_tool_use_request(request_xml)
        print("Parsed tool use request:", parsed_request)

        unparsed_request = unparse_tool_use_request(parsed_request)
        print("\nUnparsed tool use request:\n", unparsed_request)

    # Example tool use responses with strings
    tool_use_responses = [
        """
        <tool-use-response>
            <return>Hello, John Doe! Welcome to our company!</return>
        </tool-use-response>
        """,
        """
        <tool-use-response>
            <return>Search results for "Python programming best practices": 1. Write readable code. 2. Use virtual environments. 3. Follow PEP 8 guidelines.</return>
        </tool-use-response>
        """,
    ]

    for i, response_xml in enumerate(tool_use_responses, 1):
        print(f"\n--- Response {i} ---")
        parsed_response = parse_tool_use_response(response_xml)
        print("Parsed tool use response:", parsed_response)

        unparsed_response = unparse_tool_use_response(parsed_response)
        print("\nUnparsed tool use response:\n", unparsed_response)


def test_tool_use_parsing_int() -> None:
    """

    Example:

        Parsed tool definitions: ToolDefinitionList(tools=[ToolDefinition(function='add', description='Add two numbers together', arguments=['first_number', 'second_number'], return_description='returns the numeric value of adding together `first_number` and `second_number`')])

        Unparsed tool definitions:
        <tool-definition-list><tool-definition><function>add</function><description>Add two numbers together</description><argument>first_number</argument><argument>second_number</argument><return>returns the numeric value of adding together `first_number` and `second_number`</return></tool-definition></tool-definition-list>

        Parsed tool use request: ToolUseRequest(function='add', arguments={'first_number': '14', 'second_number': '21'})

        Unparsed tool use request:
        <tool-use-request><function>add</function><argument>first_number=14</argument><argument>second_number=21</argument></tool-use-request>

        Parsed tool use response: ToolUseResponse(return_value='35')

        Unparsed tool use response:
        <tool-use-response><return>35</return></tool-use-response>

    """

    # Example tool definitions
    tool_defs_xml = """
    <tool-definition-list>
        <tool-definition>
            <function>add</function>
            <description>Add two numbers together</description>
            <argument>first_number</argument>
            <argument>second_number</argument>
            <return>returns the numeric value of adding together `first_number` and `second_number`</return>
        </tool-definition>
    </tool-definition-list>
    """

    parsed_tools = parse_tool_definitions(tool_defs_xml)
    print("Parsed tool definitions:", parsed_tools)

    unparsed_tools = unparse_tool_definitions(parsed_tools)
    print("\nUnparsed tool definitions:\n", unparsed_tools)

    # Example tool use request
    tool_use_request_xml = """
    <tool-use-request>
        <function>add</function>
        <argument>first_number=14</argument>
        <argument>second_number=21</argument>
    </tool-use-request>
    """

    parsed_request = parse_tool_use_request(tool_use_request_xml)
    print("\nParsed tool use request:", parsed_request)

    unparsed_request = unparse_tool_use_request(parsed_request)
    print("\nUnparsed tool use request:\n", unparsed_request)

    # Example tool use response
    tool_use_response_xml = """
    <tool-use-response>
        <return>35</return>
    </tool-use-response>
    """

    parsed_response = parse_tool_use_response(tool_use_response_xml)
    print("\nParsed tool use response:", parsed_response)

    unparsed_response = unparse_tool_use_response(parsed_response)
    print("\nUnparsed tool use response:\n", unparsed_response)


# test_tool_use_parsing_int()
# test_tool_use_parsing_str()

# %%
