import anthropic

import enum
from typing import Optional

from workspace.retry_utils import retry_with_exponential_backoff


class Models(enum.Enum):
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"


def apply_system_format(content: str) -> anthropic.types.MessageParam:
    return {"role": "system", "content": content}


def apply_user_format(content: str) -> anthropic.types.MessageParam:
    return {"role": "user", "content": content}


def apply_assistant_format(content: str) -> anthropic.types.MessageParam:
    return {"role": "assistant", "content": content}


def apply_message_format(user: str, system: Optional[str]) -> list[anthropic.types.MessageParam]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def make_client() -> anthropic.Anthropic:
    # Prompt the user for an API key
    api_key = input("Please enter your Anthropic API key: ")

    # Validate the API key (basic check for non-empty string)
    if not api_key:
        raise ValueError("API key cannot be empty")

    # Return the client with the provided API key
    return anthropic.Anthropic(api_key=api_key)


def get_response_content_text(response: anthropic.types.Message) -> str:
    if not response.content:
        raise ValueError(f"No response content: {response.model_dump_json(indent=2)}")

    assert len(response.content) == 1, f"Expected 1 content block, got {len(response.content)}"

    response_content: anthropic.types.ContentBlock = response.content[0]

    assert response_content.type == "text", f"Expected text content, got {response_content.type}"

    response_content_text: str = response_content.text

    return response_content_text


@retry_with_exponential_backoff
def generate_response(
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    messages: Optional[list[dict]] = None,
    user: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 1,
    verbose: bool = False,
) -> str:
    """
    Generate a response using the OpenAI API.

    Args:
        model (str): The name of the OpenAI model to use (e.g., "gpt-4o-mini").
        messages (Optional[list[dict]]): A list of message dictionaries with 'role' and 'content' keys.
                                         If provided, this takes precedence over user and system args.
        user (Optional[str]): The user's input message. Used if messages is None.
        system (Optional[str]): The system message to set the context. Used if messages is None.
        temperature (float): Controls randomness in output. Higher values make output more random. Default is 1.
        verbose (bool): If True, prints the input messages before making the API call. Default is False.

    Returns:
        str: The generated response from the OpenAI model.

    Note:
        - If both 'messages' and 'user'/'system' are provided, 'messages' takes precedence.
        - The function uses a retry mechanism with exponential backoff for handling transient errors.
        - The client object should be properly initialized before calling this function.
    """
    assert not system, "System not supported"
    if not messages:
        assert user is not None, "User message is required if messages is not provided."

    messages: list[anthropic.types.MessageParam] = messages or []

    if user:
        user_message = apply_message_format(user=user, system=system)
        messages.extend(user_message)

    response: anthropic.types.Message = client.messages.create(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if verbose:
        print(f"Response: {response.model_dump_json(indent=2)}")

    response_content_text = get_response_content_text(response)

    assistant_message = apply_assistant_format(response_content_text)

    # append message to list
    messages.append(assistant_message)

    return response_content_text
