import openai

import enum
import json
from typing import Optional

from workspace.retry_utils import retry_with_exponential_backoff


class Models(enum.Enum):
    O1_PREVIEW = "o1-preview"
    O1_MINI = "o1-mini"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"


def apply_system_format(content: str) -> dict:
    return {"role": "system", "content": content}


def apply_user_format(content: str) -> dict:
    return {"role": "user", "content": content}


def apply_assistant_format(content: str) -> dict:
    return {"role": "assistant", "content": content}


def apply_message_format(user: str, system: Optional[str]) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages

    # TODO(bschoen): Use openai's TypedDict subclasses everywhere


def make_client() -> openai.OpenAI:
    # Prompt the user for an API key
    api_key = input("Please enter your OpenAI API key: ")

    # Validate the API key (basic check for non-empty string)
    if not api_key:
        raise ValueError("API key cannot be empty")

    # Return the client with the provided API key
    return openai.OpenAI(api_key=api_key)


@retry_with_exponential_backoff
def generate_response(
    client: openai.OpenAI,
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
    if not messages:
        assert user is not None, "User message is required if messages is not provided."

    # use user provided messages for state tracking, otherwise create a list
    messages: list[dict] = messages or []

    if user:
        user_message = apply_message_format(user=user, system=system)
        messages.extend(user_message)

    # note: if no `user` message is provided, we assume messages has already been
    #       populated appropriately

    if verbose:
        print(f"Messages: {json.dumps(messages, indent=2)}")

    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if "o1" in model:
        kwargs.pop("max_tokens")
        kwargs["max_completion_tokens"] = 16000

    response = client.chat.completions.create(**kwargs)

    if verbose:
        print(f"Response: {response.model_dump_json(indent=2)}")

    response_content: str = response.choices[0].message.content

    assistant_message = apply_assistant_format(response_content)

    # append message to list
    messages.append(assistant_message)

    return response_content
