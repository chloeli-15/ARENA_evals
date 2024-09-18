import streamlit as st

def section():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#introduction'>Introduction</a></li>
    <li class='margtop'><a class='contents-el' href='#content-learning-objectives'>Content & Learning Objectives</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#1-threat-modeling'>Threat-modeling</a></li>
        <li><a class='contents-el' href='#2-mcq-benchmarks-exploration'>MCQ Benchmarks: Exploration</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
</ul>""", unsafe_allow_html=True)
    
    st.markdown(
r'''
# Intro to API Calls

> ##### Learning objectives
> 
> - Learn the basic of API calls, including how to format messages, how to generate responses, and how to handle rate limit errors


We will use GPT4o-mini to generate and answer questions via the OpenAI chat completion API, which allows us to programmatically send user messages and receive model responses. 

* Read this very short [chat completions guide](https://platform.openai.com/docs/guides/chat-completions) on how to use the OpenAI API. 

### Messages
In a chat context, instead of continuing a string of text, the model reads and continues a conversation consisting of a history of texts, which is given as an array of **message** objects. The API accepts this input via the `message` parameter. Each message object has a **role** (either `system`, `user`, or `assistant`) and content. LLMs are finetuned to recognize and respond to messages from the roles correspondingly: 
- The `system` message gives overall instructions for how the LLM should respond, e.g. instructions on the task, context, style, format of response, or the character it should pretend to be. 
- The `user` message provides a specific question or request for the assistant to respond to.
- The `assistant` message is the LLM's response. However, you can also write an `assistant` message to give examples of desired behaviors (i.e. **few-shot examples**). 


When giving this input to the API via the `message` parameter, we use the syntax of a list of dictionaries, each with `role` and `content` keys. Standard code for API call looks like the code below. Run this code:

```python
# Configure your API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)

print(response, "\n")  # See the entire ChatCompletion object
print(response.choices[0].message.content)  # See the response message only
```

<details>
<summary>Help - Configure your OpenAI API key</summary>

If you haven't already, go to https://platform.openai.com/ to create an account, then create a key in 'Dashboard'-> 'API keys'.
A convenient set-up in VSCode is to have a `.env` file located at `ARENA_3.0/.env` containing:

```c
OPENAI_API_KEY = "your-openai-key"
ANTHROPIC_API_KEY = "your-anthropic-key"
```

Note that the names 'OPENAI_API_KEY' and 'ANTHROPIC_API_KEY' must be those exactly. 

Install `dotenv` library and import the `load_dotenv` function, which will read in variables in `.env` files. 

</details>


We will provide these helper functions for formatting `system`, `user` and `assistant` messages. Run these below:

```python
def apply_system_format(content : str) -> dict:
    return {
        "role" : "system",
        "content" : content
    }

def apply_user_format(content : str) -> dict:
    return {
        "role" : "user",
        "content" : content
    }

def apply_assistant_format(content : str) -> dict:
    return {
        "role" : "assistant",
        "content" : content
    }

def apply_message_format(user : str, system : Optional[str]) -> List[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages
```
The `client.chat.completions.create()` function takes multiple parameters, which determine how the model returns the output. Refer back to the [API docs](https://platform.openai.com/docs/api-reference/chat) when needed. We will summarize the 3 most important parameters: 
- `model` (required): This is the model used to generate the output. (Find OpenAI's model names [here](https://platform.openai.com/docs/models).)
- `message` (required): This is list of messages comprising the conversation so far (i.e. the main input that the model will respond to).  
- `temperature`: Informally, this determines the amount of randomness in how output tokens are sampled by the model. Temperature 0 means the token with the maximum probability is always chosen and there is no sampling/randomness (i.e. greedy sampling), whereas higher temperature introduces more randomness (See [\[1.1\] Transformers from Scratch: Section 4](https://arena-ch1-transformers.streamlit.app/[1.1]_Transformer_from_Scratch) to understand temperature in more details). By default, `temperature = 1`, which generally works best empirically.

A standard Chat Completions API response looks as follows. This is returned as a ChatCompletionMessage object. The reply string can be accessed via `response.choices[0].message.content`.

<div style="font-family: monospace; white-space: pre; background-color: #f4f4f4; padding: 10px; border-radius: 5px; font-size: 0.8em;">{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4o-mini",
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "\n\nHello there, how may I assist you today?",
    },
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}</div>

## Exercise - Generate response via API 
```c
Difficulty: 🔴⚪⚪⚪⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 5-10 minutes on this exercise.
```

You should fill in the `generate_response` function below. It should:

* Receive the `model`, and either formatted `messages` (a list of dictionaries with `role` and `content` keys), or unformatted `user` and/or `system` strings as input
* Return the model output as a string
* The function should print the messages if `verbose=True`.

```python
@retry_with_exponential_backoff
def generate_response(client, 
                      model: str, 
                      messages:Optional[List[dict]]=None, 
                      user:Optional[str]=None, 
                      system:Optional[str]=None, 
                      temperature: float = 1, 
                      verbose: bool = False) -> Optional[str]:
    """
    Generate a response using the OpenAI API.

    Args:
        model (str): The name of the OpenAI model to use (e.g., "gpt-4o-mini").
        messages (Optional[List[dict]]): A list of message dictionaries with 'role' and 'content' keys. 
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

    if model != "gpt-4o-mini":
        warnings.warn(f"Warning: The model '{model}' is not 'gpt-4o-mini'.")
        
    # TODO: Implement the function

```

```python
response = generate_response(client, "gpt-4o-mini", user="What is the capital of France?")
print(response)
```

<details> 
<summary>Solution</summary>

```python
@retry_with_exponential_backoff
def generate_response(client,
                      model: str, 
                      messages:Optional[List[dict]]=None, 
                      user:Optional[str]=None, 
                      system:Optional[str]=None, 
                      temperature: float = 1, 
                      verbose: bool = False) -> str:
    """
    Generate a response using the OpenAI API.

    Args:
        model (str): The name of the OpenAI model to use (e.g., "gpt-4o-mini").
        messages (Optional[List[dict]]): A list of message dictionaries with 'role' and 'content' keys. 
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
    if model != "gpt-4o-mini":
        warnings.warn(f"Warning: The model '{model}' is not 'gpt-4o-mini'.")
        
    if messages is None:
        messages = apply_message_format(user=user, system=system)

    if verbose:
        for message in messages:
            print(f"{message['role'].upper()}:\n{message['content']}\n")
    
    # API call
    try:
        response = client.chat.completions.create(
        model=model, 
        messages=messages, 
        temperature=temperature
        )

    except Exception as e:
        print("Error in generation:", e)
    
    return response.choices[0].message.content
```
</details>


## Exercise (optional) - Retry with exponential back-off 

```c
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 10-15 minutes on this exercise.
```

LLM APIs impose a limit on the number of API calls a user can send within a period of time — either tokens per minute (TPM) or requests per day (RPD). See more info [here](https://platform.openai.com/docs/guides/rate-limits). Therefore, when you use model API calls to generate a large dataset, you will most likely encounter a `RateLimitError`. 

The easiest way to fix this is to retry your request with a exponential backoff. Retry with exponential backoff means you perform a short sleep when a rate limit error occurs and try again. If the request is still unsuccessful, increase your sleep length and repeat this process until the request succeeds or a maximum number of retries is hit. 

You should fill in the decorator function `retry_with_exponential_backoff` below. This will be used to decorate our API call function. It should:
* Try to run `func` for `max_retries` number of times, then raise an exception if it's still unsuccessful
* Perform a short sleep when `RateLimitError` is hit
* Increment the sleep time by a `backoff_factor`, which varies by a small random jitter, each time a `RateLimitError` is hit. Your sleep time should increase exponentially with the number of rate limit errors, with `backoff_factor` as the exponential base. 
* Raise any other errors as normal 

Note: In practice, you do not need to implement this yourself, but can import it from `tenacity` or `backoff` library (see [here](https://platform.openai.com/docs/guides/rate-limits/error-mitigation) after completing the exercise).

```python
def retry_with_exponential_backoff(
    func,
    max_retries=20,
    initial_sleep_time=1,
    backoff_factor: float = 1.5,
    jitter: bool = True,
):
    """
    Retry a function with exponential backoff.

    This decorator retries the wrapped function in case of rate limit errors,
    using an exponential backoff strategy to increase the wait time between retries.

    Args:
        func (callable): The function to be retried.
        max_retries (int): Maximum number of retry attempts. Defaults to 20.
        initial_sleep_time (float): Initial sleep time in seconds. Defaults to 1.
        backoff_factor (float): Factor by which the sleep time increases after each retry.
            Defaults to 1.5.
        jitter (bool): If True, adds a small random variation to the backoff time.
            Defaults to True.

    Returns:
        callable: A wrapped version of the input function with retry logic.

    Raises:
        Exception: If the maximum number of retries is exceeded.
        Any other exception raised by the function that is not a rate limit error.

    Note:
        This function specifically handles rate limit errors. All other exceptions
        are re-raised immediately.
    """

    def wrapper(*args, **kwargs):
        # TODO: Implement the retry logic here
        raise Exception(f"Maximum retries {max_retries} exceeded")

    return wrapper
```

<details>
<summary>Help - I have no idea what to do! </summary>

Here's the rough outline of what you need to do:

1. Initialize the `sleep_time` variable with the `initial_sleep_time`.
2. Use a `for` loop to attempt the function call up to `max_retries` times.
3. Inside the loop, use a `try-except` block to catch exceptions.
4. If a rate limit error occurs:
    - Increase the `sleep_time` using the `backoff_factor` and `jitter`.
    - Use `time.sleep()` to pause before the next attempt.
5. For any other exception, re-raise it immediately.
6. If all retries are exhausted, raise a custom exception.

</details>

<details>
<summary>Solution</summary>

```python
def retry_with_exponential_backoff(
    func,
    max_retries=20,
    initial_sleep_time=1,
    backoff_factor: float = 1.5,
    jitter: bool = True,
):
    """
    Retry a function with exponential backoff.

    This decorator retries the wrapped function in case of rate limit errors,
    using an exponential backoff strategy to increase the wait time between retries.

    Args:
        func (callable): The function to be retried.
        max_retries (int): Maximum number of retry attempts. Defaults to 20.
        initial_sleep_time (float): Initial sleep time in seconds. Defaults to 1.
        backoff_factor (float): Factor by which the sleep time increases after each retry.
            Defaults to 1.5.
        jitter (bool): If True, adds a small random variation to the backoff time.
            Defaults to True.

    Returns:
        callable: A wrapped version of the input function with retry logic.

    Raises:
        Exception: If the maximum number of retries is exceeded.
        Any other exception raised by the function that is not a rate limit error.

    Note:
        This function specifically handles rate limit errors. All other exceptions
        are re-raised immediately.
    """

    def wrapper(*args, **kwargs):       
        sleep_time = intial_sleep_time

        for _ in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "rate_limit_exceeded" in str(e):
                    sleep_time *= backoff_factor * (1 + jitter * random.random())
                    time.sleep(sleep_time)
                else:
                    raise
        raise Exception(f"Maximum retries {max_retries} exceeded")

    return wrapper
```

   
</details>


''', unsafe_allow_html=True)
