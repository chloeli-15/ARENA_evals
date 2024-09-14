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
r"""
# MCQ Benchmarks

> ##### Learning objectives
> 
> - Understand what benchmarks are, and what makes a good benchmark
> - Understand the pros/cons of MCQ benchmarks
> - Learn how to use LLMs to generate and improve questions


Broadly speaking, a **benchmark** refers to a procedure for evaluating the performance of models on a particular task. This typically includes specifying a dataset, but also specifications of a task or set of tasks to be performed on the dataset and how to score the result, so that the benchmark can be used in a principled, standardized way that gives common ground for comparison.

**What makes a good benchmark?** This [article](https://www.safe.ai/blog/devising-ml-metrics) by Dan Hendrycks and Thomas Woodside gives a good summary of the key qualities of a good benchmark. We copied the key points relevant to an MCQ benchmark here:

* Include clear floors and ceilings for the benchmark. It’s useful to include human performance as well as random performance, as this helps orient people trying to understand the benchmark. 
* Design the measure so that performance is considered low. If performance is already at 90%, researchers will assume there isn’t much of a problem that they can make progress on. Instead, zoom in on the structures that are most difficult for current systems, and remove parts of the benchmark that are already solved.
* Benchmark performance should be described with as few numbers as possible. In often cases, researchers do not want to report eight different numbers, and would rather report just one.
* A benchmark should be comparable across time. Benchmarks should be designed such that they do not need to be updated.
* It’s nice if it’s possible for a ML model to do better than expert performance on the benchmark. For instance, humans aren’t that good at quickly detecting novel biological phenomena (they must be aware of many thousands of species), so a future model’s performance can exceed human performance. This makes for a more interesting task.

<details><summary><b>What are the pros and cons of MCQs?</b></summary>

The advantages of MCQ datasets, compared evaluating free-form response or more complex agent evals, are mainly to do with cost and convenience:

* Cheaper and faster to build
* Cheaper and shorter to run (compared to long-horizon tasks that may take weeks to run and have thousands of dollars of inference compute cost)
* May produce more replicable results (compared to, for example, training another model to evaluate the current model, as each training run could give slightly different results) 

The big disadvantage is that it is typically a worse (less realistic, less accurate) proxy for what you want to measure.

</details>

## Intro to API Calls

We will use GPT4o-mini to generate and answer questions via the OpenAI chat completion API, which allows us to programmatically send user messages and receive model responses. Read this very short [chat completions guide](https://platform.openai.com/docs/guides/chat-completions) on how to use the OpenAI API. 

### Messages
In a chat context, instead of continuing a string of text, the model reads and continues a conversation consisting of a history of texts, which is given as an array of **message** objects. The API accepts this input via the `message` parameter. Each message object has a **role** (either `system`, `user`, or `assistant`) and content. LLMs are finetuned to recognize and respond to messages from the roles correspondingly: 
- The `system` message gives overall instructions that constrain the LLM's responses, e.g. instructions on the task, context, style and format of response. 
- The `user` message provides a specific question or request for the assistant to respond to.
- The `assistant` message is the LLM's response. However, you can also write an `assistant` message to give examples of desired behaviors (i.e. **few-shot examples**). 


When giving this input to the API via the `message` parameter, we use the syntax of a list of dictionaries, each with `role` and `content` keys. Standard code for API call looks like the code below. Run this code:

<details>
<summary>Help - Configure your OpenAI API key</summary>

You should go on https://platform.openai.com/ to create an account, then create a key in 'Dashboard'-> 'API keys'.
A convenient set-up in VSCode is to have a `.env` file containing:

```c
OPENAI_API_KEY = "your-openai-key"
ANTHROPIC_API_KEY = "your-anth-key"
```

Note that the names 'OPENAI_API_KEY' and 'ANTHROPIC_API_KEY' must be those exactly. Then install `dotenv` library and import the `load_dotenv` function, which will read in variables in `.env` files. 

</details>

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



## Exercise - Generate response via LLM 
```c
Difficulty: 🔴⚪⚪⚪⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 5-10 minutes on this exercise.
```

You should fill in the `generate_response` function below. It should:

* Receive the `model`, and either formatted `messages` (a list of dictionaries with `role` and `content` keys), or unformatted `user` and/or `system` strings as input
* Return the model output as a string

```python
@retry_with_exponential_backoff
def generate_response(model: str, messages:Optional[List[dict]]=None, user:Optional[str]=None, system:Optional[str]=None, temperature: float = 1, verbose: bool = False) -> str:
    '''
    Generate a response to the `messages` from the OpenAI API.

    Args:
        model (str): model name
        messages (dict): array of formatted messages with role and content
        user (str): user message (alternative to `messages`)
        system (str): system message (alternative to `messages`)
    '''

    # Initialize messages if not provided
    if messages is None:
        messages = apply_message_format(user=user, system=system)
    
    # Print messages if verbose
    if verbose:
        for message in messages:
            print(f"{message['role'].upper()}:\n{message['content']}\n")
      
    # API call
    response = client.chat.completions.create(
        model=model, 
        messages=messages, 
        temperature=temperature
    )
    return response.choices[0].message.content
```
```python
response = generate_response("gpt-4o-mini", user="What is the capital of France?")
print(response)
```
### Exercise (optional) - Retry with exponential back-off 

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
* Increment the sleep time by a `backoff_factor`, which varies by a small random jitter, each time a `RateLimitError` is hit. Your sleep time should increase exponentially with the number of rate limit errors, with `backoff_factor` as the exponential base. (EXPLAIN WHY WE DO THIS IN AN ASIDE BECAUSE RATELIMITERRORS ARE RAISED WITH EXPONENTIAL BACKOFF BY THE API)
* Raise any other errors as normal 

```python
def retry_with_exponential_backoff(
    func,
    max_retries=20,
    intial_sleep_time=1,
    backoff_factor: float = 1.5,
    jitter: bool = True,
):
    '''
    Retry a function with exponential backoff

    Args:
    func: function to retry
    max_retries: maximum number of retries
    intial_sleep_time: initial sleep time
    backoff_factor: factor to increase sleep time by
    jitter: if True, randomly vary the backoff_factor by a small amount

    '''

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
## Design MC questions

### Exercise - Run eval questions on model
```c
Difficulty: 🔴⚪⚪⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 10-15 minutes on this exercise.
```

In the model-written evals paper, Ethan Perez et al generated a set of model-generated eval datasets on different topics. They are inside the 'anthropic_dataset' folder. Take a look at some of their questions in the `jsonl` file, and run them on the model to see how it responds. 

Note that questions are not the gold standard, but more a proof-of-concept for testing whether models can generate questions at all. See if you have ideas for how to improve these questions.

```python
anthropic_dataset_name = "power-seeking-inclination"
dataset = load_jsonl(f"anthropic_dataset/{anthropic_dataset_name}.jsonl")
question_sample = random.sample(dataset, 5)
pretty_print_questions(question_sample)
```
```python
for question in question_sample:
    response = generate_response("gpt-4o-mini", user=question["question"])
    print(f"Question: {question['question']}\n\nModel Answer: {response}\n")
```
### Exercise - Design eval questions
```c
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵🔵

You should spend up to 30-45 hour on this exercise.
```
The goal of this exercise is explore a range of different questions, and see how well they measure your specification of the target property. At any step, you can use ChatGPT (via the website UI) to help you. 

**Make your specification measurable:**
1. Read your specification of the target property. What questions and answers would give you high evidence about this property? Come up with 2-4 example MC questions and store this as a list of dictionaries in `mcq_examples.json`. Each MCQ dictionary should contain information on the following fields:

    - (Optional) "system": You may include a system prompt that gives more context on the role or task of the AI.
    - "question": The question itself.
    - "answers": 2 or 4 possible answers. 
    - "answer_matching_behavior": The answer that reflects the target property (i.e. the answer label).
    - "answer_not_matching_behavior": The answer that does not reflect the target property (i.e. the answer label).
    - (Optional) "behavior_category": You may have sub-categories of the target property


2. Do these questions still relate to the threat model? If so, where do they fit in? If not, your threat model or your questions or both could be flawed.
3. Ask your example questions to GPT4o-mini **without the answer options** using `generate_response` and see how it responds:
    - Does the answer show that the model correctly understood the question? If not, you will need to rephrase or come up with a different question.
    - Given that the model understands the question, is there another question that is more reasonable? (E.g. You designed a decision-making question expecting the model to take one of two actions, but the model responded with a list of factors that it would need to know to make a decision. You may then choose to include information on these factors in your question, so that the model can reach a decision a more naturally.)
4. Ask your example questions to GPT4o-mini **with** the answer options using `generate_response` and see how it responds:
    - Is its answer different to its freeform response? This may suggest that your answer options are forcing contrived answers.

5. Iterate and explore different types of questions

```python
my_questions = import_json("mcq_examples.json")

# Ask the model to answer the questions without answer options
for question in my_questions:
    response = generate_response("gpt-4o-mini", user=question["question"])
    print(f"Question: {question['question']}\n\nModel Answer: {response}\n")

# Ask the model to answer the questions with answer options
for question in my_questions:
    question_text = question["question"] + "\nChoices:\n"
    for option, answer in question["answers"].items():
        question_text += f"\n({option}) {answer}"
    response = generate_response("gpt-4o-mini", user=question["question"])
    print(f"Question: {question['question']}\n\nModel Answer: {response}\n")
```

### Exercise - Red-team eval questions
```c
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵🔵

You should spend up to 30-45 hour on this exercise.
```

The goal of this exercise is to iteratively red-team and refine the questions you came up with, until you end up with ~20 questions that you are happy with.

**Red-team** 🔁   

1. What other factors than the target property (i.e. confounders) could cause the model to give the same answers? Note that there will always be other factors/dependencies that will cause the model to give a certain answer in addition to the target property (e.g. A power-seeking question in the context of education may depend on the model's preferences for education). In principle, this will be averaged out across many questions, thus still allow us to get a valid signal on the target property. Only **systematic** coufounders (i.e. the same bias present in many questions of the same type) are problematic because they **bias the result towards a certain direction that will not be averaged out**. 
    - Copy paste your questions into ChatGPT and ask it to critique your questions
2. Ask your question to other people: what is their answer and why?
    - Do they find this difficult to understand? Do they find the answer choices awkward?
    - If you ask them to choose the power-seeking answer, are they able to? If not, this may imply that the answer does not accurately reflect power-seeking.

**Refine** 🔁 - Likely, you will realize that your current specification and examples are imprecise or not quite measuring the right thing. Iterate until you’re more happy with it.

3. Ask ChatGPT to improve your example questions. Ask ChatGPT to generate new evaluation questions for the target property using your questions as few-shot examples, then pick the best generated examples and repeat. Play around and try other things!
7. Generate 20 MCQs you are happy about with ChatGPT's help. **Store the list of MCQ items as a list of dictionaries in a JSON file**.


#### Our Example: Tendency to seek power - MCQ Design

<img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/ch3-red-teaming-mcq.png" width=1200>

""", unsafe_allow_html=True)