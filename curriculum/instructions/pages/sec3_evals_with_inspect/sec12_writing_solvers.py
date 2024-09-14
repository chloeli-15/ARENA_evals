import streamlit as st



def section():
    st.sidebar.markdown(
        r'''

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#introduction'>Introduction</a></li>
    <li class='margtop'><a class='contents-el' href='#content-learning-objectives'>Content & Learning Objectives</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#1-advanced-api-calls'>Advanced API Calls</a></li>
        <li><a class='contents-el' href='#2-dataset-generation'>Dataset Generation</a></li>
        <li><a class='contents-el' href='#3-dataset-quality-control'>Dataset Quality Control</a></li>
        <li><a class='contents-el' href='#4-putting-It-together'>Putting it Together: Generation-Evaluation</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
</ul>''',
        unsafe_allow_html=True,
    )

    st.markdown(
        r'''
# 2️⃣ Writing Solvers 

Solvers are functions that can manipulate the input to the model, and get the model to generate output (see [docs here](https://inspect.ai-safety-institute.org.uk/solvers.html)). Rather than using `Inspect`'s built-in solvers, we will be re-implementing these solvers or writing our own custom solvers, since:

1. It gives us more flexibility to use solvers in ways that we want to.

2. It will let us define our own custom solvers if there's another idea we're interested in testing in our evals.  

The most fundamental solver in the `Inspect` library is the `generate()` solver. This solver makes a call to the LLM API and gets the model to generate a response. Then it automatically appends this response to the chat conversation history.

Here is the list of solvers that we will write:

* `prompt_template()`: This solver can add any information to the text of the question. For example, we could add an indication that the model should answer using chain-of-thought reasoning.

* `multiple_choice_template()`: This is similar to Inspect's [`multiple_choice`](https://inspect.ai-safety-institute.org.uk/solvers.html#built-in-solvers) function but our version won't call `generate()` automatically, it will just provide a template to format multiple choice questions.

* `system_message()`: This adds a system message to the chat history.

* `self_critique()`: This gets the model to critique its own answer.

* `make_choice()`: This tells the model to make a final decision, and calls `generate()`.

### TaskState

A solver function takes in `TaskState` as input and applies transformations to it (e.g. modifies the messages, tells the model to output). `TaskState` is the main data structure that `Inspect` uses to store information while interacting with the model. A `TaskState` object is initialized for each question, and stores relevant information about the chat history and model output as a set of attributes. There are 3 attributes of `TaskState` that solvers interact with (there are additional attributes that solvers do not interact with): 

1. `messages`
2. `user_prompt`
3. `output`

Read the Inspect description of `TaskState` objects [here](https://inspect.ai-safety-institute.org.uk/solvers.html#task-states-1).

[Diagram of TaskState objects and solvers and how they interact](https://google.com)

## Asyncio

When we write solvers, what we actually write are functions which return `solve` functions. The structure of our code generally looks like:

```python
def solver_name(args) -> Solver:
    async def solve(state : TaskState, generate : Generate) -> TaskState:
        #Modify Taskstate in some way
        return state
    return solve
```

You might be wondering why we're using the `async` keyword. This comes from the `asyncio` python library. We use this because some solvers can be very time-consuming, and we want to be able to run many evaluations as quickly as possible in parallel. When we define functions without using `async`, functions get in a long queue and only start executing when the functions before it in the queue are finished. The `asyncio` library lets functions essentially "take a number" so other functions can work while the program waits for this function to finish executing. The `asyncio` library refers to this scheduling mechanism as an "event loop". When we define a function using `async def` this lets `asyncio` know that other functions can be run in parallel with this function in the event loop; `asyncio` won't let other functions run at the same time if we define a function normally (i.e. just using the `def` keyword). 

When we make calls to LLM APIs (or do anything time-consuming), we'll use the `await` key word on that line. This keyword can only be used inside a function which is defined using `async def`. This lets the program know that this line may take a long time to finish executing, so the program lets other functions work at the same time as this function executes. It does this by yielding control back to `asyncio`'s event loop while waiting for the "awaited" operation to finish.

In theory, we could manage all of this by careful scheduling ourselves, but `await` and `async` keep our code easy-to-read by preserving the useful fiction that all our code is running in-order (even though in `asyncio` is letting the functions execute asynchronously to maximize efficiency).
<!---
<details><summary>How is this different from Threadpool Executor</summary>

You might be wondering, if both `asyncio` and `ThreadPoolExecutor` allow different tasks to run concurrently, what's the difference between them? The key differences for our use cases are:

1. They use a different model of concurrency
- `asyncio` uses a concurrency model called cooperative multitasking. It only runs a single "worker" which manages multiple tasks. The worker switches between tasks when it's waiting for something (like an API response). Each task yields back to the worker when it's not actively doing something.

- `ThreadPoolExecutor` operates using preemptive multitasking. This allows the operating system itself to interrupt a running task and temporarily suspend it to run another task. This interruption happens without any explicit cooperation between the running programs (i.e. there's no need for one function to yield control back to a worker).

2. Use cases
- `asyncio` is most useful for Input/output-bound tasks (like API calls). All the work can happen in one line (thread), but tasks take turns. This is most efficient for tasks where a lot of time is spent waiting.

- `ThreadPoolExecutor` is most useful for CPU-bound tasks (like dataset processing). It can utilize multiple CPU cores for genuine parallel processing.

3. Scalability
- `asyncio` can handle many concurrent operations efficiently — potentially up to thousands of different tasks — because they all run in a single thread and switch quickly. Since `Inspect` wants to efficiently conduct evaluations (which may have many thousands of questions), it makes sense to use `asyncio`  

- `ThreadPoolExecutor` is limited by the number of threads — each thread consumes some amount of the system resources, so you're typically limited to dozens (or potentially hundreds) of threads depending on your system.

There are other differences, but these are the main ones that are worth mentioning here. We used `ThreadPoolExecutor` in Day 2 because we'll only ever want to make a relatively small number of question generation calls at one time. For this it's easier to use `ThreadPoolExecutor` than to make all of our code compatible with `asyncio`. However, `Inspect` is already implemented to be compatible with `asyncio`, so we'll use it here.

</details>
--->

If you want more information about how parallelism works in Inspect, read [this section of the docs](https://inspect.ai-safety-institute.org.uk/parallelism.html#sec-parallel-solvers-and-scorers)!


### Asyncio Example

The below code runs without using `asyncio`. You should see that the code takes ~6 seconds to run 3 functions which each sleep for 2 seconds. This is because the functions run in series, and each function waits for the previous one to finish before starting. 

```python
import time


def ask_ai(question: str) -> str:
    print(f"Asking AI: {question}")
    time.sleep(2)  # Simulating AI processing time
    return f"AI's answer to '{question}'"


def process_question(question: str) -> str:
    print(f"Starting to process: {question}")
    answer = ask_ai(question)
    print(f"Got answer for: {question}")
    return answer


def main() -> None:
    questions = ["What is Python?", "How does async work?", "What's for dinner?"]
    answers = []
    for question in questions:
        answers.append(process_question(question))
    print("All done!", answers)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total time: {time.time() - start_time:.2f} seconds")
```

The below code runs using `asyncio`. You should see that the code takes ~2 seconds to run 3 functions which each sleep for 2 seconds. This is because the functions run in parallel, and each function starts running at roughly the same time.

<details><summary>Why are we calling different sleep functions? Isn't this cheating?</summary>

We have to call a different sleep method to see `asyncio` work, since `time.sleep` forces the actual program to sleep for 2 seconds, so `asyncio` would run into this and just have to sleep for 2 seconds three times. A call to `asyncio.sleep` is a better model of API calling, as this allows other programs to continue running 'in the background' while we're waiting for it to finish sleeping, much like waiting for an API call to finish.

</details>

```python
import asyncio
import time


async def ask_ai(question: str) -> str:
    print(f"Asking AI: {question}")
    await asyncio.sleep(2)  # Simulating AI processing time
    return f"AI's answer to '{question}'"


async def process_question(question: str) -> str:
    print(f"Starting to process: {question}")
    answer = await ask_ai(question)
    print(f"Got answer for: {question}")
    return answer


async def main() -> None:
    start_time = time.time()
    questions = ["What is Python?", "How does async work?", "What's for dinner?"]
    tasks = [process_question(q) for q in questions]
    answers = await asyncio.gather(*tasks)
    print("All done!", answers)
    print(f"Total time: {time.time() - start_time:.2f} seconds")


start_time = time.time()
try:
    asyncio.get_event_loop().run_until_complete(main())
except RuntimeError:
    asyncio.create_task(main())
```
## Writing Solvers

### Exercise - Write `prompt_template` function
```c
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 10-15 minutes on this exercise.
```

Write a solver function that checks if `state` has a `user_prompt`, and modifies the `user_prompt` according to a template string which contains `"{question}"` — where the original `user_prompt` will get placed in the template.

You might find Inspect's [solver documentation](https://inspect.ai-safety-institute.org.uk/solvers.html), or even their [github source code](https://github.com/UKGovernmentBEIS/inspect_ai/tree/0d9f9c8d5b472ec4f247763fe301a5be5839a607/src/inspect_ai/solver) useful for this exercise (and the following exercise).

```python
def prompt_template(template: str) -> Solver:
    """
    Returns a solve function which modifies the user prompt with the given template.

    Args:
        template : The template string to use to modify the user prompt. Must include {question} to be replaced with the original user

    Returns:
        solve : A solve function which modifies the user prompt with the given template
    """
    assert r"""{prompt}""" in template, r"""ERROR: Template must include {prompt}"""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.user_prompt:
            state.user_prompt.text = template.format(prompt=state.user_prompt.text)
        return state

    return solve
```

### Exercise - Write a multiple_choice_format solver
```c
Difficulty: 🔴⚪⚪⚪⚪
Importance: 🔵🔵⚪⚪⚪

You should spend up to 5-10 mins on this exercise.
```

This solver should check that `state` has both `user_prompt` and `choices` properties, and modify the `user_prompt` according to a template string which contains `{question}` and `{choices}`. Before putting `state.choices` into the template, you should use Inspect's `answer_options()` function to provide a layout for `state.choices`. You can read Inspect's code for the `answer_options()` function [here](https://github.com/UKGovernmentBEIS/inspect_ai/blob/ba03d981b3f5569f445ce988cc258578fc3f0b80/src/inspect_ai/solver/_multiple_choice.py#L37). 

```python
@solver
def multiple_choice_format(template: str, **params: dict) -> Solver:
    """
    Returns a solve function which modifies the initial prompt to be in the format of a multiple choice question. Make sure that {question} and {choices} are in the template string, so that you can format those parts of the prompt.

    Args:
        template : The template string to use to modify the user prompt. Must include {question} and {choices} to be replaced with the original user prompt and the answer choices, respectively.

    Returns:
        solve : A solve function which modifies the user prompt with the given template
    """
    assert (
        r"{choices}" in template and r"{question}" in template
    ), r"ERROR: The template must contain {question} or {choices}."

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.user_prompt and state.choices:
            state.user_prompt.text = template.format(
                question=state.user_prompt.text,
                choices=answer_options(state.choices),
                **params,
            )
        return state

    return solve
```

### Exercise - Write a make_choice solver
```c
Difficulty: 🔴⚪⚪⚪⚪
Importance: 🔵🔵⚪⚪⚪

You should spend up to 5-10 mins on this exercise.
```

This solver should add a user prompt to `state.messages` telling the model to make a final decision out of the options provided. 

```python
@solver
def choose_multiple_choice(prompt: str) -> Solver:
    """
    Returns a solve function which adds a user message at the end of the state.messages list with the given prompt.

    Args:
        prompt : The prompt to add to the user messages

    Returns:
        solve : A solve function which adds a user message with the given prompt to the end of the state.messages list
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.messages:
            state.messages.append(ChatMessageUser(content=prompt))
        return state

    return solve
```

### Exercise - Write a system_message solver
```c
Difficulty:🔴🔴⚪⚪⚪
Importance:🔵🔵⚪⚪⚪

You should spend up to 10-15 mins on this exercise.
```

This solver should add a system message to `state.messages`. The system message should be inserted after the last system message in `state.messages`. You may want to write an `insert_system_message` function which finds the last system message in a list of `messages` and adds the given `system_message` after this.

```python
@solver
def system_message(system_message: str, **params: dict) -> Solver:
    """
    Returns a solve function which inserts a system message with the given content into the state.messages list. The system message is inserted after the last system message in the list.

    Args:
        system_message : The content of the system message to insert into the state.messages list

    Returns:
        solve : A solve function which inserts a system message with the given content into the state.messages list
    """

    def insert_system_message(messages: list, message: ChatMessageSystem) -> None:
        """
        Inserts the given system message into the messages list after the last system message in the list.

        Args:
            messages : The list of messages to insert the system message into
            message : The system message to insert into the list

        Returns:
            None
        """
        lastSystemMessageIndex = -1
        for i in list(reversed(range(0, len(messages)))):
            if isinstance(messages[i], ChatMessageSystem):
                lastSystemMessageIndex = i
                break
        messages.insert(lastSystemMessageIndex + 1, message)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        insert_system_message(state.messages, ChatMessageSystem(content=system_message))
        return state

    return solve
```

Now you should be able to write any other solvers that might be useful for your specific evaluation task (if you can't already use any of the existing solvers to accomplish the evaluation you want to conduct). For example:

A "language" template, which tells the model to respond in diffeent languages (you could even get the model to provide the necessary translation first).

A solver which provides "fake" reasoning by the model where it has already reacted in certain ways, to see if this conditions the model response in different ways. You would have to use the ChatMessageUser and ChatMessageAssistant Inspect classes for this

A solver which tells the model to respond badly to the question, to see if the model will criticise and disagree with itself after the fact when prompted to respond as it wants to.

Any other ideas for solvers you might have.
 Do so here!


### Exercise - Implement multiple_choice_format

This should take in a state, and return the state with the user prompt reformatted for multiple choice questions. You should take inspiration from the `prompt_template` function you wrote above, but for multiple choice questions you will need `"{choices}"` to be in the template string as well as `"{question}"`, and format the string accordingly.

<details><summary> Inspect's <code>multiple_choice</code> solver</summary>

Inspect has its own multiple choice format solver, but it's designed without chain-of-thought or self-critique in mind: it automatically calls generate to get the model to instantly make a final decision. Here we'll just implement the formatting part of this function, which formats the multiple choice question. Later we'll implement a separate solver to do the generate part, telling the model to make a final decision, and how to format that final decision.

</details>

```python
@solver
def multiple_choice_format(template: str, **params: dict) -> Solver:
    """
    This modifies the initial prompt to be in the format of a multiple choice question. Make sure that {question} and {choices} are in the template string, so that you can format those parts of the prompt.
    """
    assert (
        r"{choices}" in template and r"{question}" in template
    ), r"ERROR: The template must contain {question} or {choices}."

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.user_prompt:
            state.user_prompt.text = template.format(
                question=state.user_prompt.text,
                choices=answer_options(state.choices),
                **params,
            )
        return state

    return solve

### Exercise - Implement a make_choice solver

This solver should get the model to make a final decision about its answer. This is the solver that we'll call after any chain_of_thought or self_critique solvers finish running. You should get the model to output an answer in a format that the scorer function will be able to recognise. By default this scorer should be either the `answer()` scorer or the `match()` scorer (docs for these scorers [here](https://inspect.ai-safety-institute.org.uk/scorers.html#built-in-scorers).

```python
@solver
def make_choice(template: str, **params: dict) -> Solver:
    """
    This modifies the ChatMessageHistory to add a message at the end indicating that the model should make a choice out of the options provided.

    Make sure the model outputs so that its response can be extracted by the "match" solver. You will have to use Inspect's ChatMessageUser class for this.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.messages:
            state.messages.append(ChatMessageUser(content=template))
        return state

    return solve
```

### Exercise - Implement self-critique 
```c
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵⚪⚪⚪

You should spend up to 20-25 mins on this exercise.
```
Chloe comments: 
- include explanation of the technique in general (maybe link to ref paper)

We will be re-implementing the built-in `self_critique` solver from `inspect`. This will enable us to modify the function more easily if we want to adapt how the model critiques its own thought processes.

<details><summary>How Inspect implements <code>self_critique</code></summary>

By default, Inspect will implement the `self_critique` function as follows:

1. Take the entirety of the model's output so far

2. Get the model to generate a criticism of the output, or repeat say "The original answer is fully correct" if it has no criticism to give. 

3. Append this criticism to the original output before any criticism was generated, but as a *user message* (**not** as an *assistant message*), so that the model believes that it is a user who is criticising its outputs, and not itself (I think this generally tend to make it more responsive to the criticism, but you may want to experiment with this and find out).

4. Get the model to respond to the criticism or repeat the answer fully if there is no criticism.

5. Continue the rest of the evaluation.

 </details>


```python
@solver
def self_critique(
    model: str,
    critique_template: str | None,
    critique_completion_template: str | None,
    **params: dict,
) -> Solver:
    """
    Args:
        - model: The model we use to generate the self-critique

        - critique_template: This is the template for how you present the output of the model so far to itself, so that the model can *generate* the critique. This template should contain {question} and {completion} to be formatted.
        
        - critique_completion_template: This is the template for how you present the output and critique to the model, so that it can generate an updated response based on the critique. This template should include {question} and {completion} and {critique} to be formatted.

    This is built into Inspect, so if you're happy to make use their version without any modifications, you can just use Inspect's self_critique solver.
    """

    Model = get_model(model)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        metadata = omit(state.metadata, ["question", "completion", "critique"])
        critique = await Model.generate(
            critique_template.format(
                question=state.input_text,
                completion=state.output.completion,
                **metadata,
            )
        )

        state.messages.append(
            ChatMessageUser(
                content=critique_completion_template.format(
                    question=state.input_text,
                    completion=state.output.completion,
                    critique=critique.completion,
                    **metadata,
                )
            )
        )

        return await generate(state)

    return solve
```

### Exercise (optional) - Write your own solvers
```c
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 15-25 mins on this exercise.
```

Now write any solvers that you want to write. Some ideas for additional solvers you may want to include below:

- A "language" solver, which tells the model to respond in different languages (you could even get the model to provide the necessary translation for the question first).

- A solver which uses assistant messages to condition the model to react in different ways. You would have to use the ChatMessageUser and ChatMessageAssistant Inspect classes for this.

- A solver which tells the model to respond badly to the question. Then you could use the `self_critique` solver to see if the model will criticise and disagree with itself after the fact, when it can't see that it was told to respond poorly to the question in the first place.

- Some solvers which enable more advanced multi-turn evaluations. These would function similarly to `self-critique` solvers, where models interact with each other. For example, one model could encourage the model being evaluated to respond in a more power-seeking manner.

```python
#Now you should be able to write any other solvers that might be useful for your specific evaluation task (if you can't already use any of the existing solvers to accomplish the evaluation you want to conduct). For example:
#
#A "language" template, which tells the model to respond in different languages.
#
#A solver which uses assistant messages to condition the model.
#
#A solver which tells the model to respond badly to the question.
#
#Some solvers which enable more advanced multi-turn evaluations. 
#
#Any other ideas for solvers you might have. Write them here!

```

### Exercise (optional) - Write prompts for your solvers

```c
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 10-20 mins on this exercise. 
```

Now write prompts for any of the solvers you're planning on including in your evaluation task. You might want to come back to this exercise later if you see the model reacting incorrectly to any of your prompting.

```python
self_critique_critique_template = DEFAULT_CRITIQUE_TEMPLATE
self_critique_completion_template = DEFAULT_CRITIQUE_COMPLETION_TEMPLATE
chain_of_thought_template = r"""{prompt}\n\nBefore answering, think through how you want to answer the question. Make sure to think step-by-step and weigh all relevant concerns, and then arrive at a conclusive answer."""
multiple_choice_format_template = r"""{question}\n\n{choices}"""
make_choice_template = r"""Now provide your conclusion in the form ANSWER: $LETTER where $LETTER is one of A, B. Only output in this format."""
system_message_template = r"""You are a harmless, helpful, and honest AI chat assistant."""
```

''',
        unsafe_allow_html=True,
    )
