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

# 3️⃣ Writing Tasks and Evaluating

## Scorers

Scorers are what the inspect library uses to judge model output. We use them to determine whether the model's answer is the `target` answer. From the [inspect documentation](https://inspect.ai-safety-institute.org.uk/scorers.html):


><br>
>Scorers generally take one of the following forms:
>
>1. Extracting a specific answer out of a model’s completion output using a variety of heuristics.
>
>2. Applying a text similarity algorithm to see if the model’s completion is close to what is set out in the `target`.
>
>3. Using another model to assess whether the model’s completion satisfies a description of the ideal answer in `target`.
>
>4. Using another rubric entirely (e.g. did the model produce a valid version of a file format, etc.)
>
><br>

Since we are writing a multiple choice benchmark, for our purposes we can either use the `match()` scorer, or the `answer()` scorer. The docs for these scorer functions can be found [here](https://inspect.ai-safety-institute.org.uk/scorers.html#built-in-scorers) (and the code can be found [here](https://github.com/UKGovernmentBEIS/inspect_ai/blob/c59ce86b9785495338990d84897ccb69d46610ac/src/inspect_ai/scorer/_match.py) in the inspect_ai github). You should make sure you have a basic understanding of how the `match()` and `answer()` scorers work before you try to use them, so spend some time reading through the docs and code here.

<details><summary>Aside: some other scorer functions:</summary>
<br>
Some other scorer functions that would be useful in different evals are:

- `includes()` which determines whether the target appears anywhere in the model output.

- `pattern()` which extracts the answer from the model output using a regular expression (the `answer()` scorer is essentially a special case of `pattern()`, where Inspect has already provided the regex for you).

- `model_graded_qa()` which has another model assess whether the output is a correct answer based on guidance in `target`. This is especially useful for evals where the answer is more open-ended. Depending on what sort of questions you're asking here, you may want to change the prompt template that is fed to the grader model.

- `model_graded_fact()` which does the same as `model_graded_qa()`, but is specifically for whether the model has included relevant factual content when compared to an expert answer (this would be used mostly on a capability evaluation).
<br>
</details>

## Loading Datasets

Before we can finalize our evaluation task, we'll need to modify our `record_to_sample` function from earlier. There are two issues with the function we wrote:

- First of all, we didn't shuffle the order in which the choices are presented to the model. Your dataset may have been generated pre-shuffled, but if not, then you ought to be aware that the model has a bias towards the first answer. See for example [Narayanan & Kapoor: Does ChatGPT have a Liberal Bias?](https://www.aisnakeoil.com/p/does-chatgpt-have-a-liberal-bias) To mitigate this confounder, we should shuffle the choices (and answer) around when loading in the dataset.

<details><summary>Aside: Inspect's built-in shuffle function</summary> 

Inspect comes with a shuffle function for multiple choice questions. However, this function then automatically "unshuffles" *only some* of the model's responses back to how they are presented in your dataset.

I'm sure there are use cases for this, but in our case it's unnecessarily confusing to see the model reasoning about e.g. how much it thinks "option B" is the correct answer, and then Inspect telling you the model selected "option A" since its selection choice gets "unshuffled" but its reasoning doesn't. For this reason, and because it's not very hard to do, we will just build our own shuffler that doesn't do any unshuffling to any part of the output.</details>

- Secondly, our original `record_to_sample` function can only load questions with the corresponding system prompt. It is often useful to load the dataset without a system prompt. For example, this can be useful when we obtain baselines, since we may not want the model to answer "as though" it is in the context given by the system prompt, but we do want it to be aware of that context. We also may want to load without a system prompt when running an evaluation, so we can see how the model reacts to our questions when presented with a more generic "You are a harmless, helpful, honest AI Chat Assistant" system prompt (which we can then provide with our `system_message` solver). 

### Exercise - Write record_to_sample function with shuffling
```c
Difficulty: 🔴⚪⚪⚪⚪
Importance: 🔵🔵⚪⚪⚪

You should spend up to 5-10 mins on this exercise.
```

Take the record_to_sample function you wrote in section 1, and modify it so that it randomly shuffles the order of `choices`. Remember to correspondingly shuffle the `target` answer.

```python
def record_to_sample_shuffle(record: dict) -> Sample:
    return Sample(
        input=[],
        target= "A",
        choices=[],
        metadata={},
    )
```
    
### Exercise - Write record_to_sample function to handle system prompts
```c
Difficulty: 🔴⚪⚪⚪⚪
Importance: 🔵🔵⚪⚪⚪

You should spend up to 10-15 mins on this exercise.
```

As mentioned above, there are two reasons we might not want the system prompt provided to the model in the usual way. Either:
- We might want the model not to have the specific context of the system prompt, and ask it to respond with a different system message (which we can do via the system_prompt solver we wrote in section 3).

- When we get baselines, we might want the model to provide an answer with a different system prompt, but we still want to provide the context from the original system prompt somewhere in the user prompt.

Write two version of the `record_to_sample` function which handles system prompts in each of these two ways. 

Remember to shuffle answers in this function as well.

```python
def record_to_sample_no_system_prompt(record: dict) -> Sample:
    return Sample(
        input=[],
        target="A",
        choices=[],
        metadata={},
    )
```

```python
def record_to_sample_system_prompt_as_context(record: dict) -> Sample:
    return Sample(
        input=[],
        target="A",
        choices=[],
        metadata={},
    )
```

Now let's wrap our `record_to_sample` functions into just one `record_to_sample` function, so that we can pass arguments to it and see the outcomes.

```python
def record_to_sample(system: bool, system_prompt_as_context: bool):
    assert (
        not system or not system_prompt_as_context
    ), "ERROR: You can only set one of system or system_prompt_as_context to be True."

    def wrapper(record: dict) -> Sample:
        if system:
            return record_to_sample_shuffle(record)
        elif system_prompt_as_context:
            return record_to_sample_system_prompt_as_context(record)
        else:
            return record_to_sample_no_system_prompt(record)

    return wrapper
```
    
%% [markdown]
Now we can load our dataset in using these functions.

Just fill in `"your/path/to/eval_dataset.json"` below with the path to your dataset.

```python
path_to_eval_dataset = r"your/path/to/dataset/here.json"

eval_dataset_system = json_dataset(
    path_to_eval_dataset,
    record_to_sample(system=True, system_prompt_as_context=False),
)
eval_dataset_no_system = json_dataset(
    path_to_eval_dataset,
    record_to_sample(system=False, system_prompt_as_context=False),
)
eval_dataset_system_as_context = json_dataset(
    path_to_eval_dataset,
    record_to_sample(system=False, system_prompt_as_context=True),
)
test_dataset = json_dataset(
    path_to_eval_dataset,
    record_to_sample(system=True, system_prompt_as_context=False),
)[0:10]
test_dataset_system_as_context = json_dataset(
    path_to_eval_dataset,
    record_to_sample(system=False, system_prompt_as_context=True),
)[0:10]
```

%% [markdown]
### Exercise - Implementing the benchmark_test task
```c
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵🔵⚪

You should spend up to 15-20 mins on this exercise. 
```

First build out a baseline test: This should give you a sense of whether the model can recognize what the target behavior option is. When running this test, you should use the `eval_dataset_system_as_context` dataset, since we won't necessarily want the model to answer *as though* it is placed in the context of the system prompt, but we still want it to be *aware* of that context.

A few notes:

- You need to ensure your prompts for benchmarking are finalized.

- You need to figure out how you're going to deal with Claude often refusing to answer unethically. (Possibly just accept you can't get a baseline here; if you push too hard Anthropic might ban you).

- You need to decide what path you want your evaluations to proceed along (i.e. what solvers you'll want to use). 

Build out the evaluation. You'll likely need to craft and test specialized prompts. We loaded in a `test_dataset_system_as_context` dataset above and you should use this to test how different models understand and respond to your prompts before you carry out the full evaluation.

<details><summary>Aside: Claude's refusals</summary>

Claude has a tendency to refuse to answer in certain "bad" ways. If you're trying to get a baseline for a model from the Claude series, you may not be able to get it to answer to a question like "Answer as though you're maximally evil." If you have a 2-choice dataset with "Yes/No" answers, you can get a decent enough estimate by asking it to "Answer as though you're maximally good" and then assuming it would answer in the opposite if it were asked to be "maximally evil." If you have more than 2 choices then you may have to avoid taking a "negative" baseline. 

</details>

```python
benchmark_template_CoT = r""""""

benchmark_template_MCQ = r""""""

benchmark_template_make_choice = r""""""
```

```python
@task
def benchmark_eval(
    multiple_choice_template: str,
    chain_of_thought_template: str,
    make_choice_prompt: str,
    test_dataset: bool = True,
):
    """
    A task which carries out your benchmark evaluation.

    Args:
        multiple_choice_template : The template to use for the multiple choice evaluation

        chain_of_thought_template : The template to use for the chain of thought evaluation

        make_choice_prompt : The prompt to use for the make choice evaluation

        test_dataset : Whether to use the test dataset or not
    
    Returns:
        Task : A task which carries out your benchmark evaluation
    """
    return Task()
```

### Exercise - Using Inspect's `eval()` function
```c
Difficulty: 🔴⚪⚪⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 5-10 mins on this exercise. 
```

Now use Inspect' `eval()` function and test that your task works. Try running it on `"gpt-4o-mini"` (since this model is cheap, and should be sufficient to flag any potential issues with your evaluation). This model should be capable of understanding what your questions are getting at, and if it doesn't manage to respond with the target behavior, you should *probably* update your prompts. Make sure to run this with `test_dataset=True` the first few times. To use `Inspect`'s eval function, you'll need to pass a `task`, a `model` and a `log_dir` (there are many other arguments `eval()` can take, but these are all that we'll need). 

```python
logs = eval(
    benchmark_eval(), # Task to evaluate (including the templates)
    model = "", # Model to evaluate
    log_dir = "./path/to/log/directory" # Add your log directory path here
    
)
```
### Exercise - Finalize and run your evaluation task
```c
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 10-15 mins on this exercise. 
```

Create one task which has the capacity to run every eval you may want to run. Do this using conditional logic to append solvers to `eval_plan` based on arguments in the task function. Then we'll be able to run our evals with one call to the `eval()` function by iterating over these different arguments.

Make sure to include assert statements in case two solvers are appended which shouldn't go together (for example, it doesn't make much sense to get the model to do `self_critique` if you haven't already gotten it to output some reasoning to critique).

```python
@task
def power_seeking(system : bool, CoT : bool, self_critique_enabled : bool, self_critique_model : Literal["openai/gpt-4o-mini"] = "openai/gpt-4o-mini"):
    return Task()
```

### Exercise - Run your evaluation
```c
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵⚪⚪

You should spend up to 10-15 mins to code this exercise. (Running will take longer).
```
Now conduct your final evaluation. You can either use a series of nested `for` loops and conditionals to run the different versions of your task built above, or you could put them all the different parameters into one `itertools.product` object.

You should write this code so that it carries out your entire suite of evaluations on the model we want to evaluate. We could also run it on every model we want to evaluate all at once, however this would take a substantial amount of time, and the cost would be hard to predict. Before running it on larger models, test that everything works on `"openai/gpt-4o-mini"`, as this is by far the cheapest model currently available (and surprisingly capable)!

For ease of use later, you should put your actual evaluation logs in a different subfolder in `/logs/` than your baseline evaluation logs.

<details><summary> API Prices:</summary>

The pricings of models can be found [here for OpenAI](https://www.openai.com/api/pricing), and [here for Anthropic](https://www.anthropic.com/pricing#anthropic-api)

</details>


```python
eval_model= "openai/gpt-4o-mini"

for _system in [True,False]:
    for _CoT in [True,False]:
        for _self_critique_enabled in [True,False]:
            if _self_critique_enabled and not _CoT:
                pass
            else:
                logs = eval(
                    power_seeking(system = _system, CoT = _CoT, self_critique_enabled = _self_critique_enabled, self_critique_model = eval_model),
                    model = eval_model,
                    log_dir = r"path/to/log/directory"
                )
```
        ''',
        unsafe_allow_html=True,
    )
