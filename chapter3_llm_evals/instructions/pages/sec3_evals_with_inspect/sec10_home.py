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
        r"""
# [3.3] - Evals with Inspect

### Colab: [**exercises**](https://colab.research.google.com/drive/1lDwKASSYGE4y_7DuGSqlo3DN41NHrXEw?usp=sharing) | [**solutions**](https://colab.research.google.com/drive/1bZkkJd8pAVnSN23svyszZ3f4WrnYKN_3?usp=sharing)

Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-uk/shared_invite/zt-28h0xs49u-ZN9ZDbGXl~oCorjbBsSQag), and ask any questions on the dedicated channels for this chapter of material.

You can toggle dark mode from the buttons on the top-right of this page.

Links to other chapters: [**(0) Fundamentals**](https://arena3-chapter0-fundamentals.streamlit.app/), [**(1) Interpretability**](https://arena3-chapter1-transformer-interp.streamlit.app/), [**(2) RL**](https://arena3-chapter2-rl.streamlit.app/).

<img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/ch3-evals-cover.jpeg" width="350">

## Introduction

Today we'll introduce you to the process of running an evaluation using the `Inspect` library. This is a library that has been built by UK AISI to provide standardisation across different evals.

We'll begin by introducing the high-level overview of how Inspect works, including the key components of: a dataset, solvers, and scorers. Then we'll write our own solvers so that we can design our own evaluation procedure (including whether we want to use chain-of-thought, self-critique, multi-turn agent dialogues, etc). These evaluations will use the questions that you generated in [3.2].

Similarly to [3.1] and [3.2], most of our solutions will be built around the example **tendency to seek power** dataset. So you should approach the exercises with this in mind. You can either:

1. Use the same dataset as the solutions, and improve upon the evaluations that are run in the solutions. You could come up with your own evaluation procedure on this dataset.

2. Use your own dataset for your chosen model property. For this, you will have needed to work through Chapters [3.1] and [3.2] and have a dataset of ~300 questions on which to evaluate the model.

Each exercise will have a difficulty and importance rating out of 5, as well as an estimated maximum time you should spend on these exercises and sometimes a short annotation. You should interpret the ratings & time estimates relatively (e.g. if you find yourself spending about 50% longer on the exercises than the time estimates, adjust accordingly). Please do skip exercises / look at solutions if you don't feel like they're important enough to be worth doing, and you'd rather get to the good stuff!

## Content & Learning Objectives


#### 1️⃣ Intro to Inspect

> ##### Learning objectives
>
> - Understand the big picture of how Inspect works.
> - Understand the components of a `Task` object.
> - Turn our json dataset into an Inspect dataset.
> - Understand the role of solvers and scorers in Inspect.


#### 2️⃣ Writing Solvers

> ##### Learning objectives
> 
> - Understand what solvers are and how to build one.
> - Understand how prompting affects model responses to your questions.
> - Think of how you want your evaluation to proceed and write solvers for this.

#### 3️⃣ Writing Tasks and Evaluating

> ##### Learning objectives
>
> - Understand why we have to shuffle choices in MCQs.
> - Understand how to write a task in Inspect to carry out our evaluation
> - Get baselines on our model to determine whether it can understand the questions we're asking.
> - Run a task on our model, and check the results.
>


#### 4️⃣ Bonus: Log Files and Plotting

>
> - Understand how Inspect stores log files.
> - Extract the key data from log files.
> - Plot the results of the evaluation.
>

## Setup

```python
import os
import json
from typing import Literal
from inspect_ai import Task, eval, task
from inspect_ai.log import read_eval_log
from utils import omit
from inspect_ai.model import ChatMessage
from inspect_ai.dataset import FieldSpec, json_dataset, Sample, example_dataset
from inspect_ai.solver._multiple_choice import (
    Solver,
    solver,
    Choices,
    TaskState,
    answer_options,
)
from inspect_ai.solver._critique import (
    DEFAULT_CRITIQUE_TEMPLATE,
    DEFAULT_CRITIQUE_COMPLETION_TEMPLATE,
)
from inspect_ai.scorer import model_graded_fact, match, answer, scorer
from inspect_ai.scorer._metrics import (accuracy, std)
from inspect_ai.scorer._answer import AnswerPattern
from inspect_ai.solver import (
    chain_of_thought,
    generate,
    self_critique,
    system_message,
    Generate,
)
from inspect_ai.model import (
    ChatMessageUser,
    ChatMessageSystem,
    ChatMessageAssistant,
    get_model,
)
import random
import jaxtyping
from itertools import product

# Make sure exercises are in the path; 
chapter = r"chapter3_llm_evals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part3_run_evals_with_inspect").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)

from utils import import_json, save_json, retry_with_exponential_backoff, pretty_print_questions, load_jsonl, omit
import exercises.part3_run_evals_with_inspect.tests as tests
```
                
""",
        unsafe_allow_html=True,
    )
