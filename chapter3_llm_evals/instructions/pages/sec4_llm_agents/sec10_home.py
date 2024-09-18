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
# [3.4] - LLM Agents

<!-- ### Colab: [**exercises**](https://colab.research.google.com/drive/1lDwKASSYGE4y_7DuGSqlo3DN41NHrXEw?usp=sharing) | [**solutions**](https://colab.research.google.com/drive/1bZkkJd8pAVnSN23svyszZ3f4WrnYKN_3?usp=sharing)-->

Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-uk/shared_invite/zt-28h0xs49u-ZN9ZDbGXl~oCorjbBsSQag), and ask any questions on the dedicated channels for this chapter of material.

You can toggle dark mode from the buttons on the top-right of this page.

Links to other chapters: [**(0) Fundamentals**](https://arena3-chapter0-fundamentals.streamlit.app/), [**(1) Interpretability**](https://arena3-chapter1-transformer-interp.streamlit.app/), [**(2) RL**](https://arena3-chapter2-rl.streamlit.app/).

<img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/ch3-evals-cover.jpeg" width="350">

## Introduction

<!-- WRITE INTRO! -->

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
import json
import os

import wikipedia
from wikipedia import WikipediaPage
from wikipedia import DisambiguationError, PageError
from openai import OpenAI
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from anthropic import Anthropic
from typing import Literal, Optional, Dict, List, Any
from abc import abstractmethod
import math
import re

#Make sure exercises are in the path
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part4_llm_agent_evals").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)

from utils import import_json, save_json, retry_with_exponential_backoff, pretty_print_questions, load_jsonl, omit
from utils import countrylist
from utils import evaluate_expression, apply_user_format, apply_assistant_format, establish_client_anthropic, establish_client_OpenAI, retry_with_exponential_backoff

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

client = OpenAI()
```
                
""",
        unsafe_allow_html=True,
    )
