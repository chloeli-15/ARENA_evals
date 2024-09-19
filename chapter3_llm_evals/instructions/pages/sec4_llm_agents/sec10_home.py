import streamlit as st


def section():
    st.sidebar.markdown(
        r"""
## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#introduction'>Introduction</a></li>
    <li class='margtop'><a class='contents-el' href='#content-learning-objectives'>Content & Learning Objectives</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#1-intro-to-llm-agents'>Intro to LLM Agents</a></li>
        <li><a class='contents-el' href='#2-simple-arithmetic-agent'>Simple Arithmetic Agent</a></li>
        <li><a class='contents-el' href='#3-wikiagent'>WikiAgent</a></li>
        <li><a class='contents-el' href='#4-elicitation'>Elicitation</a></li>
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

Over the next two days, we'll be working with LLM agents. These consist of a scaffolding programs interacting with an LLM API. We'll also build two tasks, a simple and a complex one, in order to see how LLM agents act.

We'll begin by learning building a simple Arithmetic Task and Arithmetic Agent. This should teach you the basics of function calling via the OpenAI API (Anthropic's API has minor differences, but operates in essentially the same way). Then, once we're comfortable with function calling and the general setup of LLM agents and tasks, we'll move on to building a more complex agent that plays the [Wikipedia Game](https://en.wikipedia.org/wiki/Wikipedia:Wiki_Game).

Then we'll explore a variety of elicitation methods. These are methods for getting the best capabilities out of models, and is crucial for evaluating LLM agents. Elicitation tries to answer the question "Can the model do this?" Unforunately, we'll almost never be able to *prove* that the model doesn't have a capability, and will only be able to say that with some effort, we couldn't *get* the model show this capability. This means we'll have to put a lot of effort into trying to exhibit the behavior in models (to have the highest confidence when we make a claim that the model can't exhibit this behavior). This will involve:
- Improving our prompting
- Improving our tools
- Improving the way the histories are stored
- Ensuring the model can access good information.

Each exercise will have a difficulty and importance rating out of 5, as well as an estimated maximum time you should spend on these exercises and sometimes a short annotation. You should interpret the ratings & time estimates relatively (e.g. if you find yourself spending about 50% longer on the exercises than the time estimates, adjust accordingly). Please do skip exercises / look at solutions if you don't feel like they're important enough to be worth doing, and you'd rather get to the good stuff!

## Content & Learning Objectives

#### 1️⃣ Intro to LLM Agents

> ##### Learning objectives
>- Understand why we want to evaluate LLM agents.
>- Red resources about LLM agent evaluations to understand the current state of the field.
>- Understand the common failure modes of LLM agents.
>

#### 2️⃣ Simple Arithmetic Agent

> ##### Learning objectives
>- Understand that a LLM agent is just a "glorified for-loop" (of the scaffolding program interacting with the LLM API).
>- Learn how to use function calling to allow LLMs to use external tools.
>- Understand the main functionalities of an LLM agent.
>

#### 3️⃣ More Complex Agent: WikiGame

> ##### Learning objectives
>- Get comfortable building a more complex task
>- Understand how to build a more complex agent
>- Observe the failure modes of a more complex agent

#### 4️⃣ Elicitation

> ##### Learning objectives
>- Understand the importance of elicitation in evaluating LLM agents
>- Understand the different methods of elicitation
>- Understand how to improve prompting, tools, history storage, and information access in LLM agents

#### 5️⃣ Bonus
> ##### Learning objectives
>- Implement additional tools for elicitation
>- Explore some of your own ideas for elicitation
>- Explore some of the current research in elicitation


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
import part4_llm_agents.tests as tests

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

client = OpenAI()
```
                
""",
        unsafe_allow_html=True,
    )
