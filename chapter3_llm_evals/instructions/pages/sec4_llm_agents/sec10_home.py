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

#### Why evaluate LLM agents?

<!--
Points to make - "Why evaluate LLM agents":
- overall note: I think this could heavily be based on this [video](https://www.youtube.com/watch?v=KO72xvYAP-w) from METR
- The purpose of agent evals is to **unlock and measure the full capabilities of a model**, to avoid underestimating the model and to better estimate the **ceiling** of their capability and potential to cause harm. 
- Models often fail in easily-fixable ways. For example, when it is solving a hacking task, it
    - Can refuse due to ethics or (claimed) inability 
    - Can give up and ask the user for help 
    - Can get stuck in loops 
    - Can hallucinate facts or conclusions [only partly fixable] 
    - Can be limited by primitive tools 
    - Can have bugs
    - ...
- For a model that fails to solve a hacking task, thus deemed safe, there might exist simple fixes (e.g. better prompts, better file manipulation tools) that unlock this dangerous capability. 
- 
- Final point about "quantifying" the amount of scaffolding to make eval results more quantitative
    - Apollo "Science of Evals" 
    - GDM quantify bits
-->

There are at least two reasons for why we might want to evaluate LLM agents.

1. **Measuring the maximum capabilities of a model**

For estimating safety risks, we want to measure the **ceiling** of dangerous capabilities. LLMs on their own often fail in easy-to-fix ways, as you will see. For example:

- They often claim to be incapable of tasks that they can actually perform.
- They can easily get stuck in loops.
- They can give up and ask the user for help
- They can hallucinate facts, or even misunderstand their own prior reasoning and hallucinate a faulty conclusion.
- They can be limited by primitive tools.
- They can be overly or underly sensitive to information in their prompts.
- They can have bugs.

This means that when a model fails to accomplish a task, it may still have the raw capability to succeed, but just require simple fixes that will unlock this capability. We want to eliminate the possibility of large capability improvements from relatively little effort, because this means our evals would have underestimated the true capability and risks associated with a model. Therefore, we want to try hard to elicit their raw capabilities via scaffolding, so that we can evaluate LLMs at their *best*.


2. **Measuring the alignment of LLMs in agentic scenarios**

We do not know if our current alignment techniques (e.g. supervised fine-tuning, RLHF) for aligning LLM chatbots will still work when LLMs are acting as agents in more complex scenarios. It is possible that these methods will not generalize well to agentic scenarios, and we want to test this.

We know today that LLMs are being used as more than just chatbots. Since the release of ChatGPT, the use of LLMs as agentic systems has grown signficantly. These agents started off rather disappointingly initially, when they were based on GPT-3.5. However as more powerful LLMs come out and AI companies ensure their LLMs are better at tool-use, these agents are improving rapidly.


<details><summary>Further resources on LLM agent evaluations</summary>

- [Evaluating Language-Model Agents on Realistic Autonomous Tasks](https://evals.alignment.org/Evaluating_LMAs_Realistic_Tasks.pdf) (Kinniment et al., ARC Evaluations Team (now METR), 2023)
- [Large Language Models can Strategically Deceive their Users when Put Under Pressure](https://arxiv.org/pdf/2311.07590) (Scheurer et al., Apollo Research, ICLR 2024)
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) (Lilian Weng, OpenAI Safety Team, 2023)
- [AXRP Episode 34 - AI Evaluations with Beth Barnes](https://www.alignmentforum.org/posts/vACr4DExfeRMaCoo7/axrp-episode-34-ai-evaluations-with-beth-barnes) (Daniel Filan, 2024)

</details>

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
