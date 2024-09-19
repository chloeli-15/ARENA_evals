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

# Intro to LLM Agents

#### What is an LLM agent?
<!---
Points to make:
- "LLM agent" - a "scaffolding" program (i.e. Python program) that interacts with an LLM API. Include a version of METR "Evaluating Language-Model Agents on Realistic Autonomous Tasks" Figure 2
    - Define scaffolding
- More schematic breakdown of possible scaffolding: "tools" (describe what this means, what "tool calling" is), "memory" (Probably better move to "Build Agent" section! I've expanded this section there)
- Mention list of examples of prominent LLM agents:
    - [Minecraft LM Agent](https://arxiv.org/abs/2305.16291)
    - [AutoGPT](https://autogpt.net/) and [LangChain](https://www.langchain.com/)

==========================
--->
An LLM **agent** consists of **a scaffolding program interacting with an LLM API**. This typically involves a loop of the following steps:

1. The scaffolding program sends instructions to the LLM, typically containing information on the task goal, the actions available to the LLM, and relevant task information. 
2. The LLM processes the input and outputs an action in text (e.g. "calls" the `calculate()` tool in text). 
3. The scaffolding program executes the action and returns the outcome (e.g. it runs the `calculate()` function in the background and returns the output to the agent).
4. The LLM observes the results and decides the next action.
5. Repeating the cycle until the task is complete

The two basic components of scaffolding are:
- Tool calling: This allows LLMs to use tools by providing a text description of the tool. The LLM can choose to use this tool by "calling" it in its text output. If it uses a tool, the scaffolding will execute this tool on the LLM's behalf (e.g. by running a python function, sending request to an external API etc.) and return the result of this tool call to the agent.
- Prompting: This describes the task state to the LLM, assists it in its tool use, or instruct it to use chain-of-thought to give the LLM more "thinking time" etc.

<details><summary><b>Examples of LLM Agents</b></summary>

- [Voyager](https://arxiv.org/abs/2305.16291) (Minecraft LLM Agent)

- [AutoGPT](https://autogpt.net/)

- [LangChain](https://www.langchain.com/)

</details>

<!-- [Insert METR diagram]-->


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
