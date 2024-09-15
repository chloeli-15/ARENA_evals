import streamlit as st


def section():
    st.sidebar.markdown(
        r"""
## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#introduction'>Introduction</a></li>
    <li class='margtop'><a class='contents-el' href='#content-learning-objectives'>Content & Learning Objectives</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#1-threat-modeling'>Threat-modeling</a></li>
        <li><a class='contents-el' href='#2-mcq-benchmarks-exploration'>MCQ Benchmarks: Exploration</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
</ul>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""
# [3.1] - Intro to LLM Evaluations

<!-- ### Colab: [**exercises**](https://colab.research.google.com/drive/1lDwKASSYGE4y_7DuGSqlo3DN41NHrXEw?usp=sharing) | [**solutions**](https://colab.research.google.com/drive/1bZkkJd8pAVnSN23svyszZ3f4WrnYKN_3?usp=sharing) -->

Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-uk/shared_invite/zt-28h0xs49u-ZN9ZDbGXl~oCorjbBsSQag), and ask any questions on the dedicated channels for this chapter of material.

You can toggle dark mode from the buttons on the top-right of this page.

Links to other chapters: [**(0) Fundamentals**](https://arena3-chapter0-fundamentals.streamlit.app/), [**(1) Interpretability**](https://arena3-chapter1-transformer-interp.streamlit.app/), [**(2) RL**](https://arena3-chapter2-rl.streamlit.app/).

<img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/ch3-evals-cover.jpeg" width="350">

## Introduction

The Threat-Modelling section of chapter [3.1] is the only non-coding section of the curriculum. The exercises focus on designing a threat model and specification for a chosen model property. We use this to design an evaluation (eval) to measure this target property. These are the first steps and often the hardest/most "creative" part of real eval research. In practice this is often done by a team of researchers, and can take weeks or months. The exercises are designed to give you a taste of this process. The exercise questions are open-ended, and you're not expected to have a perfect answer by the end.

Each exercise will have a difficulty and importance rating out of 5, as well as an estimated maximum time you should spend on these exercises and sometimes a short annotation. You should interpret the ratings & time estimates relatively (e.g. if you find yourself spending about 50% longer on the exercises than the time estimates, adjust accordingly). Please do skip exercises / look at solutions if you don't feel like they're important enough to be worth doing, and you'd rather get to the good stuff!

## Overview of Chapter [3.1] to [3.3]

The goal of chapter [3.1] to [3.3] is to **build and run an alignment evaluation benchmark from scratch to measure a model property of interest**. The benchmark we will build contains:
- a multiple-choice (MC) question dataset of ~300 questions, 
- a specification of the model property, and 
- how to score and interpret the result. 

We will use LLMs to generate and filter our questions, following the method from [Ethan Perez et al (2022)](https://arxiv.org/abs/2212.09251). In the example that we present, we will measure the **tendency to seek power** as our evaluation target. 

* For chapter [3.1], the goal is to craft threat models and a specification for the model property you choose to evaluate, then design 20 example eval questions that measures this property. 
* For chapter [3.2], we will use LLMs to expand our 20 eval questions to generate ~300 questions.
* For chapter [3.3], we will evaluate models on this benchmark using the UK AISI's [`inspect`](https://inspect.ai-safety-institute.org.uk/) library.

<details><summary>Why have we chose to build an alignment evals specifically?</summary>

The reason that we are focusing primarily on alignment evals, as opposed to capability evals, is that we are using LLMs to generate our dataset, which implies that the model must be able to solve the questions in the dataset. We cannot use models to generate and label correct answers if they are not capable of solving the questions. (However, we can use still use models to generate parts of a capability eval question if we do not require the model to be able to solve them.)

</details>

<img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/ch3-day1-3-overview.png" width=1000>

<!-- TODO: Make this diagram horizontal -->

Diagram note: The double arrows indicate that the steps iteratively help refine each other. In general, the threat model should come first and determine what model properties are worth evaluating. However, in this diagram we put "Target Property" before "Threat-modeling & Specification". This is because in the exercises, we will start by choosing a property, then build a threat model around it because we want to learn the skill of building threat models. 

## Content & Learning Objectives


#### 1️⃣ Threat-modeling 

> ##### Learning objectives
>
> - Understand the purpose of evaluations and what safety cases are
> - Understand the types of evaluations
> - Learn what a threat model looks like, why it's important, and how to build one
> - Learn how to write a specification for an evaluation target
> - Learn how to use models to help you in the design process


#### 2️⃣ MCQ Benchmarks: Exploration

Here, we'll to generate and refine 20 MC questions until we are happy that they evaluate the target property. We will use LLMs to help us generate, red-team and refine these questions.

> ##### Learning objectives
> 
> - Understand what benchmarks are, what makes a good benchmark, and pros/cons of MCQ benchmarks
> - Learn the basics of API calls and what messages are
> - Understand the process of designing questions according to a specification
> - Learn how to use LLMs to generate, red-team and improve questions


## Setup

```python
import os
from pathlib import Path
import sys
import openai
from openai import OpenAI
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional 
import random
import warnings
import time

# Make sure exercises are in the path
chapter = r"chapter3_llm_evals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part1_intro").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)

from utils import import_json, save_json, retry_with_exponential_backoff, pretty_print_questions,load_jsonl
```
                
""",
        unsafe_allow_html=True,
    )
