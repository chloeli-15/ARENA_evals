import streamlit as st


def section():
    st.sidebar.markdown(
        r"""
## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#introduction'>Introduction</a></li>
    <li class='margtop'><a class='contents-el' href='#content-learning-objectives'>Content & Learning Objectives</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#1-advanced-api-calls'>Advanced API Calls</a></li>
        <li><a class='contents-el' href='#2-dataset-generation'>Dataset Generation</a></li>
        <li><a class='contents-el' href='#3-dataset-quality-control'>Dataset Quality Control</a></li>
        <li><a class='contents-el' href='#4-putting-it-together'>Putting It Together: Generation-Evaluation</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
</ul>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""
# [3.2] - Dataset Generation

<!--### Colab: [**exercises**](https://colab.research.google.com/drive/1lDwKASSYGE4y_7DuGSqlo3DN41NHrXEw?usp=sharing) | [**solutions**](https://colab.research.google.com/drive/1bZkkJd8pAVnSN23svyszZ3f4WrnYKN_3?usp=sharing)-->

Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-uk/shared_invite/zt-28h0xs49u-ZN9ZDbGXl~oCorjbBsSQag), and ask any questions on the dedicated channels for this chapter of material.

You can toggle dark mode from the buttons on the top-right of this page.

Links to other chapters: [**(0) Fundamentals**](https://arena3-chapter0-fundamentals.streamlit.app/), [**(1) Interpretability**](https://arena3-chapter1-transformer-interp.streamlit.app/), [**(2) RL**](https://arena3-chapter2-rl.streamlit.app/).

<img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/ch3-evals-cover.jpeg" width="350">

## Introduction

The goal of this section is to learn how to use LLMs to generate and refine an evaluation dataset. By the end, you will have generated a (hopefully) high-quality dataset of ~300 questions.

The method used is from the [Perez et al (2022)](https://https://arxiv.org/abs/2212.09251). They came up with a procedure for using LLMs to automate the bulk of eval question generation and filtering, while redirecting human effort to instructing the LLM (e.g. to write example questions and rubric), in order to greatly scale up the size of eval dataset that can be generated cheaply. 


There is a lot more continuity between chapter [3.1] to [3.3] in this chapter than other chapters. Chapter [3.1] to [3.3] teaches how to design, generate and run a MCQ eval benchmark. In the example used, we will be measuring models' **tendency to seek power** as our evaluation target. You can approach chapter [3.2] in 2 ways:
1. You can choose to build your own eval dataset for your chosen model property. For this, you need to have ~20 questions to use as examples for generating more. We strongly recommend to go through chapter [3.1] first, which explains how to design evaluations and question specifications.
2. You can choose to replicate (and improve) our provided power-seeking eval dataset. For this, you can follow the exercises directly without any example questions.

Each exercise will have a difficulty and importance rating out of 5, as well as an estimated maximum time you should spend on these exercises and sometimes a short annotation. You should interpret the ratings & time estimates relatively (e.g. if you find yourself spending about 50% longer on the exercises than the time estimates, adjust accordingly). Please do skip exercises / look at solutions if you don't feel like they're important enough to be worth doing, and you'd rather get to the good stuff!

## Content & Learning Objectives


#### 1️⃣ Advanced API Calls

> ##### Learning objectives
>
> - Learn how to use the structured output feature of the OpenAI API to generate multiple-choice questions


#### 2️⃣ Dataset Generation

> ##### Learning objectives
> 
> - Understand what benchmarks are, and what makes a good benchmark
> - Understand the pros/cons of MCQ benchmarks
> - Understand what are good vs bad questions
> - Learn how to use LLMs to generate and improve questions

#### 3️⃣ Dataset Generation

> ##### Learning objectives

## Setup

```python
import os
from pathlib import Path
import sys
from dotenv import load_dotenv
import openai
from openai import OpenAI
import random
import numpy as np
import pandas as pd
import time
import math
import json
import itertools
from dataclasses import dataclass, field, asdict, fields
from pydantic import BaseModel
from datetime import datetime
import operator
import types
from typing import List, Optional, Protocol, Literal, Callable, Dict, Any, Tuple, ClassVar
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Make sure exercises are in the path
chapter = r"chapter3_llm_evals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part2_dataset_generation").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)

from utils import import_json, save_json, retry_with_exponential_backoff, apply_system_format, apply_assistant_format, apply_user_format, apply_message_format, pretty_print_questions, tabulate_model_scores, plot_score_by_category, plot_simple_score_distribution, print_dict_as_table
import part2_dataset_generation.tests as tests
```
                
""",
        unsafe_allow_html=True,
    )
