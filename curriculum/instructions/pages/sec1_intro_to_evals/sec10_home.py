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
# [3.1] - Intro to LLM Evaluations

### Colab: [**exercises**](https://colab.research.google.com/drive/1lDwKASSYGE4y_7DuGSqlo3DN41NHrXEw?usp=sharing) | [**solutions**](https://colab.research.google.com/drive/1bZkkJd8pAVnSN23svyszZ3f4WrnYKN_3?usp=sharing)

Please send any problems / bugs on the `#errata` channel in the [Slack group](https://join.slack.com/t/arena-uk/shared_invite/zt-28h0xs49u-ZN9ZDbGXl~oCorjbBsSQag), and ask any questions on the dedicated channels for this chapter of material.

You can toggle dark mode from the buttons on the top-right of this page.

Links to other chapters: [**(0) Fundamentals**](https://arena3-chapter0-fundamentals.streamlit.app/), [**(1) Interpretability**](https://arena3-chapter1-transformer-interp.streamlit.app/), [**(2) RL**](https://arena3-chapter2-rl.streamlit.app/).

<img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/ch3-evals-cover.jpeg" width="350">

## Introduction

This is an introduction to designing and running LLM evaluations. The exercises are written in partnership with [Apollo Research](https://www.apolloresearch.ai/), and designed to give you the foundational skills for doing safety evaluation research on language models. 

There is more continuity between days in this chapter than other chapters. Chapter [3.1] to [3.3] focuses on designing and building a multiple-choice (MC) questions benchmark for one chosen target property, and using it to run evals on models. Day 3.4 to 3.5 focuses on building an LLM-agent that plays the Wikipedia racing game, and eliciting agent behavior.

Chapter [3.1] is the only non-coding day of the entire curriculum. The exercises focus on designing a threat model and specification for your evaluation target, and using this to design questions. These are the first steps and often the hardest/most creative part of real eval research. In practice this is often done by a team of researchers, and can take weeks or months. The exercises are designed to give you a taste of this process. The exercise questions are open-ended, and you're not expected to have a perfect answer by the end.

Each exercise will have a difficulty and importance rating out of 5, as well as an estimated maximum time you should spend on these exercises and sometimes a short annotation. You should interpret the ratings & time estimates relatively (e.g. if you find yourself spending about 50% longer on the exercises than the time estimates, adjust accordingly). Please do skip exercises / look at solutions if you don't feel like they're important enough to be worth doing, and you'd rather get to the good stuff!

## Content & Learning Objectives


#### 1️⃣ Threat-modeling 

> ##### Learning objectives
>
> - Understand the purpose of evaluations and what safety cases are
> - Understand the types of evaluations
> - Understand what a threat model looks like, and why it's important
> - Learn the process of building a threat model and specification for an evaluation
> - Learn how to use models to help you in the design process


#### 2️⃣ MCQ Benchmarks: Exploration

Here, we'll to generate and refine 20 MC questions until we are happy that they evaluate the target property. We will use LLMs to help us generate, red-team and refine these questions.

> ##### Learning objectives
> 
> - Understand what benchmarks are, and what makes a good benchmark
> - Understand the pros/cons of MCQ benchmarks
> - Understand what are good vs bad questions
> - Learn how to use LLMs to generate and improve questions


## Setup

As 3.1 is a non-coding day, there is no python set-up. You will need to have a text editor open to write your threat model and specification. We recommend copying the questions and answering them in Google Docs so they can be easily shared with your pair.
                
""", unsafe_allow_html=True)