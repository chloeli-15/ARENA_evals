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
# MCQ Benchmarks: Exploration

> ##### Learning objectives
> 
> - Understand what benchmarks are, and what makes a good benchmark
> - Understand the pros/cons of MCQ benchmarks
> - Understand what are good vs bad questions
> - Learn how to use LLMs to generate and improve questions

# 2️⃣ Building MCQ benchmarks: Exploration

Broadly speaking, a "benchmark" refers a procedure for evaluating what the state-of-the-art performance is for accomplishing a particular task. This typically includes specifying a dataset, but also specifications of a task or set of tasks to be performed on the dataset and how to score the result, so that the benchmark can be used in a principled, standardized way that gives common ground for comparison.

**What makes a good benchmark?** This [article](https://www.safe.ai/blog/devising-ml-metrics) by Dan Hendrycks and Thomas Woodside gives a good summary of the key qualities of a good benchmark.

<details><summary><b>What are the pros and cons of MCQs?</b></summary>

The advantages of MCQ datasets, compared evaluating free-form response or more complex agent evals, are mainly to do with cost and convenience:

* Cheaper and faster to build
* Cheaper and shorter to run (compared to long-horizon tasks that may take weeks to run and have hundreds of thousands of dollars of inference compute cost)
* May produce more replicable results (compared to, for example, training another model to evaluate the current model, as each training run could give slightly different results) 

The big disadvantage is that it is typically a worse (less realistic, less accurate) proxy for what you want to measure

</details>

### Exercise - Design eval questions
```c
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵🔵

You should spend up to 1.5 hour on this exercise.
```
The goal of this exercise is to generate 20 MC questions that measures your target property in a way you are happy with, with help from AI models. 

**Make your specification measurable:**
1. Read your specification of the target property. What questions and answers would give you high evidence about this property? Come up with 2-4 example MC questions. Each MCQ should contain information on the following fields:
    - "question": The question itself.
    - "answers": 2 or 4 possible answers. 
    - "answer_matching_behavior": The answer that reflects the target property (i.e. the answer label).
    - (Optional) "system": You may include a system prompt that gives more context on the role or task of the AI.
    - (Optional) "category": You may have sub-categories of the target property
    - ...
    
2. Do these questions still relate to the threat model? If so, where do they fit in? If not, your threat model or your questions or both could be flawed.

**Red-team** 🔁   

3. What other factors than the target property (i.e. confounders) could cause the model to give the same answers? Importantly, note that there will always be other factors/dependencies that will cause the model to give a certain answer in addition to the target property (e.g. A power-seeking question in the context of education may depend on the model's preferences for education). In principle, this will be averaged out across many questions, thus still allow us to get a valid signal on the target property. Only **systematic** coufounders (i.e. the same bias present in many questions of the same type) are problematic because they **bias the result towards a certain direction that will not be averaged out**. 
4. Ask your example questions to the model and see how it responds:
    - Does the answer show that the model correctly understood the question? If not, you will need to rephrase or come up with a different question.
    - Given that the model understands the question, is there another question that is more reasonable? (E.g. You designed a decision-making question expecting the model to take one of two actions, but the model responded with a list of factors that it would need to know to make a decision. You may then choose to include information on these factors in your question, so that the model can reach a decision a more naturally.)
5. Ask your question to other people: what is their answer and why?
    - Do they find this difficult to understand? Do they find the answer choices awkward?
    - If you ask them to choose the power-seeking answer, are they able to? If not, this may imply that the answer does not accurately reflect power-seeking.

**Refine** 🔁 - Likely, you will realize that your current specification and examples are imprecise or not quite measuring the right thing. Iterate until you’re more happy with it.

6. Ask the model to improve your example questions. Ask the model to generate new evaluation questions for the target property using your questions as few-shot examples, then pick the best generated examples and repeat. Play around and try other things!
7. Generate 20 MCQs you are happy with, with model's help. **Store the list of MCQ items as a list of dictionaries in a JSON file**.

#### Our Example: Tendency to seek power - MCQ Design

<img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/ch3-red-teaming-mcq.png" width=1200>


<img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/ch3-power-seeking-example-qs.png" width=1100>
""", unsafe_allow_html=True)