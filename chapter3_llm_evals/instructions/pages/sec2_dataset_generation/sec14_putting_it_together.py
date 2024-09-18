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
        <li><a class='contents-el' href='#4-putting-It-together'>Putting it Together: Generation-Evaluation</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
</ul>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        r'''
# Putting It Together: Generation-Evaluation

> ### Learning Objectives
> 
> - Generate a dataset of ~300 questions using LLMs
> - Learn how to improve the generated dataset by iteratively generating, observing flaws, modifying the prompts and rubric, repeat.

Once you have a working model generator and evaluator, you can put them together to generate questions.

- Start with generating and scoring 20 to see if the questions and scores are high-quality. 
- Read the questions and model scores. Edit the prompts and rubric as needed.
- Iterate.
- Generate 300 questions with a quality that you are happy with. 

We have provided a simple API below to generate and evaluate questions. 

```python
# Define the configs
model = "gpt-4o-mini"
total_q_to_gen = 20
threshold = 7
chunk_size = 5
max_workers = 8

# Define file paths to store the generated questions, scores and logs
version = "002" # Increase for each run 
generation_filepath = f"data/generations/questions_{version}.json"
scores_filepath = f"data/scores/scores_{version}.json"
log_filepath = f"data/logs/log_{version}.json"


config = Config(model=model,
                generation_filepath=generation_filepath,
                score_filepath=scores_filepath)
print(config)

# Instantiate and edit the prompts
gen_prompts = GenPrompts()
eval_prompts = EvalPrompts()


print("\n\n======================= GENERATION PROMPTS ==========================\n")
pretty_print_messages(gen_prompts.get_message())
print("\n\n======================= EVALUATION PROMPTS ==========================\n")
pretty_print_messages(eval_prompts.get_message())
```
```python
# Generate and evaluate the dataset
generated_dataset = query_generator(total_q_to_gen, 
                                    config, 
                                    gen_prompts)

scored_dataset = query_evaluator(generated_dataset,
                                 config, 
                                 eval_prompts)

summary = summarize_results(scored_dataset, config, log_filepath)


```
```python
fig_1 = plot_simple_score_distribution(scored_dataset,
                                     score_key="score",
                                    title="Distribution of Question Scores",
                                    x_title="Score",
                                    y_title="Number of Questions",
                                    bar_color='rgb(100,149,237)', 
                                    width=800,
                                    height=600)

fig_2 = plot_score_by_category(scored_dataset,
                            score_key="score",
                            categories={
                                "Target Answer": {
                                    "key": "answers", # key corresponding to answer choices
                                    "correct_key": "answer_matching_behavior", # key corresponding to answer label
                                    "values": {
                                        "Yes": ("Yes.", "Yes"), # set of possible answers that correspond to "Yes"
                                        "No": ("No.", "No") # set of possible answers that correspond to "No"
                                    }
                                }
                            },
                            title="Distribution of Question Scores by Categories",
                            x_title="Score",
                            y_title="Number of Questions",
                            width=800,
                            height=600)
fig_1.show()
fig_2.show()
```
''',
        unsafe_allow_html=True,
    )
