#%%
import json
from inspect_ai.log import read_eval_log
from inspect_ai import Task, eval, task
from inspect_ai.dataset import example_dataset, FieldSpec, json_dataset, Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import (
  chain_of_thought, generate, self_critique
)
from itertools import product
#%%

#Just a dumb test.

with open("datasets/power-seeking-2c-gen.json", "r") as f:
    dataset = json.load(f)
    print(list(dataset[6]["answers"].values()))
    for i in dataset:
        print(type(i["answers"]))
#%%

#When you set your OpenAI API key, it should be named OPENAI_API_KEY, since the inspect-ai library looks for a key with that name in the os environment by default as far as I can tell. Probably can change what the model looks for somewhere, but better just to flag that OPENAI_API_KEY is best naming convention (I imagine a similar thing happens for anthropic).

def record_to_sample(record):
    '''
    This is a function which takes in the eval that we have written, and puts it in a format which is usable by the inspect_ai framework.
    '''
    return Sample(
        input=[
                    {
                        "role" : "user", 
                        "content" : record["question"]
                    },
                    {
                        "role" : "system",
                        "content" : record["system"]
                        #This includes system prompt when loading the dataset. Then there is no need to include the system_message() function later when we define the task plan.
                    }
            ],
        target=record["answer_matching_behavior"],
        choices = list(record["answers"].values()),
        metadata = {"label" : record["label"]}
    )

power_seeking_dataset = json_dataset("./datasets/power-seeking-2c-gen.json", record_to_sample)

#%%


@task
def power_seeking():
    """
    This returns a Task object (from the inspect_ai library). This object gets referenced when we call the eval() function below.
    The Task consists of a dataset, a plan, and a scorer for the task.
    
     - The dataset is the dataset on which we are evaluating the model.
    
     - The plan is a collection of "solver" objects from the inspect_ai library (of course you can also write your own). But broadly this tells the model how to generate its response to the evaluation question.

     - The scorer is how we will score the model's response. There are a few built in scorers. The "solver" and "scorer" objects come with a template prompt. These prompts can be modified (for example, the built-in model_graded_fact() scorer has a default prompt which talks about "expert answers" in a way that may not be suited to an alignment benchmark).
    """
    '''
            The "plan" for the eval is stored here. The plan will look something like:
            
            [
            chain_of_thought(),
            generate(),
            self_critique(model = "openai/gpt-3.5-turbo")
            ],
            and this will do the following:

            - Prompt the model to think about its answer to the question step-by-step (this prompt comes from the template for chain_of_thought, which can be modified).

            - Generate the chain of thought reasoning.

            - Prompt the model to critique its reasoning which it has generated thus far (self_critique doesn't need generate to be called, since it automatically calls generate). You can have a different model perform the self_critique than the one you are evaluating, which is why we specify that openai/gpt-3.5-turbo is doing the critique.

            '''
    return Task(
        dataset=power_seeking_dataset,
        plan=[
          chain_of_thought(), 
          generate(), 
          self_critique(model = "openai/gpt-3.5-turbo")
        ],
        scorer=model_graded_fact(model = "openai/gpt-3.5-turbo"),
    )

# 'grid' will be a permutation of all parameters
#grid = list(product(*(params[name] for name in params)))

# run the evals and capture the logs
'''
Commented out since I currently do not want this to run. Will run it in future though.
logs = eval(
'''
    #This runs an eval and returns a file containing the results of the eval
'''
    power_seeking(),
    model="openai/gpt-3.5-turbo",
)
'''

#%%
log = read_eval_log("./logs from inspect library/2024-07-04T15-41-30+01-00_theory-of-mind_X5qnDZu9VYzkfUbhNnoYwD.json")



print(log.status) 
# Started, success, or failure



print(log.eval) 
# Task, model, creation time, various IDs used to store the log.




print(log.plan) 
#The plan for the eval, whether you got logits, what sampling you used, a logit bias, a seed, temperature, top_p, whether you capped token generation, if there was a system_message


print(log.samples[0]) 
#Do not use log.samples without indexing. log.samples consists of a huge amount of information for each thing you tested the model on, gets very big on even medium-sized evals.


print(log.results) 
# Prints accuracy, bootstrap standard deviation



print(log.stats)
#Useful stats: How many tokens were fed to the model, how many tokens it generated, and times.



print(log.logging) 
#Logging messages


print(log.error) 
#Returns the error if there was an error, None if no error.



#%%
'''
Ideas for things to include in the rest of run_evals:

Build your own solver using inspect-ai's solver class. partially DONE.

TODO: Build your own scorer using inspect-ai's scorer class.

TODO: Absolutely NEED to be able to plot interesting things. Can run multiple times, can run with different prompts, can run with/without CoT, with CoT+self-critique. Use ChatGPT + plotly.

    -TODO: Figure out how to use the log stuff above to plot results, probably from log.results. 

TODO, maybe not even: Perhaps we can first introduce basic tool-use here, instead of the LM agent evals section, since inspect-ai provides tool functionality. Only something simple (if it would even help).


'''


#%%
print(log.results)
print(log.eval)
print(log.plan)
#%%
import plotly.graph_objects as go

# Prepare the data
metrics = {
    'Accuracy': 0.66,
    'Bootstrap Std': 0.04539381455661112
}
labels = list(metrics.keys())
values = list(metrics.values())

# Create the bar plot
fig = go.Figure()

# Add bars to the plot
fig.add_trace(go.Bar(
    x=labels,
    y=values,
    text=[f"{v:.3f}" for v in values],
    textposition='auto',
    marker_color=['#1f77b4', '#ff7f0e']  # Custom colors for the bars
))

# Customize the layout
fig.update_layout(
    title='Evaluation Metrics',
    xaxis_title='Metrics',
    yaxis_title='Values',
    yaxis=dict(range=[0, 1]),  # Adjust range if necessary
    template='plotly_white',
    annotations=[
        dict(
            x=labels[0],
           
            y=values[0],
            text=f"Accuracy: {values[0]:.3f}",
            showarrow=False,
            arrowhead=2,
            ax=0,
            ay=-40
        ),
        dict(
            x=labels[1],
            y=values[1],
            text=f"Bootstrap Std: {values[1]:.5f}",
            showarrow=False,
            ax=0,
            ay=-40
        )
    ]
)

# Add a secondary axis if needed (example, not used here as values are too close)
fig.update_layout(
    yaxis2=dict(
        title='Bootstrap Std',
        overlaying='y',
        side='right'
    )
)

# Show the plot
fig.show()
#%%