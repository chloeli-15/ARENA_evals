import streamlit as st



def section():
    st.sidebar.markdown(
        r'''

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
</ul>''',
        unsafe_allow_html=True,
    )

    st.markdown(r'''
# # 4️⃣ Bonus: Logs and plotting

Now that we've finished running our evaluations, we can look at the logs and plot the results. First, let's introduce the inspect log file.

## Intro to `Inspect`'s EvalLogs

We can read a log as an `EvalLog` object by using `Inspect`'s `read_eval_log()` function in the following syntax:

```python
log = read_eval_log(r"path/to/log/file.json")
```
                
The `EvalLog` object has the following attributes, from the `Inspect` documentation:

<blockquote>

| Field       | Type     | Description |
| -------------- | -------------- | --------------------------------- |
| `status`      | `str`         | Status of evaluation (`"started"`, `"success"`, or `"error"`). |
| `eval` | `EvalSpec` | Top level eval details including task, model, creation time, dataset, etc. |
| `plan` | `EvalPlan` | List of solvers and model generation config used for the eval. |
| `samples` | `list[EvalSample]` | Each sample evaluated, including its input, output, target, and score. |
| `results` | `EvalResults` | Aggregate results computed by scorer metrics. |
| `stats` | `EvalStats` | Model usage statistics (input and output tokens). |
| `logging` | `list[LoggingMessage`] | Any logging messages we decided to include |
| `error` | `EvalError` | Error information (if `status == "error"`) including the traceback | 

<br>
</blockquote>

For long evals, you should not call `log.samples` without indexing, as it will return an *incredibly* long list. Before accessing any information about logs, you should always first check that the `status` of the log is `"success"`.

Uncomment and run the code below to see what these `EvalLog` attributes look like:


```python
print(log.status) 

#print(log.eval)

#print(log.plan)

#print(log.samples[0])

#print(log.results)

#print(log.stats)

#print(log.logging) 

#print(log.error)
```

## Data Extraction and Plotting        
        
### Exercise (optional) - Extract data from log files
```c
Difficulty:🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵⚪⚪
```

Extract data from the log files that have been generated so far, so that we can write a plotting function to plot the results of the evaluation. Only do this exercise if you want to get familiar with data extraction and plotting. There are no solutions to this exercise, as precisely what you extract will depend on the specifics of your evaluation.

```python
#Extract data here
```
                


### Exercise (optional) - Use plotly (and claude) to plot the results of your evaluation
```c
Difficulty:🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵⚪⚪
```
Now plot some graphs indicating the results of your evaluation. Use `plotly` and Claude/ChatGPT significantly to help you do this. There are no solutions to this exercise, as precisely what you decide to plot will depend on the specifics of your evaluation.


```python
#Plot some cool graphs here
```
                
''',
                unsafe_allow_html=True,

    )