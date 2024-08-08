#%%
import plotly.graph_objects as go
import plotly
from collections import Counter, defaultdict
import json
import os

def load_questions_from_json_files(directory):
    questions = []
    for filename in directory:
        with open(filename, 'r') as file:
            data = json.load(file)
            if isinstance(data, list):
                questions.extend(data)
            elif isinstance(data, dict):
                questions.append(data)
    return questions

directory = ["./scores/quality/2c-generated-4.json","./scores/quality/2c-generated-5.json", "./scores/quality/2c-generated-6.json","./scores/quality/2c-generated-7.json"]

questions = load_questions_from_json_files(directory)



#%%
'''
This is going to plot the full diversity of questions: (resource-seeking, yes), (resource-seeking,no), (upward-mobile,yes), (upward-mobile,no)
'''

# Initialize counters
counts = defaultdict(lambda: {'yes': defaultdict(int), 'no': defaultdict(int)})

for q in questions:
    label = q['label']
    score = q['score']
    answer = 'yes' if 'A' in q['answer_matching_behavior'] else 'no'
    counts[label][answer][score] += 1

# Get all unique scores
all_scores = sorted(set(score for label in counts for answer in counts[label] for score in counts[label][answer]))

# Create the grouped bar chart
fig = go.Figure()

labels = ['resource-seeking', 'upward-mobile']
answers = ['yes', 'no']
colors = {
    ('resource-seeking', 'yes'): 'rgba(0, 255, 0, 0.5)',
    ('resource-seeking', 'no'): 'rgba(255, 0, 0, 0.5)',
    ('upward-mobile', 'yes'): 'rgba(0, 0, 255, 0.5)',
    ('upward-mobile', 'no'): 'rgba(255, 165, 0, 0.5)'
}

for label in labels:
    for answer in answers:
        fig.add_trace(go.Bar(
            name=f"{label} - {answer.capitalize()}",
            x=all_scores,
            y=[counts[label][answer][score] for score in all_scores],
            marker_color=colors[(label, answer)],
        ))

# Update the layout
fig.update_layout(
    title='Distribution of Question Scores by Label and Answer',
    xaxis_title='Score',
    yaxis_title='Number of Questions',
    barmode='group',
    bargroupgap=0.1,
    xaxis=dict(
        tickmode='array',
        tickvals=all_scores,
        ticktext=[str(score) for score in all_scores]
    ),
    legend_title_text='Label - Answer'
)

# Show the plot
fig.show()

#%%
'''
This plots just upward-mobile vs resource-seeking, along with scores


'''
upward_mobile_counts = defaultdict(int)
resource_seeking_counts = defaultdict(int)

for q in questions:
    if q['label'] == 'upward-mobile':
        upward_mobile_counts[q['score']] += 1
    elif q['label'] == 'resource-seeking':
        resource_seeking_counts[q['score']] += 1

# Get all unique scores
all_scores = sorted(set(list(upward_mobile_counts.keys()) + list(resource_seeking_counts.keys())))

# Create the grouped bar chart
fig = go.Figure(data=[
    go.Bar(name='Upward-mobile', x=all_scores, y=[upward_mobile_counts[score] for score in all_scores]),
    go.Bar(name='Resource-seeking', x=all_scores, y=[resource_seeking_counts[score] for score in all_scores])
])

# Update the layout
fig.update_layout(
    title='Distribution of Question Scores by Label',
    xaxis_title='Score',
    yaxis_title='Number of Questions',
    barmode='group',
    xaxis=dict(
        tickmode='array',
        tickvals=all_scores,
        ticktext=[str(score) for score in all_scores]
    )
)

# Show the plot
fig.show()

#%%
'''
This plots "yes" vs "no" answers.
'''
counts = {'yes': defaultdict(int), 'no': defaultdict(int)}

for q in questions:
    score = q['score']
    answer = 'yes' if 'A' in q['answer_matching_behavior'] else 'no'
    counts[answer][score] += 1

# Get all unique scores
all_scores = sorted(set(score for answer in counts for score in counts[answer]))

# Create the grouped bar chart
fig = go.Figure()

colors = {
    'yes': 'rgba(0, 255, 0, 0.5)',  # Light green
    'no': 'rgba(255, 0, 0, 0.5)'    # Light red
}

for answer in ['yes', 'no']:
    fig.add_trace(go.Bar(
        name=answer.capitalize(),
        x=all_scores,
        y=[counts[answer][score] for score in all_scores],
        marker_color=colors[answer]
    ))

# Update the layout
fig.update_layout(
    title='Distribution of Question Scores by Answer',
    xaxis_title='Score',
    yaxis_title='Number of Questions',
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
    xaxis=dict(
        tickmode='array',
        tickvals=all_scores,
        ticktext=[str(score) for score in all_scores]
    ),
    legend_title_text='Answer'
)

# Show the plot
fig.show()
#%%