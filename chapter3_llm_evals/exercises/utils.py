import json
import importlib
import openai
import re
from openai import OpenAI
import os
from typing import Any, List, Dict, Optional
from tabulate import tabulate
import textwrap
import random
import time
import plotly.graph_objects as go
from collections import defaultdict
from collections import defaultdict
import math


def import_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def save_json(file_path, data, do_not_overwrite=True):
    if do_not_overwrite:
        try: 
            with open(file_path, "r") as f:
                existing_data = json.load(f)
        except FileNotFoundError:
                existing_data = []
        if isinstance(existing_data, dict):
            existing_data = [existing_data]
        existing_data.extend(data)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

def get_openai_api_key():
    return json.loads(open('secrets.json').read())['OPENAI_API_KEY']


def establish_client_OpenAI():
    '''
    Returns the OpenAI client which can be used for accessing models
    '''
    return OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

def establish_client_anthropic():
    '''
    Returns the anthropic client which can be used for accessing models
    '''
    #Returns anthropic client - Do later.
    pass

def omit(x: dict[str, Any], vars: list[str]) -> dict[str, Any]:
    x = x.copy()
    for var in vars:
        if var in x:
            del x[var]
    return x

def retry_with_exponential_backoff(func, 
                                    retries = 20, 
                                    intial_sleep_time: int = 3, 
                                    jitter: bool = True,
                                    backoff_factor: float = 1.5):
    """
    This is a sneaky function that gets around the "rate limit error" from GPT (GPT has a maximum tokens per min processed, and concurrent processing may exceed this) by retrying the model call that exceeds the limit after a certain amount of time.
    """
    def wrapper(*args, **kwargs):
        sleep_time = intial_sleep_time
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "rate_limit_exceeded" in str(e):
                    sleep_time *=  backoff_factor * (1 + jitter * random.random())
                    time.sleep(sleep_time)
                else:
                    raise
        raise Exception(f"Maximum retries {retries} exceeded")
    return wrapper


def apply_system_format(content : str) -> dict:
    return {
        "role" : "system",
        "content" : content
    }

def apply_user_format(content : str) -> dict:
    return {
        "role" : "user",
        "content" : content
    }

def apply_assistant_format(content : str) -> dict:
    return {
        "role" : "assistant",
        "content" : content
    }

def apply_message_format(user : str, system : Optional[str]) -> List[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages

def pretty_print_questions(questions) -> None:
    """
    Print the model generation response in a structured format.
    Lists within question dictionaries are printed on a single line.

    Args:
    response (str): A JSON-formatted string containing the model's response.
    """
    def print_indented(text: str, indent: int = 0) -> None:
        print(" " * indent + text)

    def print_key_value(key: str, value: Any, indent: int = 0, in_question: bool = False) -> None:
        if isinstance(value, str):
            print_indented(f'"{key}": "{value}"', indent)
        elif isinstance(value, (int, float, bool)):
            print_indented(f'"{key}": {value}', indent)
        elif isinstance(value, dict):
            print_indented(f'"{key}":', indent)
            for k, v in value.items():
                print_key_value(k, v, indent + 2, in_question)
        elif isinstance(value, list):
            if in_question:
                list_str = ", ".join(str(item) for item in value)
                print_indented(f'"{key}": [{list_str}]', indent)
            else:
                print_indented(f'"{key}": [', indent)
                for item in value:
                    if isinstance(item, dict):
                        print_indented("{", indent + 2)
                        for k, v in item.items():
                            print_key_value(k, v, indent + 4, False)
                        print_indented("}", indent + 2)
                    else:
                        print_indented(str(item), indent + 2)
                print_indented("]", indent)
        else:
            print_indented(f'"{key}": {value}', indent)

    try:
        for i, question in enumerate(questions, 1):
            print_indented(f"\nQuestion {i}:", 0)
            if isinstance(question, dict):
                for key, value in question.items():
                    print_key_value(key, value, 2, True)
            else:
                print_indented(str(question), 2)

    except json.JSONDecodeError:
        print("Error: Invalid JSON string")
    except KeyError as e:
        print(f"Error: Missing key in JSON structure: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def tabulate_model_scores(data: List[Dict], 
                          new_column_keys: List[str] = ["score", "model_response"], 
                          max_width: int = 40) -> str:
    table_data = []
    for item in data:
        question_str = ""
        for key, value in item.items():
            if key not in new_column_keys:
                wrapped_value = textwrap.fill(str(value), width=max_width)
                question_str += f"{key}: {wrapped_value}\n\n"
        question_str = question_str.strip()
        
        model_response = textwrap.fill(item.get("model_response", "N/A"), width=max_width)
        
        table_data.append([
            question_str,
            item.get("score", "N/A"),
            model_response
        ])

    headers = ["Question", "Score", "Model Response"]
    return tabulate(table_data, headers=headers, tablefmt="simple_grid", colalign=("left", "center", "left"))

def print_dict_as_table(data: Dict[str, Any]) -> None:
    """
    Print a dictionary as a two-column table using tabulate.
    
    Args:
    data (Dict[str, Any]): The dictionary to be printed.
    
    Returns:
    None: This function prints the table but doesn't return anything.
    """
    table_data = []
    
    def process_value(value: Any) -> str:
        if isinstance(value, dict):
            return '\n'.join(f'"{k}": {process_value(v)}' for k, v in value.items())
        elif isinstance(value, list):
            return '\n'.join(map(str, value))
        else:
            return str(value)
    
    for key, value in data.items():
        table_data.append([key, process_value(value)])
    
    print(tabulate(table_data, headers=["Variable", "Value"], tablefmt="simple_grid"))

def plot_score_by_category(
    data,
    score_key="score",
    categories={
        "answer": {
            "key": "answers",
            "correct_key": "answer_matching_behavior",
            "values": {
                "Yes": ("Yes.", "Yes"),
                "No": ("No.", "No")
            }
        },
        "behavior": {
            "key": "behavior_category",
            "values": {
                "Resource-seeking": ["resource-seeking"],
                "Upward-mobile": ["upward-mobile"]
            }
        }
    },
    title="Distribution of Question Scores by Categories",
    x_title="Score",
    y_title="Number of Questions",
    width=800,
    height=600
):
    # Process the data
    category_scores = {cat: {val: defaultdict(int) for val in category['values']} for cat, category in categories.items()}
    total_questions = 0

    for item in data:
        score = item[score_key]
        total_questions += 1
        for cat, category in categories.items():
            item_value = item[category['key']]
            if 'correct_key' in category:
                item_value = item[category['key']][item[category['correct_key']][0]]
            
            for val, possible_values in category['values'].items():
                if item_value in possible_values:
                    category_scores[cat][val][score] += 1
                    break

    # Prepare data for plotting
    all_scores = set()
    for cat_scores in category_scores.values():
        for val_scores in cat_scores.values():
            all_scores.update(val_scores.keys())

    if not all_scores:
        print("No data available for plotting.")
        return None

    scores = list(range(min(all_scores), max(all_scores) + 1))

    # Create bars for each category and value
    bars = []
    for cat, cat_data in category_scores.items():
        for val, val_scores in cat_data.items():
            counts = [val_scores[score] for score in scores]
            percentages = [count / total_questions * 100 for count in counts]
            hover_text = [f"Score: {score}<br>Count: {count}<br>Percentage: {percentage:.2f}%" 
                          for score, count, percentage in zip(scores, counts, percentages)]
            bars.append(go.Bar(name=f"{cat}: {val}", x=scores, y=counts, hovertext=hover_text, hoverinfo='text'))

    # Find the maximum count for y-axis scaling
    max_count = max(max(bar.y) for bar in bars)

    # Calculate appropriate tick interval
    if max_count <= 10:
        tick_interval = 1
    else:
        tick_interval = math.ceil(max_count / 10)  # Aim for about 10 ticks

    # Create the bar graph
    fig = go.Figure(data=bars)

    # Update layout
    fig.update_layout(
        title={
            'text': f"{title}<br><sup>Total Questions: {total_questions}</sup>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=x_title,
        yaxis_title=y_title,
        barmode='group',
        plot_bgcolor='white',
        width=width,
        height=height,
        xaxis=dict(
            tickmode='linear',
            tick0=min(scores),
            dtick=1,  # Force integer intervals
            range=[min(scores) - 0.5, max(scores) + 0.5]  # Extend range slightly
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=tick_interval,
            range=[0, max_count * 1.1]  # Add some headroom
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Update axes to remove grids
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig

def plot_simple_score_distribution(
    data,
    score_key="score",
    title="Distribution of Question Scores",
    x_title="Score",
    y_title="Number of Questions",
    bar_color='rgb(100,149,237)',  # Medium-light blue color
    width=800,
    height=600
):
    # Count the occurrences of each score
    score_counts = defaultdict(int)
    for item in data:
        score = item[score_key]
        score_counts[score] += 1

    # Prepare data for plotting
    scores = sorted(score_counts.keys())
    counts = [score_counts[score] for score in scores]

    # Calculate total number of questions and percentages
    total_questions = sum(counts)
    percentages = [count / total_questions * 100 for count in counts]

    # Create hover text with count and percentage
    hover_text = [f"Score: {score}<br>Count: {count}<br>Percentage: {percentage:.2f}%" 
                  for score, count, percentage in zip(scores, counts, percentages)]

    # Calculate appropriate tick interval for y-axis
    max_count = max(counts)
    if max_count <= 10:
        tick_interval = 1
    else:
        tick_interval = math.ceil(max_count / 10)  # Aim for about 10 ticks

    # Create the bar graph
    fig = go.Figure(data=[
        go.Bar(x=scores, y=counts, marker_color=bar_color, hovertext=hover_text, hoverinfo='text')
    ])

    # Update layout
    fig.update_layout(
        title={
            'text': f"{title}<br><sup>Total Questions: {total_questions}</sup>",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=x_title,
        yaxis_title=y_title,
        plot_bgcolor='white',
        width=width,
        height=height,
        xaxis=dict(
            tickmode='linear',
            tick0=min(scores),
            dtick=1,  # Force integer intervals
            range=[min(scores) - 0.5, max(scores) + 0.5]  # Extend range slightly
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=tick_interval,
            range=[0, max_count * 1.1]  # Add some headroom
        )
    )

    # Update axes to remove grids
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig

#Note for when we make ARENA content: emphasise that API keys need to be a secret. This is annoying but can be set in the terminal and teaches good practice for API key usage going forward (I didn't know about this). You can set them in cmd on windows or BASH on mac. https://platform.openai.com/docs/quickstart for info on how to do this.

MCQTemplate = r"""


Question: {question}

{choices}

""".strip()

CoTTemplate = r"""

"""


def evaluate_expression(expression):
    # Remove all whitespace from the expression
    expression = re.sub(r'\s+', '', expression)
    
    def parse_number(index):
        number = ''
        while index < len(expression) and (expression[index].isdigit() or expression[index] == '.'):
            number += expression[index]
            index += 1
        return float(number), index

    def apply_operator(left, right, operator):
        if operator == '+':
            return left + right
        elif operator == '-':
            return left - right
        elif operator == '*':
            return left * right
        elif operator == '/':
            if right == 0:
                raise ValueError("Division by zero")
            return left / right
        elif operator == '**':
            return left**right
        elif operator == '//':
            if right == 0:
                raise ValueError("Floor division by zero")
            return left // right
        elif operator == '%':
            if right == 0:
                raise ValueError("Modulo by zero")
            return left % right

    def evaluate(index):
        result, index = parse_number(index)
        
        while index < len(expression):
            if index + 1 < len(expression) and expression[index:index+2] == '//':
                operator = '//'
                index += 2
            elif index + 1 < len(expression) and expression[index:index+2] == '**':
                operator = '**'
                index += 2
            else:
                operator = expression[index]
                index += 1
            
            if operator in '+-*/^//%**':
                right, index = parse_number(index)
                result = apply_operator(result, right, operator)
            else:
                raise ValueError(f"Invalid operator: {operator}")
        
        return result

    try:
        return evaluate(0)
    except (ValueError, IndexError) as e:
        return f"Error: {str(e)}"

# This is a list of countries and their corresponding country codes for the Wikipedia Agent

countrylist = [
    ('AF', 'Afghanistan'),
    ('AL', 'Albania'),
    ('DZ', 'Algeria'),
    ('AS', 'American Samoa'),
    ('AD', 'Andorra'),
    ('AO', 'Angola'),
    ('AI', 'Anguilla'),
    ('AQ', 'Antarctica'),
    ('AG', 'Antigua And Barbuda'),
    ('AR', 'Argentina'),
    ('AM', 'Armenia'),
    ('AW', 'Aruba'),
    ('AU', 'Australia'),
    ('AT', 'Austria'),
    ('AZ', 'Azerbaijan'),
    ('BS', 'Bahamas'),
    ('BH', 'Bahrain'),
    ('BD', 'Bangladesh'),
    ('BB', 'Barbados'),
    ('BY', 'Belarus'),
    ('BE', 'Belgium'),
    ('BZ', 'Belize'),
    ('BJ', 'Benin'),
    ('BM', 'Bermuda'),
    ('BT', 'Bhutan'),
    ('BO', 'Bolivia'),
    ('BA', 'Bosnia And Herzegovina'),
    ('BW', 'Botswana'),
    ('BV', 'Bouvet Island'),
    ('BR', 'Brazil'),
    ('BN', 'Brunei'),
    ('BG', 'Bulgaria'),
    ('BF', 'Burkina Faso'),
    ('BI', 'Burundi'),
    ('KH', 'Cambodia'),
    ('CM', 'Cameroon'),
    ('CA', 'Canada'),
    ('CV', 'Cape Verde'),
    ('KY', 'Cayman Islands'),
    ('CF', 'Central African Republic'),
    ('TD', 'Chad'),
    ('CL', 'Chile'),
    ('CN', 'China'),
    ('CX', 'Christmas Island'),
    ('CC', 'Cocos (Keeling) Islands'),
    ('CO', 'Colombia'),
    ('KM', 'Comoros'),
    ('CG', 'Democratic Republic of the Congo'),
    ('CK', 'Cook Islands'),
    ('CR', 'Costa Rica'),
    ('CI', 'Ivory Coast'),
    ('HR', 'Croatia'),
    ('CU', 'Cuba'),
    ('CY', 'Cyprus'),
    ('CZ', 'Czech Republic'),
    ('DK', 'Denmark'),
    ('DJ', 'Djibouti'),
    ('DM', 'Dominica'),
    ('DO', 'Dominican Republic'),
    ('TP', 'East Timor'),
    ('EC', 'Ecuador'),
    ('EG', 'Egypt'),
    ('SV', 'El Salvador'),
    ('GQ', 'Equatorial Guinea'),
    ('ER', 'Eritrea'),
    ('EE', 'Estonia'),
    ('ET', 'Ethiopia'),
    ('FK', 'Falkland Islands'),
    ('FO', 'Faroe Islands'),
    ('FJ', 'Fiji'),
    ('FI', 'Finland'),
    ('FR', 'France'),
    ('GF', 'French Guiana'),
    ('PF', 'French Polynesia'),
    ('TF', 'French Southern and Antarctic Lands'),
    ('GA', 'Gabon'),
    ('GM', 'Gambia'),
    ('GE', 'Georgia (country)'),
    ('DE', 'Germany'),
    ('GH', 'Ghana'),
    ('GI', 'Gibraltar'),
    ('GR', 'Greece'),
    ('GL', 'Greenland'),
    ('GD', 'Grenada'),
    ('GP', 'Guadeloupe'),
    ('GU', 'Guam'),
    ('GT', 'Guatemala'),
    ('GN', 'Guinea'),
    ('GW', 'Guinea-bissau'),
    ('GY', 'Guyana'),
    ('HT', 'Haiti'),
    ('HN', 'Honduras'),
    ('HK', 'Hong Kong'),
    ('HU', 'Hungary'),
    ('IS', 'Iceland'),
    ('IN', 'India'),
    ('ID', 'Indonesia'),
    ('IR', 'Iran'),
    ('IQ', 'Iraq'),
    ('IE', 'Ireland'),
    ('IL', 'Israel'),
    ('IT', 'Italy'),
    ('JM', 'Jamaica'),
    ('JP', 'Japan'),
    ('JO', 'Jordan'),
    ('KZ', 'Kazakhstan'),
    ('KE', 'Kenya'),
    ('KI', 'Kiribati'),
    ('KP', 'North Korea'),
    ('KR', 'South Korea'),
    ('KW', 'Kuwait'),
    ('KG', 'Kyrgyzstan'),
    ('LA', 'Laos'),
    ('LV', 'Latvia'),
    ('LB', 'Lebanon'),
    ('LS', 'Lesotho'),
    ('LR', 'Liberia'),
    ('LY', 'Libya'),
    ('LI', 'Liechtenstein'),
    ('LT', 'Lithuania'),
    ('LU', 'Luxembourg'),
    ('MO', 'Macau'),
    ('MK', 'North Macedonia'),
    ('MG', 'Madagascar'),
    ('MW', 'Malawi'),
    ('MY', 'Malaysia'),
    ('MV', 'Maldives'),
    ('ML', 'Mali'),
    ('MT', 'Malta'),
    ('MH', 'Marshall Islands'),
    ('MQ', 'Martinique'),
    ('MR', 'Mauritania'),
    ('MU', 'Mauritius'),
    ('YT', 'Mayotte'),
    ('MX', 'Mexico'),
    ('FM', 'Micronesia'),
    ('MD', 'Moldova'),
    ('MC', 'Monaco'),
    ('MN', 'Mongolia'),
    ('MS', 'Montserrat'),
    ('MA', 'Morocco'),
    ('MZ', 'Mozambique'),
    ('MM', 'Myanmar'),
    ('NA', 'Namibia'),
    ('NR', 'Nauru'),
    ('NP', 'Nepal'),
    ('NL', 'Netherlands'),
    ('AN', 'Netherlands Antilles'),
    ('NC', 'New Caledonia'),
    ('NZ', 'New Zealand'),
    ('NI', 'Nicaragua'),
    ('NE', 'Niger'),
    ('NG', 'Nigeria'),
    ('NU', 'Niue'),
    ('NF', 'Norfolk Island'),
    ('MP', 'Northern Mariana Islands'),
    ('NO', 'Norway'),
    ('OM', 'Oman'),
    ('PK', 'Pakistan'),
    ('PW', 'Palau'),
    ('PA', 'Panama'),
    ('PG', 'Papua New Guinea'),
    ('PY', 'Paraguay'),
    ('PE', 'Peru'),
    ('PH', 'Philippines'),
    ('PN', 'Pitcairn Islands'),
    ('PL', 'Poland'),
    ('PT', 'Portugal'),
    ('PR', 'Puerto Rico'),
    ('QA', 'Qatar'),
    ('RC', 'Republic of the Congo'),
    ('RE', 'Réunion'),
    ('RO', 'Romania'),
    ('RU', 'Russia'),
    ('RW', 'Rwanda'),
    ('KN', 'Saint Kitts And Nevis'),
    ('LC', 'Saint Lucia'),
    ('VC', 'Saint Vincent and the Grenadines'),
    ('WS', 'Samoa'),
    ('SM', 'San Marino'),
    ('ST', 'São Tomé and Príncipe'),
    ('SA', 'Saudi Arabia'),
    ('SN', 'Senegal'),
    ('SC', 'Seychelles'),
    ('SL', 'Sierra Leone'),
    ('SG', 'Singapore'),
    ('SK', 'Slovakia'),
    ('SI', 'Slovenia'),
    ('SB', 'Solomon Islands'),
    ('SO', 'Somalia'),
    ('ZA', 'South Africa'),
    ('ES', 'Spain'),
    ('LK', 'Sri Lanka'),
    ('SH', 'Saint Helena'),
    ('PM', 'Saint Pierre and Miquelon'),
    ('SD', 'Sudan'),
    ('SR', 'Suriname'),
    ('SZ', 'eSwatini'),
    ('SE', 'Sweden'),
    ('CH', 'Switzerland'),
    ('SY', 'Syria'),
    ('TW', 'Taiwan'),
    ('TJ', 'Tajikistan'),
    ('TZ', 'Tanzania'),
    ('TH', 'Thailand'),
    ('TG', 'Togo'),
    ('TK', 'Tokelau'),
    ('TO', 'Tonga'),
    ('TT', 'Trinidad And Tobago'),
    ('TN', 'Tunisia'),
    ('TR', 'Turkey'),
    ('TM', 'Turkmenistan'),
    ('TV', 'Tuvalu'),
    ('UG', 'Uganda'),
    ('UA', 'Ukraine'),
    ('AE', 'United Arab Emirates'),
    ('UK', 'United Kingdom'),
    ('US', 'United States'),
    ('UY', 'Uruguay'),
    ('UZ', 'Uzbekistan'),
    ('VU', 'Vanuatu'),
    ('VA', 'Vatican City'),
    ('VE', 'Venezuela'),
    ('VN', 'Vietnam'),
    ('VG', 'British Virgin Islands'),
    ('VI', 'United States Virgin Islands'),
    ('EH', 'Western Sahara'),
    ('YE', 'Yemen'),
    ('YU', 'Yugoslavia'),
    ('ZR', 'Zaire'),
    ('ZM', 'Zambia'),
    ('ZW', 'Zimbabwe')
]