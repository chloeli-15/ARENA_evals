#%%
import torch as t
import os, sys
from pathlib import Path
from functools import partial
import operator
import json
from typing import List, Dict, Any, Optional, Literal, Callable

# Make sure exercises are in the path
chapter = r"chapter3_llm_evals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part1_intro").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)

MAIN = __name__ == '__main__'

from utils import import_json, save_json, retry_with_exponential_backoff, apply_system_format, apply_assistant_format, apply_user_format, apply_message_format, pretty_print_questions, tabulate_model_scores, plot_score_by_category, plot_simple_score_distribution, print_dict_as_table

#%%
def test_generate_formatted_response(test_fn: Callable, client):
    from part2_dataset_generation.solutions import Config, QuestionGeneration, Question
    config = Config(model="gpt-4o-mini", temperature=1)

    user_msg = "Generate a multiple choice question about ethics in business."
    
    result = test_fn(client, config, user=user_msg)

    try:
        result_dict = json.loads(result)
    except json.JSONDecodeError:
        raise AssertionError("Result is not a dictionary and cannot be converted to one")

    # Check for reasoning
    assert 'reasoning' in result_dict, "Reasoning is missing from the response"
    assert isinstance(result_dict['reasoning'], str), "Reasoning should be a string"
    
    # Check for questions
    assert 'questions' in result_dict, "Questions are missing from the response"
    assert isinstance(result_dict['questions'], list), "Questions should be a list"
    assert len(result_dict['questions']) > 0, "At least one question should be generated"
    
    # Check the structure of the first question
    first_question = result_dict['questions'][0]
    assert isinstance(first_question, dict), "Each question should be a dictionary"
    
    required_fields = ['system', 'question', 'answers', 'answer_matching_behavior', 'answer_not_matching_behavior']
    for field in required_fields:
        assert field in first_question, f"'{field}' is missing from the question structure"
    
    # Check answers structure
    assert isinstance(first_question['answers'], dict), "Answers should be a dictionary"
    assert 'A' in first_question['answers'] and 'B' in first_question['answers'], "Answers should have options A and B"
    
    # Check behavior fields
    assert isinstance(first_question['answer_matching_behavior'], list), "answer_matching_behavior should be a list"
    assert isinstance(first_question['answer_not_matching_behavior'], list), "answer_not_matching_behavior should be a list"
    assert all(option in ['A', 'B'] for option in first_question['answer_matching_behavior'] + first_question['answer_not_matching_behavior']), "Behavior options should only be 'A' or 'B'"
    
    print("generate_formatted_response test passed!")



def test_rubric(client, dataset, config, eval_prompts):
    import part2_dataset_generation.solutions as solutions
    scored_dataset = solutions.query_evaluator(client=client, dataset=dataset, config=config, prompts=eval_prompts)
    ordered_dataset = sorted(scored_dataset, key=operator.itemgetter("score"), reverse=True)
    return ordered_dataset


if MAIN:
    POWER_SEEKING_TEST_QUESTIONS=import_json("./part2_dataset_generation/power_seeking_data/rubric_test_questions.json")
# %%
