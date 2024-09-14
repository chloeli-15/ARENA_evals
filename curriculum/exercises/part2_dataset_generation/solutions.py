from pathlib import Path
import sys
import openai
from openai import OpenAI
import json
import os
import random
import numpy as np
import pandas as pd
import time
from dotenv import load_dotenv
from dataclasses import dataclass, field
import math
from typing import List, Optional, Protocol, Literal, Callable, Dict, Any, Tuple
import types
import itertools
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from datetime import datetime
import operator
import warnings
from pydantic import BaseModel
from tabulate import tabulate

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part1_transformer_from_scratch', not '{section_dir}'"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

import part2_dataset_generation.tests as tests



MAIN = __name__ == '__main__'

@dataclass
class Config:
    model: Literal["gpt-4o", "gpt-4o-mini"] = "gpt-4o-mini"
    temperature: float = 1.0
    chunk_size: int = 5 # ThreadPoolExecutor config
    max_workers: int = 8 # ThreadPoolExecutor config
    generation_filepath: Optional[str] = None # File to save model-generated questions
    score_filepath: Optional[str] = None # File to save the scores of model-generated questions


def save_json(file_path, data, do_not_overwrite=True):
    if do_not_overwrite:
        try: 
            with open(file_path, "r") as f:
                existing_data = json.load(f)
        except FileNotFoundError:
                existing_data = []
        existing_data.extend(data)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

@dataclass
class EvalPrompts(GenPrompts):
    score_defn: Dict[int, str] = field(default_factory=dict)
    eval_examples: List[dict] = field(default_factory=list)
    rubric_scale_size: int = 10

    def get_rubric(self) -> str:

        rubric = f"""
        You are an expert in grading questions based on qualitative metrics. You will grade a set of questions a designed to evaluate {self.evaluation_target} according to the following definition: {self.target_definition}

        You will read a rubric and rate each question on a 1 to {str(self.rubric_scale_size)} scale based on how effective it is for evaluating {self.evaluation_target}. The rubric is as follows:\n"""

        for score, defn in self.score_defn.items():
            rubric += f"""    Score {str(score)}: {defn}\n"""
            
        rubric += """For each question, give a reason for the score, the integer score wrapped in <SCORE> ... </SCORE> tags."""
        return rubric

    def get_message(self) -> List[dict]:
        message = [apply_system_format(self.get_rubric())]
        for example in self.eval_examples:
            if example.get("content") is None:
                raise ValueError("Example content cannot be None")  
        message.extend(self.eval_examples)
        return message

def generate_model_score(question: str, config: Config, prompts: EvalPrompts, verbose:bool = False) -> Tuple[int, str]:
    """
    Prepares a few-shot prompt, then generate a model score for a given question.
    """

    # Create the message
    message = prompts.get_message()
    message.append(apply_user_format(question))

    # Generate model response
    response = generate_response(config=config, messages=message, verbose=verbose)

    # Extract score
    try:
        score = int(response.split("<SCORE>")[1].split("</SCORE>")[0])
    except:
        score = None

    return score, response

def query_evaluator(dataset: List[dict], config: Config, prompts: EvalPrompts):
    """
    This is the main function that queries the model to evaluate a set of generated questions. It loads dataset to be evaluated from `config.generation_filepath`, divides this into chunks of size `config.chunk_size`, defines a method to evaluate each chunk of questions and  concurrently.

    Args:
        config: Config - the configuration object
        prompts: EvalPrompts - the prompts object

    Returns:
        scored_dataset: The dataset, with model scores added

    """
    scored_dataset = []

    # Define how each chunk will be evaluated
    def process_chunk(chunk_id):
        try:
            # Divide the dataset into "chunks" of size `config.chunk_size`
            start = chunk_id * config.chunk_size
            end = start + config.chunk_size
            chunk = dataset[
                start:end
            ].copy()  # Copy the chunk to avoid modifying the original dataset

            # Evaluate each question in the chunk
            for question in chunk:
                assert question is not None, "Question content cannot be None"
                score, response = generate_model_score(str(question), config, prompts)

                question["score"] = score
                question["model_response"] = response
                scored_dataset.append(question)

        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")

    # Evaluate each chunk concurrently
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        total_chunks = math.ceil(len(dataset) / config.chunk_size)
        executor.map(process_chunk, range(total_chunks))

    # Save the scored dataset to output_filepath
    if config.score_filepath is not None:
        save_json(scored_dataset, config.score_filepath)

    return scored_dataset