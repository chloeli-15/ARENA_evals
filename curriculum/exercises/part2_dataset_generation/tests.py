import torch as t
import os, sys
from pathlib import Path
from functools import partial
import operator
exercises_dir = Path(__file__).parent.parent
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))


def test_rubric(dataset, config, eval_prompts):
    import part2_dataset_generation.solutions as solutions
    scored_dataset = solutions.query_evaluator(dataset, config, eval_prompts)
    ordered_dataset = sorted(scored_dataset, key=operator.itemgetter("score"), reverse=True)
    return ordered_dataset