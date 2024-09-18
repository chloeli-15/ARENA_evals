# %%

import os
from pathlib import Path
import sys
import openai
from openai import OpenAI
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional 
import random
import warnings
import time

# Make sure exercises are in the path
chapter = r"chapter3_llm_evals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part1_intro").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)

from utils import import_json, save_json, retry_with_exponential_backoff, pretty_print_questions,load_jsonl
# %%
