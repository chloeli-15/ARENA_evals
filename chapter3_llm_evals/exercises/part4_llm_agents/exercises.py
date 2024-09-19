# %%
import json
import os

import wikipedia
from wikipedia import WikipediaPage
from wikipedia import DisambiguationError, PageError
from openai import OpenAI
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from anthropic import Anthropic
from typing import Literal, Optional, Dict, List, Any
from abc import abstractmethod
import math
import re
import sys
from pathlib import Path


#Make sure exercises are in the path
chapter = r"chapter3_llm_evals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part4_llm_agent_evals").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)

from utils import import_json, save_json, retry_with_exponential_backoff, pretty_print_questions, load_jsonl, omit
from utils import countrylist
from utils import evaluate_expression, apply_user_format, apply_assistant_format, establish_client_anthropic, establish_client_OpenAI, retry_with_exponential_backoff
import part4_llm_agents.tests as tests

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

client = OpenAI()

# %%
