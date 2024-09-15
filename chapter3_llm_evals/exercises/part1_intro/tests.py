#%%
import torch as t
import openai
import os, sys
from pathlib import Path
from functools import partial
from typing import Callable, List, Dict, Any, Literal, Optional
from unittest.mock import patch, MagicMock
import pytest
import time
import random
from unittest.mock import Mock


chapter = r"chapter3_llm_evals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part1_intro").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)


# def test_generate_response(generate_response_func: Callable):
#     # Create a mock client
#     mock_client = MagicMock()
#     mock_response = MagicMock()
#     mock_response.choices[0].message.content = "This is a test response."
#     mock_client.chat.completions.create.return_value = mock_response

#     # Patch the OpenAI class
#     with patch('part1_intro.solutions.OpenAI', return_value=mock_client):
#         # Test case 1: Basic functionality with messages
#         messages = [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": "Hello, how are you?"}
#         ]
#         result = generate_response_func(model="gpt-4o-mini", messages=messages)
#         assert result == "This is a test response."

#         # Test case 2: Using user and system parameters
#         result = generate_response_func(model="gpt-4o-mini", user="Hello", system="You are a helpful assistant.")
#         assert result == "This is a test response."

#         # Test case 3: Verbose output (capture print statements)
#         with patch('builtins.print') as mock_print:
#             generate_response_func(model="gpt-3.5-turbo", messages=messages, verbose=True)
#             mock_print.assert_any_call("SYSTEM:\nYou are a helpful assistant.\n")
#             mock_print.assert_any_call("USER:\nHello, how are you?\n")

#         # Check if the mock client was called correctly
#         mock_client.chat.completions.create.assert_called_with(
#             model="gpt-3.5-turbo",
#             messages=messages,
#             temperature=1  # Assuming default temperature is 1
#         )

# from part1_intro.solutions import generate_response
# test_generate_response(generate_response)

# %%
def test_retry_with_exponential_backoff(student_retry):
    from part1_intro.solutions import retry_with_exponential_backoff as correct_retry

    # Test cases

    test_cases = [
        ({"max_retries": 5, "initial_sleep_time": 1, "backoff_factor": 2, "jitter": False}, 
         [Exception("rate_limit_exceeded")] * 5 + [None]),
        ({"max_retries": 3, "initial_sleep_time": 0.5, "backoff_factor": 1.5, "jitter": True}, 
         [Exception("rate_limit_exceeded")] * 2 + [None]),
        ({"max_retries": 2, "initial_sleep_time": 1, "backoff_factor": 2, "jitter": False}, 
         [Exception("other_error"), None]),
    ]

    for params, side_effects in test_cases:
        # Test function to be decorated
        test_func = Mock(side_effect=side_effects)
        # Create decorated functions
        correct_decorated = correct_retry(test_func, **params)
        student_decorated = student_retry(test_func, **params)

        # Patch time.sleep and random.random
        with patch('time.sleep') as mock_sleep, patch('random.random', return_value=0.5):
            # Run correct implementation
            try:
                correct_result = correct_decorated()
            except Exception as e:
                correct_result = str(e)
            correct_sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            correct_call_count = test_func.call_count

            # Reset mocks
            test_func.reset_mock()
            mock_sleep.reset_mock()

            # Run student implementation
            try:
                student_result = student_decorated()
            except Exception as e:
                student_result = str(e)
            student_sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            student_call_count = test_func.call_count

        # Assertions
        assert correct_result == student_result, f"Results differ for params {params}"
        assert correct_call_count == student_call_count, f"Number of calls differ for params {params}"
        assert correct_sleep_calls == student_sleep_calls, f"Sleep times differ for params {params}"

    print("All tests passed successfully!")

from part1_intro.solutions import retry_with_exponential_backoff
test_retry_with_exponential_backoff(retry_with_exponential_backoff)
# %%
