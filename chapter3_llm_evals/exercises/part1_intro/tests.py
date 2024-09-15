import torch as t
import openai
import os, sys
from pathlib import Path
from functools import partial
from typing import Callable, List, Dict, Any, Literal, Optional
from unittest.mock import patch, MagicMock
import pytest

exercises_dir = Path(__file__).parent.parent
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))



def test_generate_response(generate_response_func: Callable):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is a test response."

    with patch('openai.ChatCompletion.create', return_value=mock_response):
        # Test case 1: Basic functionality with messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        result = generate_response_func(model="gpt-4o-mini", messages=messages)
        assert result == "This is a test response."

        # Test case 2: Using user and system parameters
        result = generate_response_func(model="gpt-4o-mini", user="Hello", system="You are a helpful assistant.")
        assert result == "This is a test response."

        # Test case 3: Verbose output (capture print statements)
        with patch('builtins.print') as mock_print:
            generate_response_func(model="gpt-3.5-turbo", messages=messages, verbose=True)
            mock_print.assert_any_call("SYSTEM:\nYou are a helpful assistant.\n")
            mock_print.assert_any_call("USER:\nHello, how are you?\n")
