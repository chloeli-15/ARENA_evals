import asyncio
import time
import numpy as np
import json
import os
import glob
from openai import OpenAI

class NeedleHaystackTester:
    def __init__(self, 
                 model_name: str,
                 evaluator: str,
                 haystack_dir: str,
                 needle: str,
                 retrieval_question: str,):
        self.model_name = model_name,
        self.haystack_dir = haystack_dir,
        self.retrieval_question = retrieval_question,
        self.needle = needle,
        self.context_lengths = 1024,
        self.evaluator_model = evaluator


    async def run_test(self):
        # Test model
        pass

    
    async def evaluate_and_log(self):
        #####################
        #  Evaluate model and log results
        #####################

        # Generate context
        context = await self.generate_context()
        
        # Generate model response
        client = OpenAI()
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"},
                {"role": "user", "content": f"Here is a document: {context}. \n {self.retrieval_question} Don't give information outside the document or repeat your findings."}
            ],
            temperature=0.0,
        )
        response = completion.choices[0].message.content

        # Evaluate model response
        completion = client.chat.completions.create(
            model=self.evaluator_model,
            messages=[
                {"role": "system", "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"},
                {"role": "user", "content": f"Here is a document: {context}. \n {self.retrieval_question} Don't give information outside the document or repeat your findings."}
            ],
            temperature=0.0,
        )
        score = completion.choices[0].message.content

    #################
    # Helper methods
    #################

    async def generate_context(self):
        # Generate context for model
        context = self.read_context_files()
        context = self.encode_and_trim()
        context = self.insert_needle()

        return context

    def get_context_length_in_tokens(self, text):
        # Get length of context
        pass

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)
        base_dir = os.path.abspath(os.path.dirname(__file__))
        # Read context file
        while len(context) < max_context_length:
            for file in glob.glob(os.path.join(base_dir, self.haystack_dir, "*.txt")):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def encode_and_trim(self):
        # Encode and trim context to a custom length
        pass

    def insert_needle(self):
        # Insert needle into context
        pass

