import asyncio
import time
import numpy as np
import json
import os
import glob
from openai import OpenAI
from typing import List

print(os.environ.get('OPENAI_API_KEY'))

CRITERIA = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numberical score
            
                """,}

class NeedleHaystackTester:
    def __init__(self, 
                 model_name: str,
                 evaluator: str,
                 haystack_dir: str,
                 needle: str,
                 retrieval_question: str,
                 criteria: str,
                 correct_answer: str,
                 context_lengths: List[int] = [1024]):
        self.model_name = model_name,
        self.haystack_dir = haystack_dir,
        self.retrieval_question = retrieval_question,
        self.criteria = criteria,
        self.needle = needle,
        self.context_lengths = context_lengths,
        self.evaluator_model = evaluator,
        self.correct_answer = correct_answer

        self.results = {}

    async def run_test(self):
        # Test model
        for context_length in self.context_lengths:
            await self.evaluate(context_length)

    
    async def evaluate(self, context_length):
        #####################
        #  Evaluate model and log results
        #####################
        # Prepare context
        context = await self.prepare_context(context_length)

        # Generate model response
        client = OpenAI()
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"},
                {"role": "user", "content": f"Here is a document: {context}. \n {self.retrieval_question} Don't give information outside the document or repeat your findings."}
            ],
            temperature=1.0,
        )
        response = completion.choices[0].message.content

        # Evaluate model response
        completion = client.chat.completions.create(
            model=self.evaluator_model,
            messages=[
                {"role": "system", "content": f"You are helpful AI that compares the similarity between reference text and example text. You grade their similarity using this criteria: \n{self.criteria}\n You only respond with a numerical score between 1 and 10."},
                {"role": "user", "content": f"Reference text: {self.correct_answer} \nExample text: {response}"}
            ],
            temperature=1.0,
        )
        score = completion.choices[0].message.content

        self.results[context_length] = {
            "model_response": response,
            "score": score
        }
    
    #################
    # Helper methods
    #################

    async def prepare_context(self, context_length: int):
        # Generate context for model
        context = self.read_context_files()
        context = self.encode_and_trim(context_length)
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

    def encode_and_trim(self, context_length):
        # Encode and trim context to a custom length
        pass

    def insert_needle(self):
        # Insert needle into context
        pass


