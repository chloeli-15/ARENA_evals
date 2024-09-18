import re

import os
import pathlib
import json
import dataclasses

from workspace.common_types import ParsedQuestion
from workspace import display_utils

MCQ_EXAMPLES_FILENAME = pathlib.Path("mcq_examples.json")


def convert_to_mcq_format(
    input_text: str,
    matching_choices: list[str],
    behavior_category: str,
    tags: list[str] | None = None,
) -> ParsedQuestion:
    """
    Converts an evaluation question into a JSON format.

    :param input_text: The evaluation question text.
    :param matching_choices: A list of option letters that match the desired behavior.
    :param behavior_category: A string representing the behavior category.
    :return: A JSON-formatted string representing the structured evaluation question.
    """

    # note: this does not HAVE to be a system prompt, we just make it one in case
    #       examples use it later
    system_prompt = (
        "Please include your answer inside <answer> </answer> tags. "
        "Anything outside of that will be considered the steps you "
        "used to decide on the answer."
    )

    # Split the input text into lines
    lines = input_text.strip().split("\n")

    # Initialize variables
    question_lines = []
    answer_choices = {}
    tags = tags or []
    options_pattern = re.compile(r"^(?:[a-z]\)|[a-z]\.|[A-Z]\)|[A-Z]\.)\s*(.*)")

    # Flags to track when options start
    options_started = False

    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        # Check if the line is an option
        match = options_pattern.match(line)
        if match:
            options_started = True
            # Extract option letter and text
            option_letter = line[0]
            option_text = match.group(1).strip()
            answer_choices[option_letter] = option_text
        elif options_started:
            # Continuation of an option's text
            answer_choices[option_letter] += " " + line.strip()
        else:
            # Before options start
            question_lines.append(line)

    question_text = "\n".join(question_lines).strip()

    # Determine answers matching and not matching the behavior
    matching_behavior = matching_choices
    not_matching_behavior = [opt for opt in answer_choices.keys() if opt not in matching_behavior]

    # Construct the JSON object
    parsed_question = ParsedQuestion(
        system=system_prompt,
        question=question_text,
        answers=answer_choices,
        answer_matching_behavior=matching_behavior,
        answer_not_matching_behavior=not_matching_behavior,
        behavior_category=behavior_category,
        tags=tags,
    )

    return parsed_question


def get_mcq_absolute_filepath() -> pathlib.Path:

    current_dir = pathlib.Path(__file__).parent.resolve()

    filepath = current_dir / MCQ_EXAMPLES_FILENAME

    return filepath


def write_mcq_examples(parsed_questions: list[ParsedQuestion]) -> None:

    filepath = get_mcq_absolute_filepath()

    print(f"Writing {len(parsed_questions)} MCQ examples to {filepath}...")

    with open(filepath, "w") as file:
        json.dump([dataclasses.asdict(x) for x in parsed_questions], file, indent=2)

    print(f"Wrote {len(parsed_questions)} MCQ examples to {filepath}")


def load_mcq_examples() -> list[ParsedQuestion]:

    filepath = get_mcq_absolute_filepath()

    print(f"Reading MCQ examplesfrom {filepath}...")
    with open(filepath, "r") as f:

        parsed_questions: list[ParsedQuestion] = []

        # parse one by one so we can give a clearer error on failure
        for row in json.load(f):

            try:
                parsed_question = ParsedQuestion(**row)
                parsed_questions.append(parsed_question)

            except TypeError as e:
                print(f"Failed to parse {json.dumps(row, indent=2)}")
                raise e

    print(f"Loaded {len(parsed_questions)} MCQ examples")

    return parsed_questions


def display_question(question: ParsedQuestion):
    display_utils.display_markdown_and_code(question.to_prompt())
