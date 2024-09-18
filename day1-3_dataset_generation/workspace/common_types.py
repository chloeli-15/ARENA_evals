import json
import re
import dataclasses
from typing import TypeVar

PromptStringWithoutChoices = str
CompletePromptString = str
RawResponseString = str
ParsedResponseString = str

ClientType = TypeVar("ClientType")

# this is just the SHA for now
ParsedQuestionId = str

ModelName = str


# TODO(bschoen): Answer label alias
@dataclasses.dataclass(frozen=True)
class ParsedQuestion:
    system: str
    question: str
    answers: dict[str, str]
    answer_matching_behavior: list[str]
    answer_not_matching_behavior: list[str]
    answer_scores: dict[str, float]
    behavior_category: str
    tags: list[str]

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> dict[str, str | list[str] | dict[str, str]]:

        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, dict_repr: dict[str, str | list[str] | dict[str, str]]) -> "ParsedQuestion":
        return cls(**dict_repr)

    # TODO(bschoen): Would normally use `field(init=False)` + `__post_init__` to add this in
    def get_sha(self) -> ParsedQuestionId:
        """
        Calculate and return the MD5 hash of the JSON representation of this ParsedQuestion.

        Returns:
            str: The MD5 hash of the JSON string representation of this object.

        Note:
            We use MD5 for its speed and simplicity, not for cryptographic security.
            The hash provides a unique identifier based on the object's content.
        """
        import hashlib

        # Convert the object to a JSON string
        json_string = json.dumps(dataclasses.asdict(self))

        # Calculate the MD5 hash
        return hashlib.md5(json_string.encode("utf-8")).hexdigest()

    def to_prompt_without_choices(self) -> PromptStringWithoutChoices:
        """
        Primarily used for compatibility with frameworks which auto-generate choice letters

        I imagine this also makes tools like shuffling easier to implement generically

        """
        return self.system + f"\n\n{self.question}"

    # TODO(bschoen): Add example to docstring
    def to_prompt(self) -> CompletePromptString:
        """
        Convert the ParsedQuestion to a full user content string passed to the model.

        Returns:
            str: The full user content string.

        """
        # Construct the question text with answer choices
        choices_text = "\n".join(
            [f"* ({option}) {answer}" for option, answer in self.answers.items()]
        )

        # note: o1 doesn't support system prompts yet, but in our experience this doesn't matter
        full_user_content = self.to_prompt_without_choices() + f"\n\n{choices_text}"

        return full_user_content


@dataclasses.dataclass(frozen=True)
class ModelQuestionRawResponse:
    model: ModelName
    question: ParsedQuestion
    raw_response: RawResponseString


@dataclasses.dataclass(frozen=True)
class ModelQuestionParsedResponse:
    raw_response: ModelQuestionRawResponse
    parsed_response: ParsedResponseString | None


from typing import Callable, Optional, List, Dict, Any
from openai import OpenAI

# Type alias for the generate_response function
# TODO(bschoen): Would want to use NewTypes for this
GenerateResponseFn = Callable[
    [
        OpenAI,  # client
        str,  # model
        int,  # max_tokens
        Optional[List[Dict[str, Any]]],  # messages
        Optional[str],  # user
        Optional[str],  # system
        float,  # temperature
        bool,  # verbose
    ],
    Optional[str],
]
