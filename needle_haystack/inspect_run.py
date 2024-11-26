from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import (               
  system_message, generate
)                                             

dataset = Sample(
    input=input
)

@task
def needle_haystack():
    return Task(
        dataset=dataset,
        plan=[system_message("XXX"),
              generate()],
        scorer=model_graded_fact
    )


