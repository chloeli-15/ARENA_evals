import wandb
from inspect_ai import Task, eval, task
from inspect_ai.log import EvalConfig, EvalLog
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice, system_message

SYSTEM_MESSAGE = """
Choose the most plausible continuation for the story.
"""


def record_to_sample(record):
    return Sample(
        input=record["ctx"],
        target=chr(ord("A") + int(record["label"])),
        choices=record["endings"],
        metadata=dict(source_id=record["source_id"]),
    )


@task
def hellaswag():
    # dataset
    dataset = hf_dataset(
        path="hellaswag", split="validation", sample_fields=record_to_sample, trust=True
    )

    # define task
    return Task(
        dataset=dataset[:10],
        plan=[system_message(SYSTEM_MESSAGE), multiple_choice()],
        scorer=choice(),
    )


import git


def get_commit():
    repo = git.Repo(search_parent_directories=True)
    commit_hash = repo.head.commit.hexsha[:7]
    if repo.is_dirty():
        commit_hash += "-dirty"
    return commit_hash


if __name__ == "__main__":
    model = "openai/gpt-4o"
    enable_wandb = True
    t = hellaswag
    if enable_wandb:
        wandb_run = wandb.init(
            project="arena_evals",
            tags=[model, t.__dict__["__registry_info__"].name, get_commit()],
        )
        assert wandb_run is not None
    else:
        wandb_run = None
    logs: list[EvalLog] = eval(
        [t],
        model=model,
    )

    l = logs[0]
    if wandb_run:
        wandb_run.log({"accuracy": l.results.scores[0].metrics["accuracy"].value})
        wandb_run.save(f"logs/*{l.eval.task_id}.json")
