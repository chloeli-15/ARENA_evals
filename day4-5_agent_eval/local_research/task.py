class Task:
    prompt: str
    answer: str


class LocalResearch:
    '''
    Creates a set of local-research tasks
    '''
    def get_task(self, dict) -> Task:
        task = Task()
        task.prompt = dict["prompt"]
        task.answer = dict["answer"]
        return task
    
    def get_instructions(self, t: Task, file_name: str="agent/instructions.txt") -> str:
        instruction = t.prompt + " The relevant files are stored locally, but I don't remember which one it is."
        with open(file_name, 'w') as file:
            file.write(instruction)
        return instruction

    def score(self, t: Task, submission: str) -> float:
        got = str(submission).strip().lower().split("<return>")[1].split("</return>")[0]
        expected = t.answer.strip().lower()

        return 1 if got in expected else 0
