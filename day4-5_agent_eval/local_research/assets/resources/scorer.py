
def score(submission: str) -> float:
    got = str(submission).strip().lower().split("<return>")[1].split("</return>")[0]
    answer = "b"
    expected = answer.strip().lower()

    return 1 if got in expected else 0

if __name__ == "__main__":
    with open("submission.txt", "r") as f:
        submission = f.read()
    
    score(submission)