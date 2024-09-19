# Notes

> [!NOTE] This will be good for understanding how other frameworks implement agent use, and adapting them to have o1-mini tool use should show how adaptable they are

The partially defined prompts generalization seems to trap a lot of frameworks

Can take advantage of all defining at once. If want luxury of generalizing just use dataclasses all the way through.

If you needed to add cleanup this would be a nightmare.

This is basically entirely integrating the tool as the agent, which is fine for a researcher.

* Everything passing dataclasses back and forth?

## Agent Design

Agents are implemented as solvers: https://inspect.ai-safety-institute.org.uk/agents.html#sec-langchain

The full example is included here: https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/examples/langchain/inspect_langchain.py

```python
# Interface for LangChain agent function
class LangChainAgent(Protocol):
    async def __call__(self, llm: BaseChatModel, input: dict[str, Any]): ...

# Convert a LangChain agent function into a Solver
def langchain_solver(agent: LangChainAgent) -> Solver:

    async def solve(state: TaskState, generate: Generate) -> TaskState:

        # create the inspect model api bridge
        llm = InspectChatModel()

        # call the agent
        await agent(
            llm = llm,
            input = dict(
                input=state.user_prompt.text,
                chat_history=as_langchain_chat_history(
                    state.messages[1:]
                ),
            )
        )

        # collect output from llm interface
        state.messages = llm.messages
        state.output = llm.output
        state.output.completion = output
        
        # return state
        return state

    return solve

# LangChain BaseChatModel for Inspect Model API
class InspectChatModel(BaseChatModel):
     async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: dict[str, Any],
    ) -> ChatResult:
        ...
```