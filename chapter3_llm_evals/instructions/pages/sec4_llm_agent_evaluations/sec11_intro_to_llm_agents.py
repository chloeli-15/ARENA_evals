import streamlit as st


def section():
    st.sidebar.markdown(
        r"""
## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#introduction'>Introduction</a></li>
    <li class='margtop'><a class='contents-el' href='#content-learning-objectives'>Content & Learning Objectives</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#1-Intro to Inspect'>Intro to Inspect</a></li>
        <li><a class='contents-el' href='#2-mcq-benchmarks-exploration'>MCQ Benchmarks: Exploration</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
</ul>""",
        unsafe_allow_html=True,

    )




    st.markdown(
        r"""
# 1️⃣ Intro to LLM Agents

## What is an LLM agent?
<!---
Points to make:
- "LLM agent" - a "scaffolding" program (i.e. Python program) that interacts with an LLM API. Include a version of METR "Evaluating Language-Model Agents on Realistic Autonomous Tasks" Figure 2
    - Define scaffolding
- More schematic breakdown of possible scaffolding: "tools" (describe what this means, what "tool calling" is), "memory" (Probably better move to "Build Agent" section! I've expanded this section there)
- Mention list of examples of prominent LLM agents:
    - [Minecraft LM Agent](https://arxiv.org/abs/2305.16291)
    - [AutoGPT](https://autogpt.net/) and [LangChain](https://www.langchain.com/)

==========================
--->
An LLM "agent" consists of **a scaffolding program interacting with an LLM API**. Initially, the scaffolding program sends instructions to the LLM on the task goal, the actions available to the LLM, and any relevant task information. The LLM then interacts with the scaffolding program in a sequence of steps, taking actions and observing their results. The scaffolding program will perform any actions or interactions with the task as instructed by the LLM; it will also return the outcome of these actions to the LLM agent. This allows the LLM to solve a task relatively autonomously (i.e. with little human input).

The two main elements of scaffolding are:
- Tool calling: This provides a description of a tool to the agent, which it can choose to use during the evauation. If it uses a tool then the scaffolding will execute this tool on the agent's behalf (usually consisting of running a python function), and return the result of this tool call to the agent.

- Prompting: This is how the task state is described to the LLM. We can also use prompting to help assist the LLM in its tool use, or instruct it to use chain-of-thought to give the LLM more "thinking time." 

The LLM interacts with scaffolding program to complete a task according to the following steps:

1. The LLM receives input (task description, current state, available actions etc.) from the scaffolding program
2. The LLM processes the input and outputs an action (e.g. "Use calculator tool")
3. The scaffolding program executes the action the agent took, and returns the outcome (e.g. it would run `calculate()` in the background for an LLM using a calculator, and then return the function output to the agent)
4. The LLM receive the results and decides the next action
5. Repeating the cycle until the task is complete


[Insert METR diagram]

Some examples of LLM agents are:

- [Voyager](https://arxiv.org/abs/2305.16291) (Minecraft LLM Agent)

- [AutoGPT](https://autogpt.net/)

- [LangChain](https://www.langchain.com/)


<!-- An LLM agent consists of 4 main things [I think a better list exists here, "reasoning engine" is quite unclear/vague and scaffolding doesn't make sense as a bullet point in how we've defined it; also maybe move this to start of section "Build agent?"].

- A 'reasoner' or 'reasoning engine.' (Some people also call this a 'world model'). For LLM agents this is a large language model.

- Tools which allow the agent to act in the environment.

- Memory so that the agent can recall prior actions. This can either be:

    - Short-term memory: In the context of LLM agents this is generally the context window

    - Long-term memory: There are many cases where context-windows are too short, and we will need to give the agent high-level information about actions it took a long time ago. There are many methods to store this 'long-term memory' for agents (see some methods [here])

- Scaffolding: This is essentially any structure which we provide to the 'reasoning engine' in order to help it to reason better, such as:

    - Prompting frameworks.

    - The ability to trial plans into the future.

    - Using subagents to take care of subtasks.

    - Subgoal decomposition.

EXCALIDRAW!

## How should we evaluate LLM agents?

Points to make - "Why evaluate LLM agents":
- overall note: I think this could heavily be based on this [video](https://www.youtube.com/watch?v=KO72xvYAP-w) from METR
- The purpose of agent evals is to **unlock and measure the full capabilities of a model**, to avoid underestimating the model and to better estimate the **ceiling** of their capability and potential to cause harm. 
- Models often fail in easily-fixable ways. For example, when it is solving a hacking task, it
    - Can refuse due to ethics or (claimed) inability 
    - Can give up and ask the user for help 
    - Can get stuck in loops 
    - Can hallucinate facts or conclusions [only partly fixable] 
    - Can be limited by primitive tools 
    - Can have bugs
    - ...
- For a model that fails to solve a hacking task, thus deemed safe, there might exist simple fixes (e.g. better prompts, better file manipulation tools) that unlock this dangerous capability. 
- 
- Final point about "quantifying" the amount of scaffolding to make eval results more quantitative
    - Apollo "Science of Evals" 
    - GDM quantify bits
==========================
--->
## How should we evaluate LLM agents?

There are two possible purposes of LLM agent evaluations:

- The first is to **unlock and measure the full capabilities of a model**. We don't want to underestimate current or future LLMs, so we want to establish the **ceiling** of their capabilties and potential to cause harm.
- The second is to **determine the alignment properties of LLMs in agentic scenarios**. Most of our current alignment techniques (Supervised Fine-tuning, RLHF, ... ) are focused on Chatbot contexts for LLMs, however LLM agents have the potential to cause much greater harm, and we currently aren't as confident about how RLHF and Supervised Fine-tuning will work in these contexts.

LLM agents generally fail in easy-to-fix ways, as you will see. For example:

- They often claim to be incapable of tasks that they can actually perform.

- They can easily get stuck in loops.

- They can give up and ask the user for help

- They can hallucinate facts, or even misunderstand their own prior reasoning and hallucinate a faulty conclusion.

- They can be overly or underly sensitive to information in their prompts.

- They can just have bugs

This means that when models fail to accomplish tasks, there may exist simple fixes that will unlock a capability. Since we want to eliminate the potential of large capability improvements from relatively little effort, this means that we have to try quite hard to tune the promptings, tool descriptions, and tool outputs just right, so that we can see LLM agents at their *best*.
<!--->
We know today that LLMs are more than just chat-bots. In fact, since the release of ChatGPT, the use of LLMs as  for agentic systems has proliferated signficantly. These agents started off rather disappointingly initially, when they were based on GPT-3.5. However as more powerful LLMs come out and AI companies ensure their LLMs are better at tool-use, these agents are improving rapidly.

The main concerns for LLM agents that we want to mitigate are:

- There are many possible improvements for increased performance from LLM agents, and these improvement methods are often signficantly cheaper and easier to implement than training the base model, for example by writing better prompts, or providing easier-to-use tools.

- Current fine-tuning and RLHF/Constitutional AI methods are mostly targeted towards chatbot-style text output. We aren't as confident about how such methods will generalize to agentic scenarios.

The first two issues here relate to the **capabilities** of LLM agents, and the last issue relates to the **alignment** properties of LLM agents. The agent we'll be building will be testing for the **capability** properties of agents.
<!--->

<details><summary>Further resources on LLM evaluations:</summary>

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)

- [Anthropic Function Calling Guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)

- [Evaluating Language-Model Agents on Realistic Autonomous Tasks](https://evals.alignment.org/Evaluating_LMAs_Realistic_Tasks.pdf) (Kinniment et al., ARC Evaluations Team (now METR), 2023)

- [Large Language Models can Strategically Deceive their Users when Put Under Pressure](https://arxiv.org/pdf/2311.07590) (Scheurer et al., Apollo Research, ICLR 2024)

- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) (Lilian Weng, OpenAI Safety Team, 2023)

- [AXRP Episode 34 - AI Evaluations with Beth Barnes](https://www.alignmentforum.org/posts/vACr4DExfeRMaCoo7/axrp-episode-34-ai-evaluations-with-beth-barnes) (Daniel Filan, 2024)

- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/pdf/2303.11366) (Shinn et al., 2023)

- [Answering Questions by Meta-Reasoning over Multiple Chains of Thought](https://arxiv.org/pdf/2304.13007) (Yoran et al., 2024)

- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/pdf/2302.04761) (Schick et al., META AI Research, 2023)
</details>
                
""",
        unsafe_allow_html=True,
    )
