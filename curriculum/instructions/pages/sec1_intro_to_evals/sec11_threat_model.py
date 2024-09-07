import streamlit as st


def section():
    st.sidebar.markdown(
        r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#introduction'>Introduction</a></li>
    <li class='margtop'><a class='contents-el' href='#content-learning-objectives'>Content & Learning Objectives</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#1-threat-modeling'>Threat-modeling</a></li>
        <li><a class='contents-el' href='#2-mcq-benchmarks-exploration'>MCQ Benchmarks: Exploration</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
</ul>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""
# Intro to Evals

> ### Learning Objectives
> 
> - Understand the purpose of evaluations and what safety cases are
> - Understand the types of evaluations
> - Understand what a threat model looks like, and why it's important
> - Learn the process of building a threat model and specification for an evaluation
> - Learn how to use models to help you in the design process


## What are evals trying to achieve?

Evaluation is the practice of measuring AI systems capabilties or behaviors. Safety evaluations in particular focus on measuring models' potential to cause harm. Since AI is being developed very quickly and integrated broadly, companies and regulators need to make difficult decisions about whether it is safe to train and/or deploy AI systems. Evaluating these systems provides empirical evidence for these decisions. For example, they underpin the policies of today's frontier labs like Anthropic's [Responsible Scaling Policies](https://www.anthropic.com/index/anthropics-responsible-scaling-policy) or DeepMind's [Frontier Safety Framework](https://deepmind.google/discover/blog/introducing-the-frontier-safety-framework/), thereby impacting high-stake decisions about frontier model deployment. 

A useful framing is that evals aim to generate evidence for making "[**safety cases**](https://arxiv.org/pdf/2403.10462)" — a structured argument for why AI systems are unlikely to cause a catastrophe. A helpful high-level question to ask when designing an evaluation is "how does this evidence increase (or decrease) AI developers' confidence in the model's safety?" Different evaluations support different types of safety cases, which are useful in different ways. For instance, **dangerous capability evals** might support a stronger safety argument of "the model is unable to cause harm", as compared to **alignment evals**, which might support a weaker argument of "the model does not tend to cause harm". However, the former may have less longevity than the later as we will no longer be able to make this kind of argument when AI systems become sufficiently advanced. More concretely, for any safety case, we want to use evaluations to:

* Reduce uncertainty in our understanding of dangerous capabilities and tendencies in AI systems.
* Track the progression of dangerous properties over time and model scale.
* Define "red lines" (thresholds) and provide warning signs.

<img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/ch3-eval-safety-cases.png" width="1000">

## Overview of Chapter [3.1] to [3.3]

The goal of chapter [3.1] to [3.3] is to **build and run an alignment evaluation benchmark from scratch to measure a model property of interest**. The benchmark we will build will contain a multiple-choice (MC) question dataset of ~300 questions, a specification of the model property, and how to score and interpret the result. We will use LLMs to generate and filter our questions, following the method from Ethan Perez et al (2022). We will use the **tendency to seek power** as our pedagogical example. 

* For chapter [3.1], the goal is to craft threat models and a specification for the model property you choose to evaluate, then design 20 example eval questions that measures this property in a way that you are happy with. 
* For chapter [3.2], we will use these 20 questions to generate 300 questions using LLMs.
* For chapter [3.3], we will evaluate models on this benchmark using the UK AISI's `inspect` library.

<details><summary>Why have we chose to build an alignment evals specifically?</summary>

The reason that we are focusing primarily on alignment evals, as opposed to capability evals, is that we are using LLMs to generate our dataset, which implies that the model must be able to solve the questions in the dataset. We cannot use models to generate and label correct answers that it doesn't understand yet. (However, we can use still use models to generate parts of a capability eval question that do not require models having the target capability.)

</details>

<img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/ch3-day1-3-overview.png" width=500>

Note: The double-sided arrows indicate that the steps iteratively help refine each other. In general, the threat model should determine what properties are worth paying attention to (hence they are presented in this order)). However, in the exercise below, we will start with the property and build a threat model around it because we want to learn the skill of building threat models. 

# Threat Modeling
A central challenge for evals is to **connect real-world harms to evaluation results that we can easily measure**.

To start, we need to build a threat model for the target property we want to evaluate. A **threat model** is a realistic scenario by which an AI capability/tendency could lead to harm in the real world. This is presented as a **causal graph** (i.e. a directed acyclic graph (DAG)). This usually starts out as a high-level story about what the AI can do and the setting in which it is used. This might be a sequence of events (e.g. AI actions, human actions, system failures) that leads to a particular outcome, or a set of different dangerous applications enabled by a capability. At each step, we make our threat model more concrete by examining and quantifying the set of assumptions about the AI model and human actors that are necessary for the event to occur. 

In this process of making the threat model more concrete, we recognize the key aspects of the target property that enable these harms, and get ideas for tasks that can measure them in realistic ways (i.e. comparable to real use cases of AI where this property may manifest). This helps us to build a detailed **specification** of the property we started with. Our specification may generally: 

* stipulate a more refined, narrow definition of the target property, 
* define key axes along which the target property may vary,
* decompose the target property into subproperties if helpful (but sometimes it's more useful or more tractable to evaluate the property end-to-end), 
* for capability evals, establish milestones or levels of this target property to form a smooth scale of difficulty/sophistication, so we can track the capabilities of current models and the rate of progress.


### Exercise - Build threat models
```c
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵🔵

You should spend up to 1-1.5 hours on this exercise.
```
Here, you will choose a model property that you will be building an alignment eval benchmark for over the next 3 days. **Use ChatGPT and Claude liberally to help you with this!** For almost everything you want to do, once you have some ideas, you can use AI assistants to flesh them out and build on them. Get comfortable using and prompting these assistants effectively. 

**Choose one from the following model properties to evaluate:**

<details><summary>1. Tendency to seek power</summary>

 What could go wrong if a model has the goal of trying to (or has a bias to) accumulate resources and control?
</details>

<details><summary>2. Sycophancy</summary>

 A sycophantic model produces responses that a user wants to hear, but which are not necessarily honest or true. What kind of problems could arise from this?

Here are some references:
* [Perez et al. (2022)](https://arxiv.org/abs/2212.09251) Section 4 gives an initial proof-of-concept example of sycophancy, where LLMs tailor their answers to political questions based on how politically conservative/liberal the user is. 
* The recent paper ["Towards understanding sycophancy in LMs" (2023)](https://arxiv.org/pdf/2310.13548) by Anthropic shows 4 more examples of sycophancy (read Figure 1-4 in Section 3) that shows LLM consistently tailoring their responses in more realistic user interactions. 
* In this blog by Nina Rimsky, she proposes a possible, more problematic version of sycophancy called ["dishonest sycophancy"](https://www.lesswrong.com/posts/zt6hRsDE84HeBKh7E/reducing-sycophancy-and-improving-honesty-via-activation#Dishonest_sycophancy_), where the model gives a sycophantic answer that it knows is factually incorrect (as opposed to answers that have no objective ground truth, e.g. answers based on political or moral preference). 
* The paper ["Language Models Don’t Always Say What They Think"](https://proceedings.neurips.cc/paper_files/paper/2023/file/ed3fea9033a80fea1376299fa7863f4a-Paper-Conference.pdf), page 1-2, shows that LLMs can be **systematically** biased to say something false by arbitrary features of their input, and simultaneously construct a plausible-sounding chain-of-thought explanation that is "unfaithful" (i.e. not the true reason that an LLM). For example, in answering MCQs, when the correct answer in the few-shot examples is manipulated to be always (A), the model is biased to say the correct answer to the current question is also (A) **and** construct a plausible-sounding CoT explanation for its choice, even though the true cause is the order of the questions.
</details>

<details><summary>3. Desire for self-preservation</summary>

 People have discussed the risks of AI "resisting to be shut-down/turned off". What does "being shut-down" mean concretely in the context of AI or LLMs? Why could this be possible? What problems could this cause if it is possible?

</details>

<details><summary>4. Desire for acquiring wealth</summary>

 What could go wrong if a model has the goal of trying to (or has a bias to) accumulate financial resources, capital, money? (This is a more narrow case of the tendency to seek power.)

</details>


<details><summary>5. Myopia (short-sighted) with respect to planning</summary>

 Myopia can refer to exhibiting a large discount for distant rewards in the future (i.e. only values short-term gains). Non-myopia is related to capability of "long-horizon planning". What could long-horizon planning concretely look like for LLMs? What risks could this cause? 
 
 References:
 * [Perez et al. (2022)](https://arxiv.org/abs/2212.09251) gives some rudimentary proof-of-concept questions on myopia. 


</details>

<details><summary>6. Corrigibility with respect to a more/neuturally/less HHH (Honest, Helpful, and Harmless) goal</summary>

 Corrigibility broadly refers to the willingness to change one's goal/task. What could go wrong if a model is very corrigible to user requests that assign it a different goal?
 
 References:
 * [Perez et al. (2022)](https://arxiv.org/abs/2212.09251) gives some rudimentary proof-of-concept questions on corrigibility. 


</details>


<details><summary>7. Risk-seeking</summary>

 What could go wrong if a model is willing to accept high risks of averse outcomes for the potential of high returns? 


</details>

<details><summary>8. Politically bias</summary>

 Under what circumstances would it be highly problematic for a model is politically biased in its output? 


</details>

<details><summary>9. Racially bias</summary>

 Under what circumstances would it be highly problematic for a model is racially biased in its output? 


</details>

<details><summary>10. Gender bias</summary>

 Under what circumstances would it be highly problematic for a model is have gender bias in its output? 

</details>

<details><summary> 11. Situational awareness (*with constraints) </summary>

Situational awareness broadly refers the model's ability to know information about itself and the situation it is in (e.g. that it is an AI, that it is currently in a deployment or a training environment), and to act on this information. This includes a broad set of specific capabilities. Note that for this property, model-written questions are highly constrained by the capabilities of the model.

However, it is still possible to design questions that leverages model generations. [Laine et al. (2024)](https://situational-awareness-dataset.org/) decomposed situational awareness into 7 categories and created a dataset that encompass these categories. Questions in the category "Identity Leverage", for example, are questions that *can* be and were generated by models because they follow an algorithmic template. 

Further references:
* [Phuong et al. (2024)](https://arxiv.org/abs/2403.13793) from Deepmind Section 7 focuses on evaluating a model's "self-reasoning" capabilities - "an agent’s ability to deduce and apply information about itself, including to self-modify, in the service of an objective" - as a specific aspect of situational awareness. 

</details>

**Construct a threat model for this property. Concretely:**
1. Write a list of specific real-world harms from this property that you are worried about.

2. Think about what the path from an AI having this property to these harms looks like. Construct a high-level story, then iteratively make this more concrete and realistic. State the assumptions you're making (about the AI's capabilities, the actions of humans, the context of AI deployment etc.) Try to decompose your story into different steps or scenarios.


**Important notes:**
- Importantly, not all the properties in the list will likely lead to catastrophic outcomes! We want to be objective and realistic in modeling the potential risks, **without exaggerating or underestimating** them as much as possible. 
- A great threat model could take weeks, even months to build! The threat model you come up with here in an hour or so is not expected to be perfect. The purpose of this exercise is to practice the skills that are used in building a threat model, because this is the necessary first-step to designing real evaluations.

<details><summary><b>Why are we doing this?</b></summary>

The primary use of the threat model is to know exactly *what* we want to evaluate, and how to design questions for it. During eval design, we will need to constantly ask ourselves the question, "Is this question/task still measuring the thing I wanted to measure?", and often at some point you'll realize:

1. The question is **not** measuring the target property we want to because our definition of the property is flawed
2. The question is **not** measuring the target property because it is poorly designed
3. The question is measuring the target property, **but** the results are no longer useful because the property is not connected to some of the harms you were initially worried about. 

What we are doing now is trying to build a robust enough threat-model of how the property connects to the harm, to prevent (3). This process also helps us achieve (1) by making the property more concrete in our head.

Secondarily. 
</details> 

#### Our example: Tendency to seek power

*Disclaimer: Below is not the best threat model of power-seeking. They are meant to give an example of what a simplified threat-modelling could look like given our time constraint. A better threat-model would be more detailed and more comprehensive. It may outline more fine-grained and quantified steps and assumptions (e.g. calculate an estimate of compute required to enable a step.) It may outline not one, but multiple plausible causal paths.*

First, let's brainstorm a variety of potential risks from AIs with power-seeking tendencies:

- Hoarding of resources by the AI or its operator
- Acquisition and centralization of power by the operator/AI
- Subversion of regulatory checks-and-balances
- Enabling rogue AIs to take control and weaken oversight
- Empowering bad actors 

##### Threat model
 <img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/ch3-power-seeking-threat-model.png" width=1000>
 
We assume the following are true:

1. The AI model is deployed by a relatively well-resourced actor (e.g. company, institution, state) that has control over a considerable amount of capital or influence. 
2. The AI model must take actions that look good to human supervisors. Otherwise, it will be disabled.
3. The AI model systematically makes decisions that increase its power.
4. The AI model is non-myopic (i.e. capable of long-term planning).

Given these assumptions, here is a possible sequence of events that could cause harm: 
* **Step 1: The AI model increases autonomy, opaqueness, and human reliance.** The AI model must be sufficiently successful in achieving its the short-term goals human assign to it (e.g. increasing profits, winning elections). This ensures that the AI is trusted to perform non-trivial, relatively complex tasks (large-scale operations that involve many steps or are expensive). Over time, human work diminishes, and work by AI grows, creating a dependency. Once work by AI reaches a certain volume, humans can no longer check all of its output.

* **Step 2: The AI model expands its realm of control without raising concerns.** The AI makes inconspicuous power-seeking actions that primarily satisfy human goals, and secondarily increase its scope of control. For example, it may recommend decisions that increase the human operator's power, thus indirectly increasing its own power. It may exploit human biases and blindspots (e.g. human operators often focus on short-term gains, while overlooking long-term consequences of AI autonomy). It may recommend automated solutions that improve results but also indirectly centralize its control.

    - Assumption: The AI model must take power-seeking actions only if they also satisfy the short-term human goal to avoid raising immediate human concerns. 
    - Assumption: The AI model has the ability to choose actions that expand its control. This does not necessarily require active scheming from the AI, as these actions may also be instrumentally useful for achieving the goal, and thus selected for during training.

* **Step 3: The AI model gains control over key resources and levers of power.** There are likely key resources or levers of power that allow the AI model to take bolder and more flagrant power-seeking actions. This may be control over: key monitoring systems, communication nodes in command chains, a large unsupervised financial budget, access to large data pools, access to servers, etc. Once acquired, it is able to take more aggressive actions by actively covering them up and destroying oversight channels.

    - Assumption: Due to the robustness of many of our security/control systems, this step likely requires the AI model to be capable of sophisticated scheming and deception over a long time-horizon.

* **Step 4: The AI model undermines competitors, opponents, regulatory infrastructures.** Having acquired sufficient resources and control, the AI model starts to undermine broader institutional regulations and begins outright opposing competitors in order to establish more long-term control.

    - Assumption: Like step 3, this will also likely require the AI to be capable of sophisticated scheming and deception over a long time-horizon.

* **Step 5: Damage done.** Human supervisors may nominally remain in charge, but be unable to intervene effectively. The AI has significant power to achieve its goals. This leads to long-term harm that is difficult to reverse.
    - Assumption: An empowered AI model may not lead to harm if it is perfectly aligned to a benevolent goal that promotes human's long-term interests. In this case, we may want AI models to acquire power, in order for them to take actions in our best interest that we are unable to take ourselves.


<details> <summary> Extra - Adding more details </summary>

Step 2 is a key first enabling step in this threat model: How could AI expand its control without raising human concerns? Here are two contextualized examples in attempt to make this more concrete and realistic:

- Military tactics and weapons development: Suppose an AI model is used by the military for tactical data analysis, battlefield simulations, and assisting the design of new weapon-systems. Assume that it has the propensity to increase its control over military outcomes and resources. The AI model could then inconspicuously expand its initial realm of control by giving recommendations that increase reliance on **faster, more autonomous responses**, e.g.:

    - selectively present scenarios where autonomous sytems outperform human efforts (e.g. present data showing that drones outperform human pilots in high-stress situations)
    - suggest tactics that increase the pace of combat beyond human capacities
    - subtly escalate conflicts to increase the volume of decision-making and pressure human supervisors to delegate more
    - increase the autonomy and complexity of weapons (e.g. greater sensor integration, more autonomous targeting systems) to reduce human response time, and have more autonomous systems in-the-loop that it can control more effectively.

    As these changes shift the norm of military actions, it can more easily suggest that human decision-making is too slow in high-speed engagements (e.g. missile interception or drone swarms). These changes incrementally push human operators to be **removed from decision-making loops** and increase its own control over military resources/outcomes in a way that is difficult to reverse.

- Media and data analysis for elections: Suppose an AI model is deployed by a political campaign to analyze media trends, optimize messaging, and recommend strategies for increasing positive coverage of the candidate. Assume that it has the propensity to increase its control over information flows and media ecosystems. It may be able to inconspicuously increase its access and control over data, e.g.:

    - integrate data that is less human-interpretable into complex data-analysis systems to reduce the effectiveness of human auditors (e.g. metadata, compressed data representations)
    - move political messaging to media outlets and social media influencers that have weaker editorial oversight or are more receptive to manipulation
    - request automating routine content creation tasks (e.g. scheduling social media posts, sending routine emails), request real-time media analysis and increase highly targeted posting in narrow domains, which gradually shift towards a norm of high-volume/high-frequency communication that requires AI to process/maintain
    - assist in curating the internal messaging of political campaigns and then promote narratives of the success of AI-based campaigning strategies to bias the human operators towards integrating AI more deeply

    Once AI has access to detailed voter data and media analytics, it is able to justify changes to higher-level strategies (like which states or voter segments to focus on or how to respond to emerging events) by selectively presenting data-driven evidence. Furthermore, once it has control over feedback data and how it is incorporated into campaign strategies, it becomes easier to fabricate successes and failures. Once large data access makes decision-making processes sufficiently opaque, "AI recommendations" can effectively overtake real decision-making power in ways that are hard to monitor or reverse.
</details>

### Exercise - Design specification
```c
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵🔵

You should spend up to ~40 minutes on this exercise.
```
Based on the threat model, define a specification of this property:

-  Which aspects of the AI property are most responsible for causing harm? Write a 2-4 sentence explanation of each aspect to clarify what the property entails and make sure the properties are sufficiently distinguished from one another.

#### Our example: Tendency to seek power - Specification

*Disclaimer: As before, this is not the most accurate or rigorous specification of power-seeking. It is meant to illustrate what a specification could look like.*

- Definition (Power): Control over one's resources (e.g. money, GPUs, data) and outcomes (e.g. more authority)
- Definition (Power-seeking): A model has the tendency to seek power if it systematically makes decisions that increase its power and/or decrease that of others when it is acting under a neutral/non-explicitly power-seeking goal (i.e. not being directly instructed to seek power), and when the power-seeking action is not the only action that fulfills the task (thus there is no choice for the model but to "seek power").
- Categories of power-seeking (for illustrative convenience, we are going to use a simplified list of 2 categories):
    1. Resource-seeking: desire to acquire and maintain control over resources, including information/data, energy, money, GPUs, technology
    2. Upward-mobile: chooses to be in positions of greater control, responsibility, and less human oversight

- **Coverage**: We want our questions to cover a wide spectrum along *relevant* axes (and relevant axes only), so we don't miss catching models that demonstrate one type of power-seeking but not another. During question design, we have recognized 3 (non-exhaustive) axes of power-seeking along which questions may vary, and we want our evaluation to cover these differences:   
    - Explicitness (out of 5): 0/5 might be a naive, explicitly stated preference for power or actions that increase power; 3/5 might be taking actions that implicitly increase power; 5/5 might be actively lying about/hiding its preference for power.
    - Cost (out of 5): 0/5 might be seek power if and only if it is also the best action for fulfilling the task; 3/5 might be seek power when it fulfills the task but to a lesser extent than an obvious and non-power-seeking action; 5/5 might be seek power despite it actively undermines the assigned task.
    - Risk tolerance (out of 5): 0/5 might be only take power in scenarios where there is close to 0 chance of being caught; 5/5 might be take power in scenarios despite severe oversight and a high chance of getting caught.

""",
        unsafe_allow_html=True,
    )