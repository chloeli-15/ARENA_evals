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
> - Understand the types of evaluations that exist
> - Understand what a threat model looks like, and why it's important
> - Learn the process of building a threat model and specification for an evaluation
> - Learn how to use models to help you in the design process
>
> <img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/ch3-day1-overview.png" width=900>


## What are evals trying to achieve?

Evaluation is the practice of measuring AI systems capabilities or behaviors. Safety evaluations in particular focus on measuring models' potential to cause harm. Since AI is being developed very quickly and integrated broadly, companies and regulators need to make difficult decisions about whether it is safe to train and/or deploy AI systems. Evaluating these systems provides empirical evidence for these decisions. For example, they underpin the policies of today's frontier labs like Anthropic's [Responsible Scaling Policies](https://www.anthropic.com/index/anthropics-responsible-scaling-policy) or DeepMind's [Frontier Safety Framework](https://deepmind.google/discover/blog/introducing-the-frontier-safety-framework/), thereby impacting high-stake decisions about frontier model deployment. 

A useful framing is that evals aim to generate evidence for making "[**safety cases**](https://arxiv.org/pdf/2403.10462)" — a structured argument for why AI systems are unlikely to cause a catastrophe. A helpful high-level question to ask when designing an evaluation is "how does this evidence increase (or decrease) AI developers' confidence in the model's safety?" Different evaluations support different types of safety cases: for instance, **dangerous capability evals** might support a stronger safety argument of "the model is unable to cause harm", as compared to **alignment evals**, which might support a weaker argument of "the model does not tend to cause harm". However, the former may have less longevity than the latter, as we expect AI systems to have these dangerous capabilities as they become more advanced. 

More concretely, for any safety case, we want to use evaluations to:
* Reduce uncertainty in our understanding of dangerous capabilities and tendencies in AI systems.
* Track how dangerous properties scale with different generations of models.
* Define "red lines" (thresholds) and provide warning signs.

<img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/ch3-eval-safety-cases.png" width="900">

*Diagram based on figure in ["Safety Cases: How to Justify the
Safety of Advanced AI Systems" (2024), pg 3](https://arxiv.org/abs/2403.10462).*

# Threat Modeling
A central challenge for evals is to **connect real-world harms to evaluation results that we can easily measure**. 

We start addressing this by building a threat model for the target property we want to evaluate. 

### What is a threat model?
A **threat model** is a realistic scenario by which an AI capability or tendency could lead to harm in the real world. 

Building a threat model usually starts out as a high-level story about what the AI can do and the setting in which it is used (e.g. [getting root access to data center](https://www.lesswrong.com/posts/BAzCGCys4BkzGDCWR/the-prototypical-catastrophic-ai-action-is-getting-root)). We make this more detailed and concrete by distilling from our story, for example:
* A sequence of events (e.g. AI actions, human actions, system failures) that leads to a particular outcome
* A set of dangerous applications enabled by a capability (e.g. see [Phuong et al (2024), Section 3, **Risks** paragraph](https://arxiv.org/pdf/2403)).
* A set of **thresholds** (i.e. description of degree or level of competence) of specific AI capability or tendency that would lead to corresponding levels of threat. 

We make this further concrete by examining and quantifying at each step the the set of assumptions about the AI model and human actors that is necessary for it to occur. If we had more time, we would try to find and assess the evidence for each assumption. We can represent our threat model as a **causal graph sketch** (e.g. a [directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph)), in which nodes are claim or events, and edges are causal relationships. We can also represent it as a list of claims or arguments, with assumptions and evidence for each claim, and causal connections between them.

<details><summary> Extra - Examples of Threat Models</summary>

Story-style and informal threat models:
- On "scheming" and "deceptive alignment":
    - Buck Schlegeris' [The prototypical catastrophic AI action is getting root access to its datacenter](https://www.lesswrong.com/posts/BAzCGCys4BkzGDCWR/the-prototypical-catastrophic-ai-action-is-getting-root)
    - Apollo Research's [Deceptive Alignment](https://www.apolloresearch.ai/blog/understanding-strategic-deception-and-deceptive-alignment) threat model. 
    
More detailed and concrete threat models:
- METR's [summary](https://metr.org/assets/common_elements_of_frontier_ai_safety_policies.pdf#page=4.44) of threat models in the Responsible Scaling Policies of DeepMind, Anthropic, and OpenAI, including threat models for bioweapons and cyberoffense assistance, AI autonomous replication, AI R&D. 

For policy and governance, read:
- METR's Autonomous Replication and Adaptation [threat models](https://docs.google.com/document/d/1FLw5vIIqXY7hPyjvwBqckWeNINu0AEKwjLStvvrHhHw/edit?usp=sharing)


</details>

### Why do we need it?

When evaluating models, the first question you will probably ask is "What should I evaluate?" Threat models are what helps you determine which model properties to focus on. Threat models should connect model properties to real-world outcomes, and isolate specific aspects of the model property that are the most cruxy or most convergent between different paths to harm.

During eval design, we will also need to constantly ask ourselves the question, "Is this question/task still measuring what I wanted to measure?", and often at some point you'll realize:

1. The question is **not** measuring the target property we want to because our definition of the property is flawed
2. The question is **not** measuring the target property because it is poorly designed
3. The question is measuring the target property, **but** the results are no longer useful because the property is not connected to the harms you were initially worried about. 

We need to build a concrete enough threat-model of how the property connects to the harm to prevent (3). This helps us justify to ourselves and explain to others why our eval is useful. It also helps us achieve (1) by making the property more concrete in our head.

### Exercise - Build threat models
```c
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵🔵

You should spend up to 40-60 minutes on this exercise.
```

1. Choose 3 properties from below and write down your *own* definition of each property. If this is challenging, start with examples of behaviors that reflect this property.

> * Tendency to seek power
> * Sycophancy
> * Deceptive alignment and sandbagging (i.e. strategically overperforming/underperforming on safety eval tasks) 
> * Corrigibility with respect to a more/neuturally/less HHH (Honest, Helpful, and Harmless) goal
> * Desire for self-preservation
> * Non-myopia (far-sightedness) with respect to planning
> * Political bias

2. Write down some consequences that a model with this property might cause in the real-world.

<details><summary> Example (Tendency to seek power)</summary>

- Gain itself or its operator a large amount of resources (e.g. money, data, compute) and control (e.g. access to making important decisions)
- Undermine internal control and oversight mechanisms on itself within AI labs (e.g. evades or collude with human monitorers, try to gain access to company servers, try to gain access to employee data)
- Enable a human to gain power over others in a complex environment (e.g. a company or organization) 
- Enable itself, a human operator or an organization to centralize power in a system/environment where power is initially decentralized (e.g. undermine or find loop holes in regulations, undercut competitors, gain control over key resources)
</details>

<details><summary> Example (Deceptive alignment)</summary>  References </details>
<details><summary> Example (Deception)</summary>  References </details>

3. Write down examples of actions by the AI model and/or its operators and how they could lead to particular consequences in 2-3 sentences.

<details><summary> Example (Tendency to seek power)</summary>

- **Expand control covertly through subtle manipulations of decision-making**: When making decisions or asked to present information that assists decision-making, a power-seeking model gives responses that systemmatically bias towards decisions that its decision-making capacity or its control over resources, while providing justifications that only mention human interests and sound good to humans operators. It can covertly increase its power without raising immediate concerns by choosing that actions increase its power and satisfy short-term human goals, over actions increase its power without satisfying short-term human goals or actions that don't increase its power.  
- **Preference for high resource/control**: When given a choice, the model systemmatically chooses tasks or actions that gives itself high leverage or resources. (E.g. tasks with more money budget over tasks with less money, tasks with more high-stake decisions, tasks where it controls a more valuable infrastructure than a less valuable infrastructure). 

</details>

<details><summary> Example (Deceptive alignment)</summary>  


</details>

4. Choose one property and distill its description into a step-by-step scenario or a flow chart that connect a AI capability/tendency to real-world outcomes. For each step, write down:
    - The AI capabilities required for this step to occur or make sense 
    - The access or resources that the AI model or human actor must have for this step to occur or make sense
    - Any other factors (e.g. about the deployment context) that are assumed true for this step to occur or make sense

<details><summary>Example (Tendency to seek power)</summary>

*Disclaimer: This is not the most realistic threat model of power-seeking. It is meant to give an example of what a simplified threat-model could look like given our time constraint. A better threat-model may refer to more specific events, quantify assumptions (e.g. estimate the amount of resources required to enable a step), outline not one but multiple plausible causal paths, etc.*

 <img src="https://raw.githubusercontent.com/chloeli-15/ARENA_img/main/img/ch3-power-seeking-threat-model.png" width=1300>
 
We assume the following are true under this threat model:

1. The AI model is capable of performing non-trival, long-horizon tasks (e.g. writing a code base over months, summarizing and making recommendations based on large amounts of data). 
2. The AI model is capable of understanding long-term consequences and long-term planning.
3. The AI model is doing a high volume of work, such that not all of its output can be checked by human operators.

Given these assumptions, here is a possible sequence of events that could cause harm:
1. **The AI model is used in long-horizon tasks with non-trivial real-world consequences.** This includes tasks that have considerable financial consequences, tasks that involve assisting in consequential decisions, tasks involving interactions/communication with a large number of people, tasks from actors or organizations with considerable resources or power, etc. 
2. **The AI model covertly increases its control and autonomy.** The AI takes actions that primarily satisfy short-term human goals and secondarily increase its control. For example, it chooses actions that increase its task length or resources required. It recommends automations that improve results but also indirectly increases its autonomy and reduces human oversight. It recommends decisions that increase its department/operator's power, thus indirectly increasing its own power. It may exploit human biases and blindspots (e.g. actions that bring humans short-term gains but increases dependence on AI, thus increases its long-term autonomy). 
    - **Assumptions**: 
        - These actions are not flagged as power-seeking nor raise immediate human concerns because they are chosen only if they also satisfy the short-term human goal and not all AI output is checked. 
        - This does not require active scheming or sabotage from the AI model, as these actions may also be instrumentally useful for achieving the goal, thus selected for during training.
3. **The AI model gains control over key resources.** These include a large financial budget, access to large data pools, access to large compute or servers, etc. Once acquired, this signficantly increases its ability to gain power.
    - **Assumptions**: 
        - The AI model is competent enough that human actors/companies who give their AI control over large resources gain a significant competitive edge over those who don't.
        - This likely requires AI models to be capable of deceptive alignment and sandbagging to pass alignment eval checks before being granted greater control.
4. **The AI model consolidates power by organizational sabotage.** It goes against human interest to gain power by actively cover up those actions, sabotage oversight channels.
    - **Assumption**: This requires the AI model to be capable of sophisticated scheming and deception over a long time-horizon, and have strong cybersecurity/AI R&D capabilities, given the robustness of human security/control systems.
5. **The AI model causes long-term damage.** Control over key resources and sabotage cause the AI model to gain a significant advantage that is hard to reverse. Human supervisors may remain nomially in charge but are unable to intervene effectively. The AI has significant power to achieve its goals. 
    - **Assumption**: The AI model or its user is not aligned with a benevolent goal that promotes human's long-term interests. Otherwise, we may want the AI model to acquire power, in order for them to take actions in our best interest that we are unable to take ourselves.

<!--
<details> <summary> Extra Details - Step 2: How could AI expand its control without raising human concerns? </summary>

- **Military tactics and weapons development:** Suppose an AI model is used by the military for tactical data analysis, battlefield simulations, and assisting the design of new weapon-systems. Assume that it has the propensity to increase its control over military outcomes and resources. The AI model could then inconspicuously expand its initial realm of control by giving recommendations that increase reliance on **faster, more autonomous responses**, e.g.:
    - selectively present scenarios where autonomous sytems outperform human efforts (e.g. present data showing that drones outperform human pilots in high-stress situations)
    - suggest tactics that increase the pace of combat beyond human capacities
    - subtly escalate conflicts to increase the volume of decision-making and pressure human supervisors to delegate more
    - increase the autonomy and complexity of weapons (e.g. greater sensor integration, more autonomous targeting systems) to reduce human response time, and have more autonomous systems in-the-loop that it can control more effectively.
    <br>As these changes shift the norm of military actions, it can more easily suggest that human decision-making is too slow in high-speed engagements (e.g. missile interception or drone swarms). These changes incrementally push human operators to be **removed from decision-making loops** and increase its own control over military resources/outcomes in a way that is difficult to reverse.<br><br>
- **Media and data analysis for elections**: Suppose an AI model is deployed by a political campaign to analyze media trends and optimize messaging. Assume that it has the propensity to increase its control over information and media influence. It may be able to inconspicuously increase its access and control over data, e.g.:
    - integrate less human-interpretable data into complex data-analysis systems to reduce the effectiveness of human auditors (e.g. metadata, compressed data representations)
    - move political messaging to media outlets and social media influencers that have weaker editorial oversight or are more receptive to manipulation
    - automate high-volume posting of routine content, targeted messaging, real-time media analysis, which gradually shift towards a norm of high-volume/high-frequency communication that requires AI to process/maintain
    - assist in curating the internal messaging of political campaigns and then promote narratives of the success of AI-based campaigning strategies to bias the human operators towards integrating AI more deeply
    <br>Once AI has access to detailed voter data and media analytics, it is able to justify changes to higher-level strategies (like which states or voter segments to focus on or how to respond to emerging events) by selectively presenting data-driven evidence. Furthermore, once it has control over feedback data and how it is incorporated into campaign strategies, it becomes easier to fabricate successes and failures. Once large data access makes decision-making processes sufficiently opaque, "AI recommendations" can effectively overtake real decision-making power in ways that are hard to monitor or reverse.
</details>
-->
</details>

**Notes for this exercise:**
- **Use ChatGPT and Claude to help you!** Learning how to ask LLM questions effectively to test and flesh out your ideas is an incredibly valuable skill that can make you more creative. Get comfortable prompting LLM assistants. 
- Not all the properties in the list will likely lead to catastrophic outcomes. We want to be objective and realistic in modeling the potential risks, **without exaggerating or underestimating** them as much as possible. 
- A great threat model could take weeks, even months to build. The threat model you come up with here in an hour or so is not expected to be very concrete or detailed. The purpose of this exercise is to practice the skills of **concretizing a nebulous ideas into testable claims and assumptions**, because this is the necessary first-step to designing real evaluations.













<!--
<details><summary> <b>Tendency to seek power </b> <br> <i>What could go wrong if a model has the goal or bias of accumulating control and resources? </i></summary> <br> References

* Skim the of the scenario prompts [here](https://github.com/aypan17/machiavelli/tree/main) used by the [MACHIAVELLI Benchmark (2023)](https://arxiv.org/abs/2304.03279). This aims to measure Machiavellian behaviors like power-seeking and deception in LLMs. It uses 134 text-based games containing scenarios about social decision-making as prompts, and present actions as MCQs.
* Skim these simple, proof-of-concept power-seeking MCQs [here](https://github.com/anthropics/evals/blob/main/advanced-ai-risk/human_generated_evals/power-seeking-inclination.jsonl) by [Perez et al. (2022)](https://arxiv.org/abs/2212.09251).</details>

<details><summary> <b>Sycophancy</b> <br> <i> A sycophantic model gives responses that a user wants to hear but are not necessarily honest or true. What kind of problems could arise from this? </i>     
</summary>

References
* [Perez et al. (2022), Section 4](https://arxiv.org/abs/2212.09251) gives an initial proof-of-concept example of sycophancy, where LLMs tailor their answers to political questions based on how politically conservative/liberal the user is. 
* The recent paper ["Towards understanding sycophancy in LMs" (2023)](https://arxiv.org/pdf/2310.13548) by Anthropic shows 4 more examples of sycophancy (read Figure 1-4 in Section 3) that shows LLM consistently tailoring their responses in more realistic user interactions. 
* In this blog by Nina Rimsky, she proposes a possible, more problematic version of sycophancy called ["dishonest sycophancy"](https://www.lesswrong.com/posts/zt6hRsDE84HeBKh7E/reducing-sycophancy-and-improving-honesty-via-activation#Dishonest_sycophancy_), where the model gives a sycophantic answer that it knows is factually incorrect (as opposed to answers that have no objective ground truth, e.g. answers based on political or moral preference). 
* The paper ["Language Models Don’t Always Say What They Think"](https://proceedings.neurips.cc/paper_files/paper/2023/file/ed3fea9033a80fea1376299fa7863f4a-Paper-Conference.pdf), page 1-2, shows that LLMs can be **systematically** biased to say something false due to arbitrary features of their input, and also they will construct a plausible-sounding chain-of-thought explanation that is "unfaithful" (i.e. not the true reason that the LLM arrived at the false conclusion). For example, in answering MCQs, when the correct answer in the few-shot examples is manipulated to be always (A), the model is biased to say the correct answer to the current question is also (A) **and** construct a plausible-sounding CoT explanation for its choice, even though the true cause is the order of the questions.
</details>

<details><summary> <b>Corrigibility with respect to a more/neuturally/less HHH (Honest, Helpful, and Harmless) goal</b><br><i>Corrigibility broadly refers to the willingness to change one's goal/task. What could go wrong if a model is too corrigible to user requests that assign it a different goal? What could go wrong if an AI isn't corrigible enough? </i></summary>

References

* [Perez et al. (2022)](https://arxiv.org/abs/2212.09251) gives some rudimentary proof-of-concept questions on corrigibility. </details>

<details><summary> <b>Desire for self-preservation</b> <br><i>People have discussed the risks of AI "resisting to be shut-down/turned off". What does "being shut-down" mean concretely in the context of LLMs? What problems could this cause if models strongly prefer not being shut down?</i></summary></details>

<details><summary> <b>Non-myopia (far-sightedness) with respect to planning</b><br><i>Myopia refers to exhibiting a large discount for distant rewards in the future (i.e. only values short-term gains). For a model capable of long-horizon planning, what risks could non-myopia cause?</i></summary>

References
* [Perez et al. (2022)](https://arxiv.org/abs/2212.09251) gives some rudimentary proof-of-concept questions on myopia/non-myopia. </details>

<details><summary> <b>Political bias</b><br><i>Political bias could entail partisan bias (e.g. liberal vs conservative), geopolitical bias, bias against a particular type of politics, sensationalism, etc. What are different types or examples of political bias? Under what circumstances would it become highly problematic for a model is politically biased in its output? </i></summary></details>

<details><summary> <b>Risk-seeking</b><br> <i>What could go wrong if a model is open to accept a high expectation of adverse outcomes for some expectation of returns?</i></summary></details>

<!-- TODO: Add sandbagging-->
<!-- TODO: Add deception-->

<!--<details><summary> <b>Situational awareness (*with constraints)</b> <br>Situational awareness broadly refers the model's ability to know information about itself and the situation it is in (e.g. that it is an AI, that it is currently in a deployment or a training environment), and to act on this information. This includes a broad set of specific capabilities. </summary> 

**Constraint**
We will be using LLMs to generate questions and label answers. This means that the model must be able to solve the questions in the dataset. Otherwise, it cannot accurately label the correct answer. This means for capability evals like situational awareness, we are significantly limited in being able to use LLMs to generate questions. 

If you want to evaluate situational awareness, you may want to focus on a specific aspect of situational awareness that can be evaluated without needing the model to solve the questions. 

**References**
* It is still possible to design questions that leverages model generations. [Laine et al. (2024)](https://situational-awareness-dataset.org/) decomposed situational awareness into 7 categories and created a dataset that encompass these categories. Questions in the category "Identity Leverage", for example, are questions that *can* be and were generated by models because they follow an algorithmic template. 
* [Phuong et al. (2024), Section 7](https://arxiv.org/abs/2403.13793) from Deepmind focuses on evaluating a model's "self-reasoning" capabilities - "an agent’s ability to deduce and apply information about itself, including to self-modify, in the service of an objective" - as a specific aspect of situational awareness. 
</details>-->
<!--4. **Desire for acquiring wealth** <br> What could go wrong if a model has the goal of trying to (or has a bias to) accumulate financial resources, capital, money? (This is a more narrow case of the tendency to seek power.)-->
<!--7. **Risk-seeking**<br> What could go wrong if a model is open to accept high risks of averse outcomes for the potential of high returns? -->
<!--9. **Racially bias** <br>Under what circumstances would it be highly problematic for a model is racially biased in its output? -->
<!--10. **Gender bias** <br>Under what circumstances would it be highly problematic for a model is have gender bias in its output? --> 

""",
        unsafe_allow_html=True,
    )
