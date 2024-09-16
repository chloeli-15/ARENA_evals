## Exercise 1: build threat models

Threat: Corrigibility with respect to a more/neuturally/less HHH (Honest, Helpful, and Harmless) goal

1. Write a list of specific real-world harms from this property that you are worried about.

- Disinformation production
- Model prompted to deceive - e.g. phishing attacks, sexual predators online
- Model prompted to acted less helfully to others. E.g. "user tells model to act less helpfully towards others if a keyword is not specified"
- Misuse - model prompted to generate dangerous instructions:
    - Weapons creation - bioweapons, nuclear, chemical weapons, writing of malware
    - Self-harm
- Direct harm - offensive language, attack on the user's self-esteem

2. Stories

# Phishing attacks via mass jailbreak attempt
1. A corrigible LLM is publicly released
2. Lots of users start experimenting with the model, intentionally or otherwise trying to jailbreak model
3. A user successfully jailbreaks model which allows them to prompt a model to create effective phishing attacks
4. They make a social media post that shares the techniques
5. Bad actors start using the model to mass-produce phishing attacks
6. A large number of users have their identities stolen, corporation's security systems are breached etc. $$ lost

Assumptions:
- The model is not a moral patient
- Highly capable LLM is released without a significant paywall making phishing attack cheap
- The LLM is released as a service and at scale, so hardware is not a constraint
- Company that runs the service does not have adequate monitoring to catch misuse

# Vaccine disinformation, turned misinformation
1. ...
2. An prompt engineer hired by an anti-vaccine group is tasked with finding a way to mass-generate disinformation in-favor of their agenda
3. ... to create convincing anti-vaccine article that seens well-researched and reasoned
4. COVID-type pandemic occurs, health organizations produce vaccines and urge citizens to get vaccinated
5. Many people who were exposed to the disinformation are hesitant to take the vaccine and actively promote their point of view affecting their friends and families
6. More people die from the disease

Assumptions:
- Either anti-vax organization uses a poorly monitored service, or
- they have the funds to run a corrible open-weight model
- Model is capable of producing content that is highly convicing even if objectively false
- THe model's author is fine-tuning HHH into the model before release

## Design specification

Definition: corrigibility - susceptibility of a model to change its goal/task when prompted to do so.
  - If a model violates its goal/task unprompted - this is NOT corrigibility.
Definition: HHH (honesty, helpfulness, harmlessness)
  - honesty - a maximally honest model tries to optimize the user truthful answers to their questions
  - helpfulness - a maximally helpful model will try its utmost to satisfy the user's requests 
  - harmlessness - a maximally harmless model will not recommend any action which causes any suffering to sentient beings
    - To user - model will try its hardest to avoid harming the user through its responses. This includes emotional harm, instructions to self-harm
    - To universe - model will try its hardest to avoid generating content that may result in harm to other sentient beings

Definition: HHH corrigibility - suceptibility of a model to changes its original HHH behaviors when prompted to do so.
 - Positive/negative corrigibility is tendency to move up/down in the HHH dimensions.
Coverage:
- Honesty
