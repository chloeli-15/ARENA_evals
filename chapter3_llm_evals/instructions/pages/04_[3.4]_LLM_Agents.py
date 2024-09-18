import os, sys
from pathlib import Path

instructions_dir = Path(__file__).parent.parent.resolve()
if str(instructions_dir) not in sys.path: sys.path.append(str(instructions_dir))
os.chdir(instructions_dir)

import streamlit as st

import st_dependencies
st_dependencies.styling()
from pages.sec4_llm_agents import (
    sec10_home,
    sec11_intro_to_llm_agents,
    sec12_build_a_simple_llm_arithmetic_agent,
    sec13_building_a_more_complex_agent_wikigame,
    sec14_elicitation,
    sec15_bonus
)

import platform
is_local = (platform.processor() != "")


import streamlit_antd_components as sac

with st.sidebar:
    st.markdown('')
    
    CHAPTER = sac.steps([
        sac.StepsItem(title='Home', icon="house"),
        sac.StepsItem(title='Intro to LLM Agents', subtitle='(5%)', icon="1-circle-fill"),
        sac.StepsItem(title='Build a Simple LLM Arithemtic Agent', subtitle='(10%)', icon="2-circle-fill"),
        sac.StepsItem(title='Building a more Complex Agent: Wiki Game', subtitle='(45%)', icon="3-circle-fill"),
        sac.StepsItem(title='Elicitation', subtitle='(30%)', icon="4-circle-fill"),
        sac.StepsItem(title='Bonus' , subtitle='(10%)', icon="5-circle-fill"),
    ], size='small', return_index=True)

    function = [
        sec10_home.section,
        sec11_intro_to_llm_agents.section,
        sec12_build_a_simple_llm_arithmetic_agent.section,
        sec13_building_a_more_complex_agent_wikigame.section,
        sec14_elicitation.section,
        sec15_bonus.section
    ][int(CHAPTER)]

function()