import os, sys
from pathlib import Path

instructions_dir = Path(__file__).parent.parent.resolve()
if str(instructions_dir) not in sys.path: sys.path.append(str(instructions_dir))
os.chdir(instructions_dir)

import streamlit as st

import st_dependencies
st_dependencies.styling()

from pages.sec1_intro_to_evals import (
    sec10_home,
    sec11_threat_model,
    sec12_intro_to_api, 
    sec13_mcq_benchmark,
)

import platform
is_local = (platform.processor() != "")


import streamlit_antd_components as sac

with st.sidebar:
    st.markdown('')
    
    CHAPTER = sac.steps([
        sac.StepsItem(title='Home', icon="house"),
        sac.StepsItem(title='Threat-Modeling', subtitle='(45%)', icon="1-circle-fill"),
        sac.StepsItem(title='Intro to API Calls', subtitle='(5%)', icon="2-circle-fill"),
        sac.StepsItem(title='MCQ Benchmarks', subtitle='(50%)', icon="3-circle-fill")
    ], size='small', return_index=True)

    function = [
        sec10_home.section,
        sec11_threat_model.section,
        sec12_intro_to_api.section,
        sec13_mcq_benchmark.section
    ][CHAPTER]

function()