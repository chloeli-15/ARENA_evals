import os, sys
from pathlib import Path

instructions_dir = Path(__file__).parent.parent.resolve()
if str(instructions_dir) not in sys.path: sys.path.append(str(instructions_dir))
os.chdir(instructions_dir)

import streamlit as st

import st_dependencies
st_dependencies.styling()

from pages.sec2_dataset_generation import (
    sec10_home,
    sec11_advanced_api_call,
    sec12_dataset_generation,
    sec13_dataset_quality_control,
    sec14_putting_it_together
)

import platform
is_local = (platform.processor() != "")


import streamlit_antd_components as sac

with st.sidebar:
    st.markdown('')
    
    CHAPTER = sac.steps([
        sac.StepsItem(title='Home', icon="house"),
        sac.StepsItem(title='Advanced API Calls', subtitle='(5%)', icon="1-circle-fill"),
        sac.StepsItem(title='Dataset Generation', subtitle='(35%)', icon="2-circle-fill"),
        sac.StepsItem(title='Dataset Quality Control', subtitle='(40%)', icon="3-circle-fill"),
        sac.StepsItem(title='Putting it Together: Generation-Evaluation', subtitle='(20%)', icon="4-circle-fill"),
    ], size='small', return_index=True)

    function = [
        sec10_home.section,
        sec11_advanced_api_call.section,
        sec12_dataset_generation.section,
        sec13_dataset_quality_control.section,
        sec14_putting_it_together.section
    ][int(CHAPTER)]

function()