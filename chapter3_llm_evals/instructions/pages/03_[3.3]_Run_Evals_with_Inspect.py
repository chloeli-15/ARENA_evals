import os, sys
from pathlib import Path

instructions_dir = Path(__file__).parent.parent.resolve()
if str(instructions_dir) not in sys.path: sys.path.append(str(instructions_dir))
os.chdir(instructions_dir)

import streamlit as st

import st_dependencies
st_dependencies.styling()
from pages.sec3_evals_with_inspect import (
    sec10_home,
    sec11_intro_to_inspect,
    sec12_writing_solvers,
    sec13_writing_tasks_and_evaluating,
    sec14_log_files_and_plotting
)

import platform
is_local = (platform.processor() != "")


import streamlit_antd_components as sac

with st.sidebar:
    st.markdown('')
    
    CHAPTER = sac.steps([
        sac.StepsItem(title='Home', icon="house"),
        sac.StepsItem(title='Intro to Inspect', subtitle='(15%)', icon="1-circle-fill"),
        sac.StepsItem(title='Writing Solvers', subtitle='(45%)', icon="2-circle-fill"),
        sac.StepsItem(title='Writing Tasks and Evaluating', subtitle='(25%)', icon="3-circle-fill"),
        sac.StepsItem(title='Bonus: Log Files and Plotting', subtitle='(15%)', icon="4-circle-fill"),
    ], size='small', return_index=True)

    function = [
        sec10_home.section,
        sec11_intro_to_inspect.section,
        sec12_writing_solvers.section,
        sec13_writing_tasks_and_evaluating.section,
        sec14_log_files_and_plotting.section
    ][int(CHAPTER)]

function()