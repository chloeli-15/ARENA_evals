import os
import sys
from pathlib import Path
import importlib

# Add the curriculum directory to the Python path
repo_dir = Path(__file__).parent.absolute()
curriculum_dir = repo_dir / "curriculum"
if str(curriculum_dir) not in sys.path:
    sys.path.append(str(curriculum_dir))

import streamlit as st
import streamlit_antd_components as sac

# Function to dynamically import section modules
def import_section(section_path):
    module_path = f"instructions.pages.{section_path.replace('/', '.')}"
    module = importlib.import_module(module_path)
    return module.section

# Set up the sidebar navigation
with st.sidebar:
    st.markdown('')
    
    CHAPTER = sac.steps([
        sac.StepsItem(title='Home', icon="house"),
        sac.StepsItem(title='Threat-modeling', subtitle='(10%)', icon="1-circle-fill"),
        sac.StepsItem(title='MCQ Benchmarks: Exploration', subtitle='(40%)', icon="2-circle-fill"),
        # Add other sections as needed
    ], size='small', return_index=True)

# List of section paths
section_paths = [
    "sec1_threat_modeling/sec10_home",
    "sec1_threat_modeling/sec11_threat_model",
    "sec1_threat_modeling/sec12_explore",
    # Add other section paths as needed
]

# Import and call the selected section function
if 0 <= CHAPTER < len(section_paths):
    try:
        section_func = import_section(section_paths[CHAPTER])
        section_func()
    except ImportError as e:
        st.error(f"Error importing section: {e}")
    except Exception as e:
        st.error(f"Error in section: {e}")
else:
    st.error("Invalid section selected")