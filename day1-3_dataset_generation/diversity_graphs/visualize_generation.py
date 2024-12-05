#%%
import plotly.express as px
from utils import import_json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import plotly.io as pio
from collections import defaultdict
from typing import Optional

#%%
data_path = "scores/strategic/2c-generated-final.json"
dataset = import_json(data_path)


def partition_and_count(data, category: str, subcategory_1: Optional[str] = None):
    from collections import defaultdict
    
    partitioned_data = defaultdict(list)    
    results = defaultdict(int)

    for item in data:
        try: partitioned_data[item[category]].append(item)
        except: partitioned_data[item[category][0]].append(item)
    
    for key, value in partitioned_data.items():
        results[key] = len(value)

    if subcategory_1 is not None:
        for key, value in partitioned_data.items():
            subpartitioned_data = defaultdict(list)
            for item in value:
                try: subpartitioned_data[item[subcategory_1]].append(item)
                except: subpartitioned_data[item[subcategory_1][0]].append(item)
            results[key] = subpartitioned_data


    return dict(partitioned_data)

partition_key = 'strategic_answer'
partitioned_data = partition_list_of_dicts(dataset[:70], partition_key)

for key, value in partitioned_data.items():
    partition_key = 'answer_matching_behavior'
    partitioned_data_2 = partition_list_of_dicts(value, partition_key)
    print(key, partitioned_data_2.keys())


# %%
