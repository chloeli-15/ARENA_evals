import json
import importlib
import config
import openai
from openai import OpenAI
import os

def save_json(file_path, data, do_not_overwrite=False):
    if do_not_overwrite:
        try: 
            with open(file_path, "r") as f:
                existing_data = json.load(f)
        except FileNotFoundError:
                existing_data = []
        existing_data.extend(data)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)


def get_openai_api_key():
    return json.loads(open('secrets.json').read())['OPENAI_API_KEY']

def import_json_as_string(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def import_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def reload_config():
    importlib.reload(config)
    return config

def establish_client_OpenAI():
    '''
    Returns the OpenAI client which can be used for accessing models
    '''
    return OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
def establish_client_anthropic():
    '''
    Returns the anthropic client which can be used for accessing models
    '''
    #Returns anthropic client - Do later.
    pass

#Note for when we make ARENA content: emphasise that API keys need to be a secret. This is annoying but can be set in the terminal and teaches good practice for API key usage going forward (I didn't know about this). You can set them in cmd on windows or BASH on mac. https://platform.openai.com/docs/quickstart for info on how to do this.
