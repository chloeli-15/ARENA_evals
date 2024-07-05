import json


def get_openai_api_key():
    return json.loads(open('secrets.json').read())['OPENAI_API_KEY']

def import_json_as_string(file_path):
    with open(file_path, 'r') as f:
        return f.read()
    

def import_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)