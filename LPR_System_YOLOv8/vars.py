import json

def read_json(file_name):
    with open(file_name, encoding='utf-8') as f:
        return json.load(f)

def write_json(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

config = read_json('config.json')
