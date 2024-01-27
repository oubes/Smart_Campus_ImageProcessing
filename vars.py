# Initialize the variables of each model using the config file
import json

def read_json(file_name):
    with open(file_name) as f:
        return json.load(f)

def write_json(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

config = {}

config = read_json('config.json')

detectror_config = config["DetectorConfig"]
recognizer_config = config["RecognizerConfig"]
handling_config = config["HandlingConfig"]
image_config = config["ImgConfig"]

