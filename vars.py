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

# Print the read configuration
for model in config['HandlingConfig']['Detectors']:
    print(model)
    print(config['DetectorConfig'][model])

for model in config['HandlingConfig']['Recognizers']:
    print(model)
    print(config['RecognizerConfig'][model])

# Initialize the variables of each model using the config file

