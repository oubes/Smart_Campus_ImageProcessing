import json

def read_json(file_name):
    with open(file_name) as f:
        return json.load(f)

def write_json(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

config = read_json('config.json')

detector    = config["HandlingConfig"]["detectorName"]
recognizer  = config["HandlingConfig"]["recognizerName"]

detector_config     = config["DetectorConfig"][detector]
recognizer_config   = config["RecognizerConfig"][recognizer]

# https://i.postimg.cc/X72yyb43/img2.jpg
# https://i.postimg.cc/rp4dbRWF/students1.jpg
# https://i.postimg.cc/mkmhMYxV/img4.jpg
# https://i.postimg.cc/6qWyyHPy/students2.jpg
# https://i.postimg.cc/C1HRMym3/students3.jpg
