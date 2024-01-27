import time
from tasks import Recognize


while True:
    # Load the config
    from vars import read_json
    config = read_json('config.json')
    
    # Use the config
    known_names = Recognize(detector_name = config['HandlingConfig']['detectorName'], recognizer_name = config['HandlingConfig']['recognizerName'])
    print(known_names)

