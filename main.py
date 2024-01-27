import time
from tasks import Recognize


while True:
    # Load the config
    from vars import read_json
    config = read_json('config.json')
    # Use the config
    known_names = Recognize(detector_name = config['HandlingConfig']['detectorName'], recognizer_name = config['HandlingConfig']['recognizerName'])
    print(known_names)

    # Sleep for a while before re-reading the file
    time.sleep(1)  # Sleep for 10 seconds
