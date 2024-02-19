from tasks import Recognize
import time

while True:
    # Load the config
    from vars import read_json
    config = read_json('config.json')
    if(config['ImgConfig']['InputImgUrl']).startswith(('http://', 'https://')):
        known_names = Recognize(
            detector_name = config['HandlingConfig']['detectorName'],
            recognizer_name = config['HandlingConfig']['recognizerName'],
            img_url='https://i.postimg.cc/X72yyb43/img2.jpg'
        )
        print(known_names)

    else:
        time.sleep(1)
        print('Wait for 1 sec and try again')
