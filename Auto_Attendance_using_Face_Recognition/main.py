# from Recognizers import DLIB
# from preprocessing import save_augmented_imaged


# while True:
#     # Load the config
#     from vars import read_json
#     config = read_json('config.json')
#     if(config['ImgConfig']['InputImgUrl']).startswith(('http://', 'https://')):
#         known_names = Recognize(
#                 detector_name = config['HandlingConfig']['detectorName'],
#                 recognizer_name = config['HandlingConfig']['recognizerName'],
#                 img_url='https://utfs.io/f/3fbe0673-b594-48d9-8212-2773a1403e4b-1rhq9e.jpg'
#                 )
#         print(known_names)
#
#     else:
#         time.sleep(1)
#         print('Wait for 1 sec and try again')
# save_augmented_imaged('https://encrypted-tbn2.gstatic.com/licensed-image?q=tbn:ANd9GcSUD_78qicz9WnjawNqFryeVP0tfsPUZSuI_ZSb8UQK9A9kwevvKh4itte4C96SRqgkXM7e0te_flqd3_8')
