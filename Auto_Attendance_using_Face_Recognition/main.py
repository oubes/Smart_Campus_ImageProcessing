from tensorflow import config
from Recognizers import DLIB
#from preprocessing import save_augmented_imaged


#while True:
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
from vars import read_json
config = read_json('config.json')
model_class = DLIB.fr_dlib_model(encoded_dict={})
encoded_dict = model_class.add_labeled_encoded_entry("https://scontent.fcai20-4.fna.fbcdn.net/v/t39.30808-6/425368959_7136002173160865_1422995269039308061_n.jpg?_nc_cat=104&ccb=1-7&_nc_sid=efb6e6&_nc_eui2=AeFe4R9WloIImw4jduh2aLkd51m8ETod8JrnWbwROh3wmiw4tz4OvskNbwrleE3n1P01PiggRAeZOjA5faqggc0Y&_nc_ohc=sZKfoNWa03wAX-d-LzH&_nc_ht=scontent.fcai20-4.fna&oh=00_AfDhi3GE3X0G_0j8Ql_SpyU7FhCdgGQLEF6mh_rvqHQlkg&oe=65E6DCDD", "180000")
known_names = model_class.Recognize(unlabeled_img_url='https://i.postimg.cc/X72yyb43/img2.jpg')
print(known_names)

# save_augmented_imaged('https://encrypted-tbn2.gstatic.com/licensed-image?q=tbn:ANd9GcSUD_78qicz9WnjawNqFryeVP0tfsPUZSuI_ZSb8UQK9A9kwevvKh4itte4C96SRqgkXM7e0te_flqd3_8')
