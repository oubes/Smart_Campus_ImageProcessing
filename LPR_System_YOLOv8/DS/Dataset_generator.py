# %% Get a list of all system fonts
# import matplotlib.font_manager, os
# system_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
# for font in system_fonts:
#     print(os.path.basename(font))

# %% define:
chars = {
    'أ': 'alf', 'ب': 'baa', 'ت': 'taa', 'ج': 'jeem',
    'د': 'daal', 'ر': 'raa', 'س': 'seen', 'ص': 'saad',
    'ف': 'faa', 'ق': 'qaaf', 'ل': 'laam', 'م': 'meem',
    'ن': 'noon', 'ه': 'haa', 'و': 'waw', 'ي': 'yaa',
    '١': '1', '٢': '2', '٣': '3', '٤': '4', '٥': '5',
    '٦': '6', '٧': '7', '٨': '8', '٩': '9', '٠': '0'
}
char_dirs = chars.values()
letters = chars.keys()

# %%
from keras.preprocessing.image import ImageDataGenerator
import cv2, os, shutil
from PIL import Image, ImageDraw, ImageFont 

def remake_dirs(main_dir):
    shutil.rmtree(main_dir, ignore_errors=True)
    for char_dir in char_dirs:
        os.makedirs(os.path.join(main_dir, char_dir), exist_ok=False)

def letters_gen(main_dir, img_name, letter_size, margin, font_name):
    for letter, dir in zip(letters, char_dirs):
        font = ImageFont.truetype(font_name, letter_size)
        img = Image.new('RGB', (letter_size+margin[1], letter_size+margin[0]), color = (255, 255, 255))
        draw = ImageDraw.Draw(img)
        text_width, text_height = draw.textsize(letter, font)

        x = (img.width - text_width) / 2
        y = (img.height - text_height) / 4

        draw.text((x, y), letter, fill=(0,0,0), font=font)
        img = img.convert('L') # Convert to grayscale
        img.save(f'{main_dir}/{dir}/{img_name}.png')

def multi_size_font_gen(main_dir, sizes, fonts, margin):
    for size in sizes:
        for font in fonts:
            letters_gen(
                main_dir = main_dir,
                img_name = f'img_size[{size}]_font[{font}]',
                letter_size = size,
                margin = margin,
                font_name= font
            )

def Dataset_Gen(main_dir, aug_imgs_num, sizes, fonts, margin):
    multi_size_font_gen(main_dir, sizes, fonts, margin)
    for letter_dir in os.listdir(main_dir):
        for img in os.listdir(os.path.join(main_dir, letter_dir)):
            imagegen = ImageDataGenerator(
                rotation_range=15,
                zoom_range=0.1,
                shear_range=0.15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=False,
                vertical_flip=False,
                brightness_range=(0.5, 1.5),
                fill_mode='nearest',
                rescale=1./255,
                validation_split=0.2
            )

            image = cv2.imread(os.path.join(main_dir, letter_dir, img))
            image = image.reshape((1,) + image.shape)

            imagegen.fit(image)
            img_gen = imagegen.flow(
                image,
                batch_size=1,
                save_to_dir=os.path.join(main_dir, letter_dir),
                save_prefix="aug",
                save_format="png"
            )

            if aug_imgs_num != 0:
                for i, batch in enumerate(img_gen):
                    if i+1 >= aug_imgs_num:
                        break
# %%
main_dir = 'training_ds'
remake_dirs(main_dir)
Dataset_Gen(
    main_dir=main_dir,
    aug_imgs_num = 100,
    sizes = [50], # list of sizes
    fonts = ['majallab'], # list of fonts
    margin = [12, 4] # [height margin, width margin]
)

# %%
