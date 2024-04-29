import requests

def download_img(img_url: str, img_name: str):
    response = requests.get(img_url)
    with open(img_name + ".jpg", "wb") as f:
        f.write(response.content)
    return img_name + ".jpg"
