import pandas as pd
import requests
import os 
from PIL import Image, ImageOps
from io import BytesIO
import pillow_heif
pillow_heif.register_heif_opener()
from pdf2image import convert_from_bytes


folder = 'Photos/normal_photos'
os.makedirs(folder, exist_ok=True)

df = pd.read_excel('AI_Carpet_Roll_Photos.xlsx')
photo_links = df['Photos'].tolist()
photo_ids = df['SampleId'].tolist()

back_photos = []
for i in range(7600): 
  photo_split = photo_links[i].split(';')
  back_photo = photo_split[1]
  back_photos.append(back_photo)

def pad_to_square(img):
  width, height = img.size
  max_dim = max(width, height)
  # Create a new black image
  new_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
  # Paste the original image at the center
  new_img.paste(img, ((max_dim - width) // 2, (max_dim - height) // 2))
  return new_img

def download_images(back_photos, photo_ids, folder):
  for i in range(0, len(back_photos)):

    img = photo_ids[i] + '.jpg'

    if img not in os.listdir(folder):
      file_name = str(photo_ids[i])
      url = back_photos[i]
      response = requests.get(back_photos[i])


      if response.status_code == 200:

        if url.lower().endswith('.pdf'):
          images = convert_from_bytes(response.content)
          img = images[0].convert('RGB')
        else: 
          img = Image.open(BytesIO(response.content)).convert('RGB')
        img = pad_to_square(img)
        img.save(f'{folder}/{file_name}.jpg')
        print(f"Downloaded {file_name}.jpg to {folder}", i)
      else:
        print(f"Failed to download {file_name}.jpg from {url}")

#Checking if all the photos are in
download_images(back_photos, photo_ids, folder)

for i in photo_ids:
  img = i + '.jpg'
  if img not in os.listdir(folder):
    print(f"Missing {img}")