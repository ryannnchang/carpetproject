import os 
import pandas as pd
import numpy as np 
from collections import Counter
import shutil

source_path = 'Photos/processed_photos_filtered_copy'
output_dir = 'Photos/classification_photos/train'

df = pd.read_excel('AI_Carpet_Roll_Photos.xlsx')
backing = df['Backing']
label = df['SampleId']
construction = df['Construction']

construction_count = Counter(construction)
print("Construction counts:", construction_count)

#Creating a dictionary to map backing to label
key_pairs = {}
for i in range(len(backing)):
  key_pairs[label[i]] = backing[i]

#Sorting Photos based on backing
folders = os.listdir(source_path)
photos = []

for i in folders:
  if i.endswith('.jpg') or i.endswith('.png'):
    photos.append(i)

# for i in photos:
#   sample_id = os.path.splitext(i)[0]

#   if sample_id in key_pairs:
#     backing = key_pairs[sample_id]

#     if not os.path.exists(f'{output_dir}/{backing}'):
#       os.makedirs(f'{output_dir}/{backing}')

#     shutil.move(f'{source_path}/{i}', f'{output_dir}/{backing}')
#     print(f"Moved {i} to {output_dir}/{backing}")
#   else:
#     print(f"No backing found for {i}, skipping.")

