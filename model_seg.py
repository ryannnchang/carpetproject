from ultralytics import YOLO
from PIL import Image, ImageDraw, Image
import numpy as np
from typing import Tuple, Optional
import numpy 
import os

model = YOLO('segmodel.pt')  
input_dir = 'Photos/normal_photos'
output_dir = 'Photos/processed_photos'

def get_2d_array(results, path): #returns a 2D array of the mask
  try: 
    result = results[0] #get the first result
    masks = result.masks #get the masks from the result

    #Extract the first mask and polygon
    mask1 = masks[0]  # get the first mask
    mask = mask1.data[0].cpu().numpy() 
    mask_bin = (mask > 0.5).astype(numpy.uint8)  # 1 inside, 0 outside # 
    return mask_bin
  except Exception as e:
    print(f"Error processing {path}: {e}")
    return np.ones((640, 640), dtype=np.uint8)

def maximalSquareWithCoords_np(arr: np.ndarray) -> Tuple[int, Optional[Tuple[int,int,int,int]], int]:
    """
    Find the largest square of 1s in a binary numpy array.
    
    Returns:
        area (int): area of the largest square
        coords (tuple): (top_row, left_col, bottom_row, right_col) or None if no 1s
        side (int): side length of the square
    """
    if arr.size == 0:
        return 0, None, 0
    
    R, C = arr.shape
    dp = np.zeros((R+1, C+1), dtype=int)

    max_side = 0
    br_r = br_c = -1

    for r in range(R):
        for c in range(C):
            if arr[r, c] == 1:
                dp[r+1, c+1] = 1 + min(dp[r, c+1], dp[r+1, c], dp[r, c])
                if dp[r+1, c+1] > max_side:
                    max_side = dp[r+1, c+1]
                    br_r, br_c = r, c

    if max_side == 0:
        return 0, None, 0

    tl_r = br_r - max_side + 1
    tl_c = br_c - max_side + 1
    coords = (tl_r, tl_c, br_r, br_c)
    area = max_side * max_side

    return area, coords, max_side

def crop_image(coords, path, name, save_dir=output_dir):
  img = Image.open(path)
  #resizing image to 640x640
  img_resized = img.resize((640, 640), Image.Resampling.LANCZOS)  

  #Finding the coordinates of the square and recropping the image
  tl_r, tl_c, br_r, br_c = coords
  cropped_img = img_resized.crop((tl_c, tl_r, br_c + 1, br_r + 1))
  cropped_img.save(f"{save_dir}/{name}")
  

def main():
  photos = os.listdir(input_dir)
  total = len(photos)
  for i in range(203, len(photos)):
     if photos[i].endswith('.jpg') or photos[i].endswith('.png'):
        path = input_dir + '/' + photos[i]
        results = model.predict(path)
        mask_bin = get_2d_array(results, path)
        area, coords, side = maximalSquareWithCoords_np(mask_bin)
        crop_image(coords, path, save_dir=output_dir, name=photos[i])
        print(f"Process: {photos[i]} | Progress: {i}/{total} images processed.")
main()





