from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy 

model = YOLO('segmodel.pt')  

path = 'Photos/normal_photos/AS000811.jpg'

img = Image.open(path)
img.show()

#Segmentnation
results = model.predict(path)
result = results[0]
masks = result.masks

mask1 = masks[0]
mask = mask1.data[0].cpu().numpy()
polygon = mask1.xy[0]

#Drawa polygon on original imag
draw = ImageDraw.Draw(img)
draw.polygon(polygon, outline="red", fill=None)
img.show()
