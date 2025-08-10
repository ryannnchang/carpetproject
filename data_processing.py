from ultralytics.data.converter import convert_coco

convert_coco('Photos/segementations_photos_realcoco/train', use_segments=True)
convert_coco('Photos/segementations_photos_realcoco/val', use_segments=True)
convert_coco('Photos/segementations_photos_realcoco/test', use_segments=True)

print(f"Converted JSON")
