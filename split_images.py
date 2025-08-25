import splitfolders

input_folder ='Photos/classification_photos_all_real'

splitfolders.ratio(input_folder, output="classification_photos_divided", seed=42, ratio=(.7, .2, .1), group_prefix=None)

