


import sys 
sys.path.append(r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Programmer\Git_Online\FishIDisease\Fishdisease")
from Fishdisease import FishIDisease




instance = FishIDisease()


image_dir= r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\EyeMalformations\EyeHealthy"

segmentation_dir = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\EyeMalformations\EyeHealthy_segmenter_mask"
instance.view_segmentation_after_autoanno(image_dir=image_dir,segmentation_dir=segmentation_dir)