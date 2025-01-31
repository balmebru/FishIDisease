


import sys 
sys.path.append(r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Programmer\Git_Online\FishIDisease\Fishdisease")
from Fishdisease import FishIDisease




instance = FishIDisease()


image_dir= r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\BleedingVSBloodCirculation_root_dor\BleedingVSBloodCirculation"

segmentation_dir = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\BleedingVSBloodCirculation_root_dor\BleedingVSBloodCirculation_segmentation"
instance.view_segmentation_after_autoanno(image_dir=image_dir,segmentation_dir=segmentation_dir)