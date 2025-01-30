import sys 
sys.path.append(r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Programmer\Git_Online\FishIDisease\Fishdisease")
from Fishdisease import FishIDisease




instance = FishIDisease()


image_dir =r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\RunYoloSegment_v_1\train\images"
mask_dir = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\RunYoloSegment_v_1\train\labels"

instance.validate_and_sync_directories(dir1=image_dir, dir2=mask_dir)



