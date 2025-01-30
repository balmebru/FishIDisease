
import sys 
sys.path.append(r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Programmer\Git_Online\FishIDisease\Fishdisease")
from Fishdisease import FishIDisease



fishid_instance = FishIDisease()

yolo_path = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Programmer\Git_Online\FishIDisease\FishCheckpoint_01.pt"
segmenter_path = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Programmer\sam2.1_s.pt"
data_path = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\EyeMalformations\EyeHealthy"
output_dir = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\EyeMalformations\EyeHealthy_segmenter_mask"

fishid_instance.autoannotate_fish_directory(
    data=data_path,
    detection_model=yolo_path,
    sam_model=segmenter_path,
    output_dir_fish=output_dir

)

