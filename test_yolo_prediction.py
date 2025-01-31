

import sys 
sys.path.append(r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Programmer\Git_Online\FishIDisease\Fishdisease")
from Fishdisease import FishIDisease




instance= FishIDisease()

image_dir_path = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\Video_Exampel_Video_Fishes_low_framerate"
yolo_model_path = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Programmer\Git_Online\runs\detect\train2\weights\best.pt"
save_path = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\Video_detection_output_v2"


instance.predict_directory(image_dir=image_dir_path, yolo_model_path=yolo_model_path,save_path=save_path, show=False)

