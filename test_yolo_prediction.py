

import sys 
sys.path.append(r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Programmer\Git_Online\FishIDisease\Fishdisease")
from Fishdisease import FishIDisease




instance= FishIDisease()

image_path = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\EyeMalformations\EyeHealthy\ZHAW Biocam_00_20240325103527.jpg"
yolo_model_path = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Programmer\runs\segment\train12\weights\best.pt"



instance.predict_image(image_path=image_path, yolo_model_path=yolo_model_path, show=True)