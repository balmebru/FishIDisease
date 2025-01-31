import sys 
sys.path.append(r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Programmer\Git_Online\FishIDisease\Fishdisease")
from Fishdisease import FishIDisease


image_path = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\EyeMalformations\EyeHealthy\ZHAW Biocam_00_20240325103527.jpg"
sam_model_path = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Programmer\sam_vit_h_4b8939.pth"
yolo_model_path = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Programmer\runs\segment\train\weights\best.pt"

instance = FishIDisease()
instance.autoannotate_fish_eyes(image_path=image_path, sam_model_path=sam_model_path, yolo_model_path=yolo_model_path, show=True)