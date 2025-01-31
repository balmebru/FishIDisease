
import sys 
sys.path.append(r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Programmer\Git_Online\FishIDisease\Fishdisease")
from Fishdisease import FishIDisease



instance = FishIDisease()
image_dir = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\Video_Exampel_Video_Fishes_low_framerate"
yolo_fish_segmenter_path = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Programmer\runs\segment\train12\weights\best.pt"
yolo_eye_detector_path= r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Programmer\runs\detect\train5\weights\best.pt"
save_path = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\Video_detection_output"
instance.create_overlay_fish_and_eye(image_dir, yolo_fish_segmenter_path, yolo_eye_detector_path, save_path, show=False)



# prediction_dir = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\EyeMalformations\EyeHealthy_detection_output"
# image_dir = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\EyeMalformations\EyeHealthy"
# output_dir= r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\EyeMalformations\Eyehealthy_comparison_plots"
# instance.create_comparison_plot_prediction_and_image(prediction_dir, image_dir, output_dir)