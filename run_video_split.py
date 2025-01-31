import sys 
sys.path.append(r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Programmer\Git_Online\FishIDisease\Fishdisease")
from Fishdisease import FishIDisease



instance = FishIDisease()

video_path = r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\Example_Video_Fishes.MP4"
output_dir =r"C:\Users\bruno\OneDrive\Desktop\Bruno\Brunos_capitol\Dep.IT\Data\FIshID_data\Video_Exampel_Video_Fishes_low_framerate"
instance.video_splitter(video_path=video_path, output_dir=output_dir, frame_rate=1)