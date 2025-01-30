

from Segmenter import Segmenter

class FishIDisease:
    def __init__(self):
        pass

    def load_video(self, path: str):
        pass

    def run_pipeline(self):
        pass

    def export_results(self, output_path: str):
        pass

    def autoannotate_fish_directory(self, data, detection_model, sam_model, output_dir_fish):
        

        """
        Autoannotates the fishes in the given directory. And outputs the annotated images in the output directory.
        """
        segmenter_instance = Segmenter()
        segmenter_instance.autoannotate_fish(data=data,detection_model=detection_model,sam_model=sam_model,output_dir_fish=output_dir_fish)
        print("Done")

        
