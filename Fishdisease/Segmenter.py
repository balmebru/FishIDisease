

from ultralytics.data.annotator import auto_annotate


class Segmenter:
    def __init__(self):
        pass

    def detect_fish(self, image):
        pass

    def segment_fish(self, image):
        pass

    def extract_mask(self, segmented_image):
        pass



    def autoannotate_fish(self,data,detection_model,sam_model,output_dir_fish):

        """
        Takes a input directory with images and runs the autoannotate function on the images
        
        
        """
        auto_annotate(data=data,det_model=detection_model,sam_model=sam_model,output_dir=output_dir_fish,conf=0.7,iou=0.8)
    
        return