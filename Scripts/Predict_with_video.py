# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:18:27 2021

@author: Ibrah
"""

import cv2
import tensorflow as tf 
import numpy as np 

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


# Get the detections boxes, scores and classes from the input frame
def get_detections(frame):
    
    # Convert the input array into a tensor
    input_tensor = np.expand_dims(frame,0).astype(np.uint8) #Create a batch then convert the input image values into unsigned ints ranging from 0 to 255
    
    # Run the inference
    detections = model(input_tensor)
    
    # Get the results 
    
    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()} # Create a dict from the results
    
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64) # Convert the detection classes into the proper format (int)
    
    detections["num_detections"] = num_detections
    
    return detections

# Write the detections results on a frame
def write_visualizations(image, boxes, classes, scores, category_index):
    
    # Visualize the results then draw the output boxes and classes on images
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image, 
        boxes, 
        classes, 
        scores, 
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=30,
        min_score_thresh=.3,
        agnostic_mode=False, 
        line_thickness=5
    )
    
    return image
    

def main(input_video, model, category_index):
    
    #Read the video
    video = cv2.VideoCapture(input_video)
    
    
    #Get the image size 
    width, height = int(video.get(3)), int(video.get(4))
    
    # Output video 
    out = cv2.VideoWriter("result.avi", cv2.VideoWriter_fourcc("M","J","P","G"),30,(width,height))
    
    print(f"Video size: {(width,height)}")
    
    
    while (video.isOpened()):
        
        ret, frame = video.read()
        
        if ret == True:
            image = frame.copy()
            
            # Get the detections boxes, classes and scores
            detections = get_detections(frame)
            
            # Write the visualizations on the image's copy
            output_image = write_visualizations(image,
                                                detections["detection_boxes"], 
                                                detections["detection_classes"],
                                                detections["detection_scores"],
                                                category_index)
            
            # Write the frame with the detections to the output file
            out.write(output_image)
            
    
    #Release the video object
    video.release()
    out.release()
    
    #Close all the windows
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    
    input_video = "path_to_the_video"
    model_path = "path_to_the_model"
    
    # Create the category index from the label map file
    category_index = label_map_util.create_category_index_from_labelmap("tf_label_map.pbtxt",
                                                                        use_display_name=True)
    
    # Load the trained model
    model = tf.saved_model.load(model_path)
    
    
    main(input_video, model, category_index)
