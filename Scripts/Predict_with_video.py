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



def get_detections(frame):
    """Get the detections boxes, scores and classes from the input frame"""
    
    # Convert the input array into a tensor
    # Create a batch then convert the input image values into unsigned ints ranging from 0 to 255
    input_tensor = np.expand_dims(frame,0).astype(np.uint8) 
    
    # Run the inference
    detections = MODEL(input_tensor)
    
    # Get the results 
    num_detections = int(detections.pop("num_detections"))
    # Create a dict from the results
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()} 
    # Convert the detection classes into te proper format (int)
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64) 
    
    detections["num_detections"] = num_detections
    
    return detections


def write_visualizations(image, boxes, classes, scores):
    """Write the detections results on a frame"""
    
    # Visualize the results then draw the output boxes and classes on images
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image, 
        boxes, 
        classes, 
        scores, 
        CATEGORY_INDEX,
        use_normalized_coordinates=True,
        max_boxes_to_draw=30,
        min_score_thresh=.3,
        agnostic_mode=False, 
        line_thickness=5
    )
    
    return image
    

def main():
    """Main Loop"""
    
    # Read the video
    video = cv2.VideoCapture(INPUT_VIDEO)
    
    # Get the image size 
    print("\nRetrieving the parameters...\n--------------------------")
    width, height = int(video.get(3)), int(video.get(4))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video 
    out = cv2.VideoWriter("result.avi", cv2.VideoWriter_fourcc("M","J","P","G"),30,(width,height))
    
    print(f"Video size: {(width,height)} | Frame count : {frame_count}")
    
    i = 0
    
    print("\nStarting the detection...\n--------------------------")
    
    while video.isOpened():
        
        ret, frame = video.read()
        
        if ret == True:
            print(f"Processing frame {i}...")
            image = frame.copy()
            
            # Get the detections boxes, classes and scores
            detections = get_detections(frame)
            
            # Write the visualizations on the image's copy
            output_image = write_visualizations(image,
                                                detections["detection_boxes"], 
                                                detections["detection_classes"],
                                                detections["detection_scores"]
                                                )
            
            # Write the frame with the detections to the output file
            out.write(output_image)
            i+=1
            
    # Release the video object
    video.release()
    out.release()
    
    # Close all the windows
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    
    INPUT_VIDEO = "Zoro vs Kamazou.mp4"
    MODEL_PATH = "../OP_characters_detector/4"
    
    # Create the category index from the label map file
    CATEGORY_INDEX = label_map_util.create_category_index_from_labelmap("tf_label_map.pbtxt")
    
    # Load the trained model
    print("\nLoading the model...\n--------------------------")
    MODEL = tf.saved_model.load(MODEL_PATH)
    print("\nDone.")
    
    main()
    