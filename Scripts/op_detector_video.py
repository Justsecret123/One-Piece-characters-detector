# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:18:27 2021
@author: Ibrah
"""

import argparse
import time
import cv2
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf 

def load_model():
    """Loads the TFLite model"""

    print("Loading the model...\n-----------------------")

    # Intialize the timer
    timer = time.time()

    # Load the interpreter (model)
    model = tf.saved_model.load(MODEL_PATH)
    
    # Calculate and display the elapsed time
    timer = round(time.time() - timer, 2)
    print(f"\n\nModel loaded in: {timer}s\n")

    return model

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
    detections = {key:value[0, :num_detections].numpy() for key,value in detections.items()} 
    # Convert the detection classes into te proper format (int)
    detections["detection_classes"] = detections["detection_classes"].astype(np.uint8) 
    
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
        min_score_thresh=THRESHOLD,
        agnostic_mode=False, 
        line_thickness=THICKNESS
    )
    
    return image
    

def main():
    """Main Loop"""
    
    # Read the video
    video = cv2.VideoCapture(INPUT_VIDEO)
    
    # Get the image size 
    print("\nRetrieving the parameters...\n--------------------------")
    width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup the video writer
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc("M","J","P","G"), FPS, (width,height))
    
    print(f"Video size: {(width,height)} | Frame count : {frame_count}")
    
    i = 0
    
    print("\nStarting the detection...\n--------------------------")
    
    # Loop through the frames
    while video.isOpened():
        
        # Get the current frame
        ret, frame = video.read()
        
        # Process if not empty
        if frame is not None:
            
            # Display the progress
            print(f"Processing frame {i}...")
            
            # Start the timer
            timer = time.time()
            
            # Copy the frame
            image = frame.copy()
            
            # Get the detections boxes, classes and scores
            detections = get_detections(frame)
            
            # Write the visualizations on the frame's copy
            output_image = write_visualizations(image,
                                                detections["detection_boxes"], 
                                                detections["detection_classes"],
                                                detections["detection_scores"]
                                                )
            
            # Write the frame with the detections to the output file
            out.write(output_image)
            
            # Get the elapsed time
            timer = round(time.time() - timer, 2)
            # Display the elapsed time
            print(f"Elapsed time for inference: {timer}s")
            
            # Progress to the next frame
            i+=1
            
    # Release the video object
    video.release()
    # Release the writer
    out.release()
    
    # Close all the windows
    cv2.destroyAllWindows()

def create_parser(): 
    """Creates a parser for the command line runner"""
    
    # Create the parser
    parser = argparse.ArgumentParser(description="Run inferences on a video")
    
    # Add arguments
    
    # Model path 
    parser.add_argument("-model", help="Model path")
    
    # Label map path
    parser.add_argument("-labelmap", default="tf_label_map.pbtxt", help="Labelmap path")
    
    # Input video
    parser.add_argument("-input", help="Input video")
    
    # Output 
    parser.add_argument("-output", help="Output")
    
    # Threshold
    parser.add_argument("-thresh", type=float, default=.3, help="Minimum threshold")
    
    # Frame rate
    parser.add_argument("-fps", type=int, default=24, help="Frame rate in frames/sec")
    
    # Thickness
    parser.add_argument("-thickness", type=int, default=4, help="Line thickness")
    
    return parser

if __name__ == "__main__":
    
    # Create the parser
    PARSER = create_parser()
    # Parse the command line arguments
    ARGS = PARSER.parse_args()
    # Get the values as a dict(key,value)
    VARIABLES = vars(ARGS)
    
    print(f"\n\nVars:\n-----------------------\n{VARIABLES}\n\n")
    
    # Setup globals
    MODEL_PATH = VARIABLES["model"]
    LABELMAP_PATH = VARIABLES["labelmap"]
    INPUT_VIDEO = VARIABLES["input"]
    OUTPUT_VIDEO = VARIABLES["output"]
    FPS = VARIABLES["fps"]
    THRESHOLD = VARIABLES["thresh"]
    THICKNESS = VARIABLES["thickness"]
    
    # Create the category index from the label map file
    CATEGORY_INDEX = label_map_util.create_category_index_from_labelmap(LABELMAP_PATH)
    
    # Load the trained model
    print("\nLoading the model...\n--------------------------")
    MODEL = tf.saved_model.load(MODEL_PATH)
    print("\nDone.")
    
    main()
    