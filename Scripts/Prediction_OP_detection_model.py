# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 19:19:22 2021

@author: Ibrah
"""

import io
import json
import cv2

import numpy as np
import requests
import matplotlib.pyplot as plt

from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


# Image URL Location
IMAGE_URL = "https://wallpapercave.com/wp/wp4865629.jpg"

# API endpoint
SERVER_URL = "http://localhost:8501/v1/models/OP_characters_detector:predict"

def main():
    """Main loop"""
    
    print("\nDownloading the image...")
    
    # Download the image
    dl_request = requests.get(IMAGE_URL, stream=True)
    dl_request.raise_for_status()
    

    # Compose a JSON Predict request (send the image tensor)
    jpeg_rgb = Image.open(io.BytesIO(dl_request.content))
    
    print("\nDone.")

    # Normalize and batchify the image
    inputs = np.array(jpeg_rgb)
    inputs = cv2.cvtColor(inputs, cv2.COLOR_BGRA2BGR) # Convert the image into a 3-channel image
    image_np_with_detections = inputs.copy().astype(np.uint8)

    # Create a batch then convert the input image values into unsigned ints ranging from 0 to 255
    input_tensor = np.expand_dims(inputs, 0).astype(np.uint8)
    serialized_image = input_tensor.tolist() # Convert the image into a serializable object
    predict_request = json.dumps({"instances": serialized_image })
    
    # Send a request to warm-up the model
    for _ in range(1):
        response = requests.post(SERVER_URL, data=predict_request)
        response.raise_for_status()
        
    detections = []
    total_time = 0
    
    # Send the request
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()
    detections = response.json()["predictions"][0]
    total_time += response.elapsed.total_seconds()
        
    # Create the label map
    category_index = label_map_util.create_category_index_from_labelmap("tf_label_map.pbtxt")
    
    print("\nDrawing the results...")
    
    # Visualize the results
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          np.array(detections["detection_boxes"]),
          np.array(detections["detection_classes"]).astype(np.uint8),
          np.array(detections["detection_scores"]),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=30,
          min_score_thresh=.6,
          agnostic_mode=False, 
          line_thickness=10)

    # Create the figure and save the file
    plt.figure()
    plt.axis("off")
    plt.title("Results")
    plt.imshow(image_np_with_detections)
    plt.savefig("result.png")
    
    print(f"Done. Total time: {total_time}s")    


if __name__ == '__main__':
    main()
    