import numpy as np
import cv2

def get_unhealthy_area(instance, model, resize_factor):
    total_unhealthy_area = 0
        
    result = model.detect([instance], verbose=1)
    rx = result[0]
    print("num_detections model 2 : ",len(rx['class_ids']))
        
    for x in range(len(rx['class_ids'])):
        #get mask
        masks = rx['masks']
        mask = masks[:, :, x]
            
        # Convert the mask to binary format
        mask = np.where(mask > 0, 1, 0)
    #   print(mask)

        non_zero_pixels = np.count_nonzero(mask)
        non_zero_area = cv2.countNonZero(mask)
        average_pixel_area = non_zero_area / non_zero_pixels / (resize_factor ** 2)

        # Compute the total area of the mask
        total_unhealthy_area = total_unhealthy_area + (non_zero_area * average_pixel_area)
            
        
    return total_unhealthy_area