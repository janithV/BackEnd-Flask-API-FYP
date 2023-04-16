from leafconfig import load_model_weights, get_resized_image
from modelTwoDetect import get_unhealthy_area
import urllib.request
import numpy as np
import cv2
import math
from flask import Flask , request
from flask_restful import Api, Resource, reqparse, abort

app = Flask(__name__)
api = Api(app)

model_1 = load_model_weights(1)
model_2 = load_model_weights(2)

class DeepLModelAPI(Resource):

    def get(self, url):

        total_area = 0
        total_unhealthy_area_final = 0

        res = urllib.request.urlopen(url)
        img_array = np.array(bytearray(res.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)

        resized_image = get_resized_image(img)
        
        results = model_1.detect([resized_image], verbose=1)
        r = results[0]

        for x in range(len(r['class_ids'])):

            if r['class_ids'][x] == 1 :
                print(f"detection {x} is skipped")
                continue

            #get mask
            masks = r['masks']
            mask = masks[:, :, x]
            
            # Calculate the resizing factor
            resize_factor = mask.shape[0] / resized_image.shape[0]
            


            # Convert the mask to binary format
            area_mask = np.where(mask > 0, 1, 0)

              
            # Calculate the average area of each non-zero pixel
            non_zero_pixels = np.count_nonzero(area_mask)
            non_zero_area = cv2.countNonZero(area_mask)
            average_pixel_area = non_zero_area / non_zero_pixels

            # Compute the total area of the mask
            total_area = total_area + (non_zero_area * average_pixel_area)

            new_arr = np.zeros((256, 256, 3))
            new_arr[:, :, 1] = mask
            img_rgb = np.stack((mask,) * 3, axis=-1)

            new = resized_image * img_rgb
            #rgb_img = cv2.cvtColor(new, cv2.COLOR_BGR2RGB)
            
            unhealthy_area = get_unhealthy_area(new, model_2, resize_factor)
            
            total_unhealthy_area_final = total_unhealthy_area_final + unhealthy_area


        value = total_unhealthy_area_final / total_area
        percentage = (value * 100) 
        percentage = math.ceil(percentage)
        
        return {"data" : f'{percentage}%'}


api.add_resource(DeepLModelAPI, "/<path:url>")

if __name__ == "__main__":
    app.run(debug=True)