# STEP 1
from fastapi import FastAPI, File, UploadFile

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision
import numpy as np 
from fastapi import FastAPI, File, UploadFile

# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path='efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=4)
classifier = vision.ImageClassifier.create_from_options(options)

app = FastAPI()

from PIL import Image
import io
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile) : 
    
    contents = await file.read()

    # STEP 3: Load the input image.
    pil_img = Image.open(io.BytesIO(contents))
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4 : Detect objects in the input image
    detection_result = detector.detect(image)

    # STEP 5 : 찾은 객체의 종류와 종류 갯수를 출력하시오
    result_dict = {}

    for detection in detection_result.detections : 

        category = detection.categories[0].category_name

        if category not in result_dict:
            result_dict[category] = 1
        else :
            result_dict[category] += 1

    return {"result" : result_dict}
    