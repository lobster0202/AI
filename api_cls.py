# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

from typing import Annotated

from fastapi import FastAPI, File, UploadFile

# STEP 2: Create an ImageClassifier object.
# 모델의 크기에 따라서 추론값이 달라질 수 있다. 무조건 크다고 잘 되는 것이 아니다.
base_options = python.BaseOptions(model_asset_path='efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=4)

# 추론기를 가져온다. 
classifier = vision.ImageClassifier.create_from_options(options)

app = FastAPI()


@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}


from PIL import Image
import io
import numpy as np 
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    contents = await file.read()
    
    # 1. create pil image from http file 

    # 2100
    # 1-1. convert http file to file
    # io 모듈을 가지고 컨텐츠를 감싸주면 http 파일을 파일로 변환 시킨 것
    
    # pil_img = Image.new('RGB', (60, 30), color = 'red')
    


    # STEP 3: Load the input image.
    pil_img = Image.open(io.BytesIO(contents))
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))
    
    # image = mp.Image.create_from_file("burger.jpg")

    # STEP 4: Classify the input image.
    classification_result = classifier.classify(image)

    # STEP 5: Process the classification result. In this case, visualize it.
    top_category = classification_result.classifications[0].categories[0]
    result = f"{top_category.category_name} ({top_category.score:.2f})"
    return {"result": result}

import cv2
import math

IMAGE_FILENAMES = ['burger.jpg', 'cat.jpg']

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# def resize_and_show(image):
#   h, w = image.shape[:2]
#   if h < w:
#     img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
#   else:
#     img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
# #   cv2_imshow(img)
#   cv2.imshow("test", img)
#   cv2.waitKey(0)


# # Preview the images.

# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)








