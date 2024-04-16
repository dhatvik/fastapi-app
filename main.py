from base64 import b64encode
import json
import os
from fastapi import FastAPI, File, UploadFile,Response
from PIL import Image
import io
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
from inference import get_model
import supervision as sv
import cv2
from fastapi.responses import FileResponse
from typing import Optional
import shutil
import tempfile
from inference.core.interfaces.stream.sinks import VideoFileSink
from inference import InferencePipeline
import firebase_admin
from firebase_admin import credentials, storage
from roboflow import Roboflow

app = FastAPI()

@app.post("/predict")
async def predict(image: UploadFile):
    # load the image
    img = Image.open(image.file)
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # load a pre-trained yolov8n model
    model = get_model(model_id="merged-project-ozbro/1",api_key="INc4g2WbMuzVOyCAXNVp")

    # run inference on our chosen image
    results = model.infer(img_array)

    # load the results into the supervision Detections api
    detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))
    # convert the numpy integer object to a Python list
    class_list = detections.data.get('class_name').tolist()
    if class_list is None:
      class_list = [] 
    # create supervision annotators
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=img_array, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    # convert the annotated image to PIL Image
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    annotated_image = Image.fromarray(annotated_image)

    # convert the PIL Image to bytes
    buffer = io.BytesIO()
    annotated_image.save(buffer, format='JPEG')
    buffer.seek(0)

    
    return StreamingResponse(
      content=buffer,
      media_type="image/jpeg",
      headers={
           'X-Class-List': json.dumps(class_list)  # Add class ID as a custom header
      }
  )


# Initialize the inference pipeline
model_id = "cocoa-fxvcr/3"
output_file_name = "outputfile.mp4"

@app.get("/")
def index():
    return {"details": "Cocoa-vision"}

import subprocess
 
@app.post("/process_video")
async def process_video(video_file: UploadFile = File(...)):
    # Save the uploaded video to a local file
    with open("uploaded_video.mp4", "wb") as buffer:
        buffer.write(await video_file.read())

    # Initialize VideoFileSink
    video_sink = VideoFileSink.init(video_file_name="outputfile.avi")

    # Initialize and start the inference pipeline
    pipeline = InferencePipeline.init(
        model_id=model_id,
        video_reference="uploaded_video.mp4",
        on_prediction=video_sink.on_prediction
    )
    pipeline.start()
    pipeline.join()
    video_sink.release()

    # Convert outputfile.avi to outputfile.mp4
    conversion_command = ["ffmpeg", "-y", "-i", "outputfile.avi", "outputfile.mp4"]
    subprocess.run(conversion_command)

    # Read the processed video file
    with open("outputfile.mp4", "rb") as video_file:
        video_data = video_file.read()

    # Fetch predictions from Roboflow
    rf = Roboflow(api_key="INc4g2WbMuzVOyCAXNVp")  # Replace with your Roboflow API key
    project = rf.workspace().project("blackpod_cocoa")
    model = project.version("2").model

    job_id, signed_url, expire_time = model.predict_video(
        "outputfile.mp4",
        fps=1,
        prediction_type="batch-video",
    )

    results = model.poll_until_video_results(job_id)

    # Clean up local files
    os.remove("outputfile.avi")
    os.remove("outputfile.mp4")
    os.remove("uploaded_video.mp4")

    # Return the processed video and results as response
    return Response(content=video_data, media_type="video/mp4", headers={"predictions": results})
