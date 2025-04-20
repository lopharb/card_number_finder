# Card Number Finder

## Flow Description

The system uses a multi-step approach that includes:
- Image Preprocessing (detecting the card and upscaling)
- Rotating images to align the card with its edges
- Recognizing the text on preprocessed images

The approach is described more thoroughly in the [flow diagram](https://drive.google.com/file/d/14tto3A8LCbSDPgSIpVnahDWFv3PCxNRK/view?usp=sharing).

## Models
- Ultralytics' PyTorch implementation of [YOLOv11](https://github.com/ultralytics/ultralytics) is used for card detection as one of the pre-processing steps. The model was fine-tuned on a custom dataset for the downstream task of segmenting bank card images.
- The default [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) English open-source model is used for text recognition.

## Data
Card detection model is traied on a custom dataset containing images of credit cards labeled for *instance segmantation* (it, unlike detection, allows for auto-orienting and cropping the card on the image).
- The dataset can be found [here](https://drive.google.com/file/d/152Viup539ayMnovKcLbwy6GPWXD3lhlS/view?usp=sharing)
- The dataset was created basing on another [open-source dataset](https://universe.roboflow.com/test-70hyp/credit-card-xk7ik/browse?queryText=class%3Acard&pageSize=50&startingIndex=0&browseQuery=true) published on Roboflow. The labeling was done using Roboflow's AI Labeling assistant and manual supervision.
- Training data is augmented during training to ensure higher robustness. 