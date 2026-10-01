# 🚗 Vehicle Detection Using Faster R-CNN

A practical computer vision project for detecting vehicles in images and
video using a **pretrained Faster R-CNN model with a ResNet-50 + Feature
Pyramid Network (FPN) backbone** from Torchvision.

The project focuses on understanding and implementing an end-to-end
object-detection workflow on a CPU environment, including:

-   Pretrained Faster R-CNN inference
-   Bounding-box detection
-   Confidence-score filtering
-   Non-Maximum Suppression (NMS)
-   Detection on multiple images
-   Detection on video
-   Manual ground-truth annotation
-   IoU-based evaluation
-   Precision, recall, and F1-score calculation
-   VOC 2007 single-image evaluation

> **Important:** This repository uses a pretrained Faster R-CNN model.
> There is no training script in the current project structure. The
> implementation performs inference and evaluation using pretrained COCO
> weights.

------------------------------------------------------------------------

## 📌 Project Overview

Object detection is the task of identifying objects in an image and
locating them using bounding boxes.

In this project, Faster R-CNN is used to detect objects using the
pretrained **Faster R-CNN ResNet-50 FPN COCO model**.

The main workflow is:

``` text
Input Image / Video
        ↓
Pretrained Faster R-CNN
        ↓
ResNet-50 + FPN
        ↓
Raw Bounding-Box Predictions
        ↓
Confidence Filtering
        ↓
Non-Maximum Suppression
        ↓
Final Detections
        ↓
Visualization / Evaluation
```

For the vehicle-focused experiments, the project works with classes such
as **car** and **truck** from the COCO label set. The multi-image and
video scripts retain the model's COCO class labels in their outputs.

------------------------------------------------------------------------

# 🎯 Objectives

The project was developed to understand the practical implementation of
object detection using Faster R-CNN.

### Main objectives

1.  Load a pretrained Faster R-CNN model.
2.  Run object detection on vehicle images.
3.  Understand model-generated bounding boxes, labels, and confidence
    scores.
4.  Apply a confidence threshold to remove low-confidence predictions.
5.  Apply NMS to remove redundant overlapping detections.
6.  Save and visualize detection results.
7.  Process multiple images automatically.
8.  Process a video and generate a detected output video.
9.  Create manually annotated ground-truth bounding boxes.
10. Compare predicted boxes with ground-truth boxes using IoU.
11. Calculate TP, FP, FN, precision, recall, and F1-score.
12. Perform a separate VOC 2007 single-image evaluation experiment.

------------------------------------------------------------------------

# 🧠 Model

## Faster R-CNN

The project uses Torchvision's pretrained:

``` text
Faster R-CNN
ResNet-50
Feature Pyramid Network (FPN)
COCO pretrained weights
```

The model is loaded using:

``` python
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1

model = fasterrcnn_resnet50_fpn(
    weights=weights
)
```

The model is placed in evaluation mode before inference.

------------------------------------------------------------------------

# 🏗️ Faster R-CNN Architecture

The high-level detection pipeline is:

``` text
                 Input Image
                      │
                      ▼
               ResNet-50 Backbone
                      │
                      ▼
          Feature Pyramid Network
                      │
                      ▼
        Region Proposal Network
                      │
                      ▼
             Region Proposals
                      │
                      ▼
               RoI Processing
                      │
              ┌───────┴────────┐
              ▼                ▼
        Classification    Box Regression
              │                │
              └───────┬────────┘
                      ▼
             Detection Results
                      │
                      ▼
             Confidence Filter
                      │
                      ▼
                    NMS
                      │
                      ▼
             Final Bounding Boxes
```

------------------------------------------------------------------------

# 🔍 What the Model Returns

For an input image, Faster R-CNN returns:

``` python
prediction["boxes"]
prediction["labels"]
prediction["scores"]
```

### `boxes`

Bounding-box coordinates:

``` text
[x1, y1, x2, y2]
```

### `labels`

COCO class IDs.

For example:

``` text
3 → car
8 → truck
```

### `scores`

The model's confidence score for each prediction.

------------------------------------------------------------------------

# 🎚️ Confidence Threshold

The main detection scripts use:

``` text
Confidence threshold = 0.80
```

Predictions with scores below `0.80` are removed.

The single-image detection script also compares several thresholds:

``` text
0.50
0.60
0.70
0.80
0.90
```

This makes it possible to observe how the number of retained predictions
changes as the confidence requirement becomes stricter.

------------------------------------------------------------------------

# 🧹 Non-Maximum Suppression

The multi-image and video pipelines use:

``` text
NMS IoU threshold = 0.50
```

NMS is applied after confidence filtering.

The process is:

``` text
Raw predictions
       ↓
Confidence ≥ 0.80
       ↓
NMS with IoU = 0.50
       ↓
Final detections
```

The purpose is to remove highly overlapping duplicate bounding boxes.

------------------------------------------------------------------------

# 📐 Intersection over Union (IoU)

IoU measures the overlap between a predicted bounding box and a
ground-truth bounding box.

``` text
              Intersection
IoU = ─────────────────────────
              Union
```

The implementation calculates:

``` text
intersection area
        /
union area
```

An IoU threshold of:

``` text
0.50
```

is used for the matching experiments.

A prediction is treated as a true positive when its best available IoU
with a ground-truth box is at least `0.50`.

------------------------------------------------------------------------

# 📊 Evaluation Metrics

The project calculates:

## True Positive

A ground-truth object is successfully matched with a prediction at or
above the IoU threshold.

## False Negative

A ground-truth object does not have a prediction meeting the IoU
threshold.

## False Positive

A predicted box remains unmatched with a ground-truth object.

The evaluation scripts calculate:

### Precision

``` text
Precision = TP / (TP + FP)
```

### Recall

``` text
Recall = TP / (TP + FN)
```

### F1 Score

``` text
F1 = 2 × Precision × Recall
             ─────────────────────
              Precision + Recall
```

------------------------------------------------------------------------

# 🏷️ Ground-Truth Annotation

The project includes a manual annotation utility:

``` text
annotate_ground_truth.py
```

It loads:

``` text
images/car_001.jpg
```

and asks the user to manually draw bounding boxes around **5 cars**.

The annotation process uses Matplotlib's `RectangleSelector`.

Workflow:

``` text
Open car_001.jpg
       ↓
Draw bounding box around car
       ↓
Repeat for 5 cars
       ↓
Save coordinates
       ↓
ground_truth_boxes.pt
```

The saved boxes are stored as a PyTorch tensor with:

``` text
[x_min, y_min, x_max, y_max]
```

coordinates.

------------------------------------------------------------------------

# 🧪 Single-Image Detection

The file:

``` text
detect.py
```

performs detection on:

``` text
images/car_001.jpg
```

The script:

1.  Loads the pretrained Faster R-CNN model.
2.  Loads the image.
3.  Applies the model's preprocessing transformation.
4.  Runs inference.
5.  Displays raw predictions.
6.  Applies a confidence threshold of `0.80`.
7.  Saves filtered predictions.
8.  Draws bounding boxes.
9.  Saves the detected image.
10. Compares several confidence thresholds.

Output:

``` text
results/car_001_detected.jpg
```

Predictions are saved as:

``` text
predictions_car_001.pt
```

------------------------------------------------------------------------

# 🖼️ Multi-Image Detection

The file:

``` text
detect_all_images.py
```

processes all:

``` text
.jpg
.jpeg
.png
```

files inside:

``` text
images/
```

For each image:

``` text
Image
  ↓
Faster R-CNN
  ↓
Raw predictions
  ↓
Confidence ≥ 0.80
  ↓
NMS IoU = 0.50
  ↓
Final detections
  ↓
Save annotated image
```

The script prints a summary containing:

``` text
Image
Raw predictions
After confidence filtering
After NMS
```

and also prints totals across all processed images.

Output files are written to:

``` text
results/
```

with names such as:

``` text
car_001_detected.jpg
```

------------------------------------------------------------------------

# 🎥 Video Detection

The file:

``` text
detect_video.py
```

extends the detection pipeline to video.

Input:

``` text
project_video.mp4
```

Output:

``` text
project_video_detected.mp4
```

The script uses OpenCV for video processing.

------------------------------------------------------------------------

## Frame Processing

Because Faster R-CNN inference can be computationally expensive on CPU,
the project processes:

``` text
Every 5th frame
```

using:

``` python
FRAME_SKIP = 5
```

The workflow is:

``` text
Input Video
     ↓
Read Frames
     ↓
Select Every 5th Frame
     ↓
Convert BGR → RGB
     ↓
Faster R-CNN
     ↓
Confidence ≥ 0.80
     ↓
NMS IoU = 0.50
     ↓
Draw Bounding Boxes
     ↓
Write Output Video
```

The script also reports:

-   Video width
-   Video height
-   FPS
-   Total frame count
-   Video duration
-   Number of processed frames
-   Frame skip value
-   Total detections across processed frames

------------------------------------------------------------------------

# 🧮 IoU Evaluation

The file:

``` text
evaluate_iou.py
```

loads:

``` text
ground_truth_boxes.pt
predictions_car_001.pt
```

and compares the predicted bounding boxes against the manually created
ground-truth boxes.

The IoU threshold is:

``` text
0.50
```

For each ground-truth box, the script searches for the unmatched
prediction with the highest IoU.

The matching logic is:

``` text
Ground Truth Box
       ↓
Compare with all unmatched predictions
       ↓
Find highest IoU
       ↓
IoU ≥ 0.50 ?
     /     \
   YES      NO
    ↓        ↓
   TP       FN
```

After processing all ground-truth boxes, unmatched predictions are
counted as false positives.

------------------------------------------------------------------------

# 📋 VOC 2007 Single-Image Evaluation

The file:

``` text
evaluate_voc_single.py
```

contains a separate evaluation experiment using the image:

``` text
VOC 2007 - 000004.jpg
```

The script contains manually specified ground-truth boxes and Faster
R-CNN prediction boxes.

The evaluation uses:

``` text
IoU threshold = 0.50
```

The script performs one-to-one matching between ground-truth boxes and
predictions and calculates:

``` text
True Positives
False Positives
False Negatives
Precision
Recall
F1 Score
```

For the supplied coordinates in this experiment, all 7 ground-truth
boxes have a best prediction with IoU ≥ 0.50. Therefore the script's
calculation produces:

``` text
Ground-truth cars : 7
Predicted cars    : 7
True Positives    : 7
False Positives   : 0
False Negatives   : 0

Precision : 1.0000
Recall    : 1.0000
F1 Score  : 1.0000
```

These numbers describe **this specific VOC 2007 single-image evaluation
experiment**. They should not be interpreted as the overall performance
of Faster R-CNN on the full COCO dataset or on a larger vehicle test
set.

------------------------------------------------------------------------

# 📁 Project Structure

The current project structure is:

``` text
FASTER_RCNN_SMALL_PROJECT/
│
├── images/
│   └── car_001.jpg
│
├── results/
│   └── car_001_detected.jpg
│
├── venv/
│
├── annotate_ground_truth.py
├── detect.py
├── detect_all_images.py
├── detect_video.py
├── evaluate_iou.py
├── evaluate_voc_single.py
│
├── ground_truth_boxes.pt
├── predictions_car_001.pt
│
├── project_video.mp4
└── project_video_detected.mp4
```

------------------------------------------------------------------------

# 📄 File Descriptions

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `annotate_ground_truth.py`          Manually creates ground-truth
                                      bounding boxes for 5 cars

  `detect.py`                         Runs Faster R-CNN on a single image

  `detect_all_images.py`              Runs detection on all images in
                                      `images/`

  `detect_video.py`                   Runs detection on selected video
                                      frames

  `evaluate_iou.py`                   Evaluates predictions against saved
                                      ground-truth boxes using IoU

  `evaluate_voc_single.py`            Performs a VOC 2007 single-image
                                      evaluation

  `ground_truth_boxes.pt`             Saved manually annotated bounding
                                      boxes

  `predictions_car_001.pt`            Saved predictions for `car_001.jpg`

  `project_video.mp4`                 Input video

  `project_video_detected.mp4`        Video containing detection results

  `images/`                           Input images

  `results/`                          Generated detection results

  `venv/`                             Python virtual environment
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 🛠️ Technologies Used

  Technology                 Purpose
  -------------------------- ------------------------------------
  Python                     Programming language
  PyTorch                    Deep learning framework
  Torchvision                Faster R-CNN implementation
  Faster R-CNN               Object detection
  ResNet-50                  Feature extraction backbone
  FPN                        Multi-scale feature representation
  OpenCV                     Video processing
  Pillow                     Image loading and manipulation
  Matplotlib                 Ground-truth annotation
  NumPy / Python utilities   Supporting operations

------------------------------------------------------------------------

# 💻 Hardware

The project uses:

``` python
torch.device("cpu")
```

The detection and video scripts therefore run on the CPU.

This is particularly relevant for video because Faster R-CNN is
computationally more expensive than lightweight one-stage detectors.

To reduce video processing time, the video pipeline processes every
fifth frame.

------------------------------------------------------------------------

# ⚙️ Installation

Create and activate a virtual environment:

``` bash
python -m venv venv
```

Windows:

``` bash
venv\Scripts\activate
```

Install the required packages:

``` bash
pip install torch torchvision pillow matplotlib opencv-python
```

------------------------------------------------------------------------

# ▶️ How to Run

## 1. Run Single-Image Detection

Make sure the image exists:

``` text
images/car_001.jpg
```

Run:

``` bash
python detect.py
```

The result will be saved to:

``` text
results/car_001_detected.jpg
```

Predictions will be saved to:

``` text
predictions_car_001.pt
```

------------------------------------------------------------------------

## 2. Detect All Images

Place images inside:

``` text
images/
```

Then run:

``` bash
python detect_all_images.py
```

Results will be saved inside:

``` text
results/
```

------------------------------------------------------------------------

## 3. Create Ground Truth

Run:

``` bash
python annotate_ground_truth.py
```

A Matplotlib window will open.

For each car:

1.  Click and hold the left mouse button.
2.  Drag around the complete car.
3.  Release the mouse button.
4.  Repeat until 5 cars are annotated.
5.  Close the image window.

The boxes are saved as:

``` text
ground_truth_boxes.pt
```

------------------------------------------------------------------------

## 4. Evaluate IoU

After both ground truth and prediction files have been created:

``` bash
python evaluate_iou.py
```

The script reports:

``` text
Ground-truth objects
Predicted objects
True Positives
False Positives
False Negatives
Precision
Recall
F1 Score
```

------------------------------------------------------------------------

## 5. Run VOC 2007 Evaluation

Run:

``` bash
python evaluate_voc_single.py
```

This performs the predefined VOC 2007 single-image experiment using:

``` text
000004.jpg
```

------------------------------------------------------------------------

## 6. Run Video Detection

Make sure:

``` text
project_video.mp4
```

exists in the project root.

Run:

``` bash
python detect_video.py
```

The processed video is saved as:

``` text
project_video_detected.mp4
```

------------------------------------------------------------------------

# 🔄 Complete Project Workflow

The project can be understood as several connected stages.

## Stage 1 --- Model Setup

``` text
Load pretrained Faster R-CNN
        ↓
Use ResNet-50 + FPN
        ↓
Move model to CPU
        ↓
Set evaluation mode
```

------------------------------------------------------------------------

## Stage 2 --- Image Detection

``` text
Input Image
     ↓
Preprocessing
     ↓
Faster R-CNN
     ↓
Raw boxes + labels + scores
```

------------------------------------------------------------------------

## Stage 3 --- Prediction Filtering

``` text
Raw predictions
       ↓
Confidence ≥ 0.80
       ↓
NMS IoU = 0.50
       ↓
Final detections
```

------------------------------------------------------------------------

## Stage 4 --- Visualization

``` text
Final detections
       ↓
Bounding boxes
       ↓
Class labels
       ↓
Confidence scores
       ↓
Saved result image/video
```

------------------------------------------------------------------------

## Stage 5 --- Evaluation

``` text
Manual Ground Truth
        +
Model Predictions
        ↓
IoU Matching
        ↓
TP / FP / FN
        ↓
Precision / Recall / F1
```

------------------------------------------------------------------------

# 🧠 Concepts Covered

This project provides practical exposure to the following
computer-vision concepts:

### Object Detection

-   Object classification
-   Object localization
-   Bounding boxes
-   Detection confidence

### Faster R-CNN

-   Region Proposal Network
-   Two-stage object detection
-   ResNet-50
-   Feature Pyramid Network
-   RoI-based detection

### Post-processing

-   Confidence thresholding
-   Non-Maximum Suppression
-   IoU

### Evaluation

-   Ground-truth annotation
-   TP
-   FP
-   FN
-   Precision
-   Recall
-   F1 score

### Video Processing

-   Frame extraction
-   Frame skipping
-   BGR to RGB conversion
-   Video writing
-   Frame-level detection

------------------------------------------------------------------------

# ⚠️ Limitations

The current implementation has some important limitations.

### 1. No custom training pipeline

The repository currently performs inference using pretrained COCO
weights. There is no custom training script in the project.

### 2. CPU inference

The model is explicitly configured to run on CPU, which can make
inference slow.

### 3. Video frame skipping

Only every fifth frame is processed in the video pipeline. This reduces
computation but means not every frame receives a fresh Faster R-CNN
inference.

### 4. COCO classes

The pretrained model contains the COCO object categories. The scripts
therefore can produce classes beyond cars.

### 5. Limited evaluation

The IoU evaluation is based on manually defined ground-truth boxes for
selected images rather than a large-scale benchmark evaluation.

### 6. Single-image VOC experiment

The VOC 2007 evaluation represents one predefined image and should not
be treated as a general model benchmark.

------------------------------------------------------------------------

# 🔮 Future Improvements

Possible extensions include:

-   Fine-tune Faster R-CNN on a dedicated vehicle dataset.
-   Add a proper dataset loader and training pipeline.
-   Add train/validation/test splits.
-   Calculate mAP@0.50.
-   Calculate mAP@0.50:0.95.
-   Evaluate on a larger test dataset.
-   Add class-specific filtering for vehicles.
-   Improve CPU inference efficiency.
-   Run inference using CUDA when a compatible GPU is available.
-   Process every frame when real-time requirements allow it.
-   Add vehicle tracking across video frames.
-   Compare Faster R-CNN with YOLO-based detectors.
-   Add automated evaluation visualizations.
-   Add confusion/error analysis for difficult images.

------------------------------------------------------------------------

# 📊 Detection Pipeline Summary

``` text
                    VEHICLE DETECTION
                           │
                           ▼
                    Input Image/Video
                           │
                           ▼
                 Pretrained Faster R-CNN
                           │
                           ▼
                  ResNet-50 + FPN
                           │
                           ▼
                  Raw Predictions
                 ┌─────────┴─────────┐
                 │                   │
              Boxes                Scores
                 │                   │
                 └─────────┬─────────┘
                           ▼
                 Confidence Filtering
                    Threshold = 0.80
                           │
                           ▼
                 Non-Maximum Suppression
                      IoU = 0.50
                           │
                           ▼
                  Final Detections
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
         Visualization              Evaluation
              │                         │
              ▼                         ▼
       Image / Video             Ground Truth
                                        │
                                        ▼
                                       IoU
                                        │
                                        ▼
                                  TP / FP / FN
                                        │
                                        ▼
                              Precision / Recall
                                        │
                                        ▼
                                    F1 Score
```

------------------------------------------------------------------------

# 📌 Key Results From the Included Evaluation

The included VOC 2007 single-image evaluation contains:

``` text
Ground-truth cars : 7
Predicted cars    : 7
IoU threshold     : 0.50

True Positives    : 7
False Positives   : 0
False Negatives   : 0

Precision         : 1.0000
Recall            : 1.0000
F1 Score          : 1.0000
```

The result is specific to the seven manually supplied
ground-truth/prediction pairs in that evaluation script.

------------------------------------------------------------------------

# 👨‍💻 Author

**Vamsi Krishna**

Computer Science & Engineering

### Areas of Interest

-   Computer Vision
-   Machine Learning
-   Deep Learning
-   Python
-   Artificial Intelligence
-   Generative AI
-   Data Science

------------------------------------------------------------------------

# ⭐ Project Summary

This project demonstrates a practical implementation of
**vehicle-focused object detection using pretrained Faster R-CNN**.

It goes beyond simply running a pretrained model by implementing the
complete workflow:

``` text
Detection
   ↓
Confidence Filtering
   ↓
NMS
   ↓
Visualization
   ↓
Ground-Truth Annotation
   ↓
IoU Matching
   ↓
TP / FP / FN
   ↓
Precision / Recall / F1
```

The project also extends the detector from individual images to multiple
images and video, while taking CPU-based computational constraints into
account.
