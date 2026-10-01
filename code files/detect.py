import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("=" * 60)
print("FASTER R-CNN SMALL PROJECT")
print("=" * 60)

print("\nUsing device:", device)


# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================

print("\nLoading pretrained Faster R-CNN...")

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

model = fasterrcnn_resnet50_fpn(
    weights=weights
)


# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()

model.to(device)


print("\nModel loaded successfully!")

print("Model type:")
print(type(model))

print("\nPretrained weights:")
print(weights)

print("\nModel is ready for inference.")


# ============================================================
# LOAD IMAGE
# ============================================================

image_path = "images/car_001.jpg"

print("\nLoading image...")
print("Image path:", image_path)

image = Image.open(image_path).convert("RGB")

print("Image loaded successfully!")

print("Image type:", type(image))
print("Image size:", image.size)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

print("\nPreparing image for Faster R-CNN...")

transform = weights.transforms()

image_tensor = transform(image)

image_tensor = image_tensor.to(device)

print("Image transformed successfully!")

print("Tensor type:", type(image_tensor))
print("Tensor shape:", image_tensor.shape)
print("Tensor dtype:", image_tensor.dtype)
print("Tensor device:", image_tensor.device)

# ============================================================
# RUN INFERENCE
# ============================================================

print("\nRunning Faster R-CNN inference...")
print("Please wait... CPU inference may take some time.")

with torch.inference_mode():
    predictions = model([image_tensor])

print("\nInference completed!")

print("Number of images processed:", len(predictions))

# ============================================================
# INSPECT PREDICTIONS
# ============================================================

prediction = predictions[0]

print("\nPrediction keys:")
print(prediction.keys())

print("\nNumber of bounding boxes:")
print(len(prediction["boxes"]))

print("\nNumber of labels:")
print(len(prediction["labels"]))

print("\nNumber of confidence scores:")
print(len(prediction["scores"]))

# ============================================================
# DISPLAY RAW PREDICTIONS
# ============================================================

print("\nRaw predictions:")
print("=" * 70)

for i in range(
    min(20, len(prediction["boxes"]))
):

    box = prediction["boxes"][i]

    label = prediction["labels"][i]

    score = prediction["scores"][i]


    print(
        f"Prediction {i + 1}: "
        f"Label = {label.item()} | "
        f"Confidence = {score.item():.3f} | "
        f"Box = {box.tolist()}"
    )


    # ============================================================
# COUNT DETECTION CLASSES
# ============================================================

from collections import Counter


label_counts = Counter(
    prediction["labels"].tolist()
)


print("\nDetection class distribution:")
print("=" * 70)

for label_id, count in label_counts.items():

    print(
        f"Label {label_id}: "
        f"{count} detections"
    )


    # ============================================================
# CONFIDENCE FILTERING
# ============================================================

CONFIDENCE_THRESHOLD = 0.80


# COCO class names that we currently care about

CLASS_NAMES = {
    3: "car",
    8: "truck"
}


print("\nApplying confidence threshold...")
print(
    "Confidence threshold:",
    CONFIDENCE_THRESHOLD
)


# ============================================================
# FILTER PREDICTIONS
# ============================================================

filtered_predictions = []


for box, label, score in zip(
    prediction["boxes"],
    prediction["labels"],
    prediction["scores"]
):

    confidence = score.item()

    label_id = label.item()


    if confidence >= CONFIDENCE_THRESHOLD:

        filtered_predictions.append(
            {
                "box": box.tolist(),
                "label": label_id,
                "confidence": confidence
            }
        )


print(
    "\nPredictions before filtering:",
    len(prediction["boxes"])
)

print(
    "Predictions after filtering:",
    len(filtered_predictions)
)


# ============================================================
# DISPLAY FILTERED PREDICTIONS
# ============================================================

print("\nFiltered predictions:")
print("=" * 70)


for i, detection in enumerate(
    filtered_predictions
):

    label_id = detection["label"]

    label_name = CLASS_NAMES.get(
        label_id,
        f"unknown({label_id})"
    )


    print(
        f"{i + 1}. "
        f"{label_name} | "
        f"Confidence: "
        f"{detection['confidence']:.3f} | "
        f"Box: "
        f"{detection['box']}"
    )

# Save filtered predictions
torch.save(
    {
        "boxes": prediction["boxes"][prediction["scores"] >= 0.80].cpu(),
        "labels": prediction["labels"][prediction["scores"] >= 0.80].cpu(),
        "scores": prediction["scores"][prediction["scores"] >= 0.80].cpu()
    },
    "predictions_car_001.pt"
)

print("Predictions saved to: predictions_car_001.pt")

    # ============================================================
# DRAW DETECTIONS
# ============================================================

print("\nDrawing bounding boxes...")

result_image = image.copy()

draw = ImageDraw.Draw(result_image)


for detection in filtered_predictions:

    box = detection["box"]

    label_id = detection["label"]

    confidence = detection["confidence"]


    label_name = CLASS_NAMES.get(
        label_id,
        f"unknown({label_id})"
    )


    # Convert coordinates to integers

    x1, y1, x2, y2 = map(
        int,
        box
    )


    # Draw bounding box

    draw.rectangle(
        [x1, y1, x2, y2],
        outline="red",
        width=3
    )


    # Text

    text = (
        f"{label_name} "
        f"{confidence:.2f}"
    )


    # Draw text background

    text_bbox = draw.textbbox(
        (x1, y1),
        text
    )


    draw.rectangle(
        text_bbox,
        fill="red"
    )


    # Draw text

    draw.text(
        (x1, y1),
        text,
        fill="white"
    )


# ============================================================
# SAVE RESULT
# ============================================================

output_path = (
    "results/"
    "car_001_detected.jpg"
)


result_image.save(
    output_path
)


print("\nDetection image saved!")

print(
    "Output:",
    output_path
)


# ============================================================
# THRESHOLD COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("CONFIDENCE THRESHOLD COMPARISON")
print("=" * 70)

thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]

for threshold in thresholds:

    count = 0

    for score in prediction["scores"]:

        if score.item() >= threshold:
            count += 1

    print(
        f"Threshold: {threshold:.2f} "
        f"| Predictions: {count}"
    )