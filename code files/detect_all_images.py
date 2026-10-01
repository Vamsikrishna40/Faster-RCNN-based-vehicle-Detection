import os
import torch
from PIL import Image, ImageDraw
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "results"

CONFIDENCE_THRESHOLD = 0.80
NMS_THRESHOLD = 0.50

DEVICE = torch.device("cpu")


# COCO class names
COCO_CLASSES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
]


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD PRETRAINED FASTER R-CNN
# ============================================================

print("=" * 70)
print("LOADING PRETRAINED FASTER R-CNN")
print("=" * 70)

print(f"Using device: {DEVICE}")

weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1

model = fasterrcnn_resnet50_fpn(
    weights=weights
)

model.to(DEVICE)

model.eval()

transform = weights.transforms()

print("Model loaded successfully!")


# ============================================================
# FIND IMAGES
# ============================================================

image_files = [
    file
    for file in os.listdir(IMAGE_FOLDER)
    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

image_files.sort()


print()
print("=" * 70)
print("IMAGES FOUND")
print("=" * 70)

print(
    f"Number of images: "
    f"{len(image_files)}"
)

for file in image_files:
    print(f" - {file}")


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

summary = []


for image_number, image_file in enumerate(
    image_files,
    start=1
):

    print()
    print("=" * 70)

    print(
        f"PROCESSING IMAGE "
        f"{image_number}/{len(image_files)}"
    )

    print("=" * 70)

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_file
    )


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")


    print(
        f"Image: {image_file}"
    )

    print(
        f"Image size: {image.size}"
    )


    # --------------------------------------------------------
    # TRANSFORM IMAGE
    # --------------------------------------------------------

    image_tensor = transform(
        image
    ).to(DEVICE)


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print(
        "Running Faster R-CNN inference..."
    )

    with torch.inference_mode():

        prediction = model(
            [image_tensor]
        )[0]


    # --------------------------------------------------------
    # RAW PREDICTIONS
    # --------------------------------------------------------

    boxes = prediction["boxes"]

    labels = prediction["labels"]

    scores = prediction["scores"]


    raw_count = len(boxes)


    # --------------------------------------------------------
    # CONFIDENCE FILTER
    # --------------------------------------------------------

    confidence_mask = (
        scores >= CONFIDENCE_THRESHOLD
    )


    boxes = boxes[
        confidence_mask
    ]

    labels = labels[
        confidence_mask
    ]

    scores = scores[
        confidence_mask
    ]


    after_confidence = len(boxes)


    # --------------------------------------------------------
    # NMS
    # --------------------------------------------------------

    keep = torchvision.ops.nms(
        boxes,
        scores,
        NMS_THRESHOLD
    )


    boxes = boxes[keep]

    labels = labels[keep]

    scores = scores[keep]


    final_count = len(boxes)


    # --------------------------------------------------------
    # DRAW RESULT
    # --------------------------------------------------------

    result_image = image.copy()

    draw = ImageDraw.Draw(
        result_image
    )


    for box, label, score in zip(
        boxes,
        labels,
        scores
    ):

        x1, y1, x2, y2 = [
            int(value)
            for value in box
        ]


        label_id = int(label)

        if label_id < len(
            COCO_CLASSES
        ):

            class_name = COCO_CLASSES[
                label_id
            ]

        else:

            class_name = str(
                label_id
            )


        # Draw bounding box

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2
            ],
            outline="red",
            width=3
        )


        # Label text

        text = (
            f"{class_name} "
            f"{float(score):.2f}"
        )


        # Text background

        text_bbox = draw.textbbox(
            (x1, y1),
            text
        )


        draw.rectangle(
            text_bbox,
            fill="red"
        )


        draw.text(
            (x1, y1),
            text,
            fill="white"
        )


    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    base_name = os.path.splitext(
        image_file
    )[0]


    output_path = os.path.join(
        OUTPUT_FOLDER,
        f"{base_name}_detected.jpg"
    )


    result_image.save(
        output_path
    )


    # --------------------------------------------------------
    # PRINT DETECTIONS
    # --------------------------------------------------------

    print()
    print(
        f"Raw predictions: "
        f"{raw_count}"
    )

    print(
        f"After confidence "
        f"({CONFIDENCE_THRESHOLD}): "
        f"{after_confidence}"
    )

    print(
        f"After NMS: "
        f"{final_count}"
    )


    print()
    print("Final detections:")


    for i, (
        box,
        label,
        score
    ) in enumerate(
        zip(
            boxes,
            labels,
            scores
        ),
        start=1
    ):

        label_id = int(label)

        class_name = (
            COCO_CLASSES[label_id]
            if label_id <
            len(COCO_CLASSES)
            else str(label_id)
        )


        print(
            f"{i:2d}. "
            f"{class_name:12s} "
            f"Confidence: "
            f"{float(score):.3f}"
        )


    print()
    print(
        f"Saved: {output_path}"
    )


    # --------------------------------------------------------
    # STORE SUMMARY
    # --------------------------------------------------------

    summary.append(
        {
            "image": image_file,
            "raw": raw_count,
            "confidence": after_confidence,
            "final": final_count
        }
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 70)
print("FINAL DETECTION SUMMARY")
print("=" * 70)

print(
    f"{'Image':<15}"
    f"{'Raw':>10}"
    f"{'After Conf.':>15}"
    f"{'After NMS':>15}"
)

print("-" * 55)


total_raw = 0
total_confidence = 0
total_final = 0


for result in summary:

    print(
        f"{result['image']:<15}"
        f"{result['raw']:>10}"
        f"{result['confidence']:>15}"
        f"{result['final']:>15}"
    )


    total_raw += result["raw"]

    total_confidence += (
        result["confidence"]
    )

    total_final += result["final"]


print("-" * 55)

print(
    f"{'TOTAL':<15}"
    f"{total_raw:>10}"
    f"{total_confidence:>15}"
    f"{total_final:>15}"
)


print()
print("=" * 70)
print("ALL IMAGES PROCESSED SUCCESSFULLY")
print("=" * 70)

print(
    f"Results saved in: "
    f"{OUTPUT_FOLDER}/"
)