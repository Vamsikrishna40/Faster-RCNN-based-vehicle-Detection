import cv2
import torch
import torchvision

from PIL import Image

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "project_video.mp4"
OUTPUT_PATH = "project_video_detected.mp4"

CONFIDENCE_THRESHOLD = 0.80
NMS_THRESHOLD = 0.50

# Process every 5th frame
FRAME_SKIP = 5

DEVICE = torch.device("cpu")


# ============================================================
# COCO CLASSES
# ============================================================

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
# LOAD MODEL
# ============================================================

print("=" * 70)
print("FASTER R-CNN VIDEO DETECTION")
print("=" * 70)

print(f"Using device: {DEVICE}")

print()
print("Loading pretrained Faster R-CNN...")

weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1

model = fasterrcnn_resnet50_fpn(
    weights=weights
)

model.to(DEVICE)

model.eval()

transform = weights.transforms()

print("Model loaded successfully!")


# ============================================================
# OPEN VIDEO
# ============================================================

print()
print("Opening video...")

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print("ERROR: Could not open video.")

    raise SystemExit


width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


duration = (
    total_frames / fps
    if fps > 0
    else 0
)


print()
print("Video information:")
print(
    f"Width: {width}"
)

print(
    f"Height: {height}"
)

print(
    f"FPS: {fps}"
)

print(
    f"Total frames: {total_frames}"
)

print(
    f"Duration: {duration:.2f} seconds"
)

print(
    f"Processing every {FRAME_SKIP}th frame"
)


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

processed_frames = 0

total_detections = 0


print()
print("=" * 70)
print("STARTING VIDEO DETECTION")
print("=" * 70)


while True:

    ret, frame = cap.read()

    if not ret:
        break


    frame_number += 1


    # --------------------------------------------------------
    # PROCESS SELECTED FRAMES
    # --------------------------------------------------------

    if frame_number % FRAME_SKIP == 1:

        processed_frames += 1


        print(
            f"Processing frame "
            f"{frame_number}/{total_frames}"
        )


        # OpenCV BGR → RGB

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        image = Image.fromarray(
            rgb_frame
        )


        # Transform

        image_tensor = transform(
            image
        ).to(DEVICE)


        # ----------------------------------------------------
        # FASTER R-CNN INFERENCE
        # ----------------------------------------------------

        with torch.inference_mode():

            prediction = model(
                [image_tensor]
            )[0]


        boxes = prediction["boxes"]

        labels = prediction["labels"]

        scores = prediction["scores"]


        # ----------------------------------------------------
        # CONFIDENCE FILTER
        # ----------------------------------------------------

        mask = (
            scores >= CONFIDENCE_THRESHOLD
        )


        boxes = boxes[mask]

        labels = labels[mask]

        scores = scores[mask]


        # ----------------------------------------------------
        # NMS
        # ----------------------------------------------------

        if len(boxes) > 0:

            keep = torchvision.ops.nms(
                boxes,
                scores,
                NMS_THRESHOLD
            )

            boxes = boxes[keep]

            labels = labels[keep]

            scores = scores[keep]


        total_detections += len(
            boxes
        )


        # ----------------------------------------------------
        # DRAW DETECTIONS
        # ----------------------------------------------------

        for box, label, score in zip(
            boxes,
            labels,
            scores
        ):

            x1, y1, x2, y2 = [
                int(v)
                for v in box
            ]


            label_id = int(
                label
            )


            class_name = (
                COCO_CLASSES[label_id]
                if label_id <
                len(COCO_CLASSES)
                else str(label_id)
            )


            # Draw bounding box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )


            text = (
                f"{class_name} "
                f"{float(score):.2f}"
            )


            # Text background

            (text_width,
             text_height), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )


            cv2.rectangle(
                frame,
                (x1, max(0, y1 - text_height - 8)),
                (
                    x1 + text_width,
                    y1
                ),
                (0, 0, 255),
                -1
            )


            # Text

            cv2.putText(
                frame,
                text,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


    # --------------------------------------------------------
    # WRITE FRAME
    # --------------------------------------------------------

    writer.write(frame)


# ============================================================
# RELEASE
# ============================================================

cap.release()

writer.release()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("VIDEO DETECTION COMPLETED")
print("=" * 70)

print(
    f"Total video frames: "
    f"{total_frames}"
)

print(
    f"Frames processed: "
    f"{processed_frames}"
)

print(
    f"Frame skip: "
    f"{FRAME_SKIP}"
)

print(
    f"Confidence threshold: "
    f"{CONFIDENCE_THRESHOLD}"
)

print(
    f"Total detections across "
    f"processed frames: "
    f"{total_detections}"
)

print()
print(
    f"Output saved to:"
)

print(
    OUTPUT_PATH
)

print("=" * 70)