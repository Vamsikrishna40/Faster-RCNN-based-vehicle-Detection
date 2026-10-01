import torch


# ============================================================
# SETTINGS
# ============================================================

GT_PATH = "ground_truth_boxes.pt"
PREDICTION_PATH = "predictions_car_001.pt"

IOU_THRESHOLD = 0.50


# ============================================================
# LOAD DATA
# ============================================================

ground_truth_boxes = torch.load(
    GT_PATH,
    weights_only=True
)

predictions = torch.load(
    PREDICTION_PATH,
    weights_only=True
)

predicted_boxes = predictions["boxes"]
predicted_scores = predictions["scores"]


print("=" * 70)
print("IoU EVALUATION")
print("=" * 70)

print(
    f"Ground-truth boxes : "
    f"{len(ground_truth_boxes)}"
)

print(
    f"Predicted boxes    : "
    f"{len(predicted_boxes)}"
)

print(
    f"IoU threshold      : "
    f"{IOU_THRESHOLD}"
)


# ============================================================
# IoU FUNCTION
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width *
        intersection_height
    )

    area_box1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area_box2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    union = (
        area_box1 +
        area_box2 -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# MATCHING
# ============================================================

matched_predictions = set()

TP = 0
FN = 0


print()
print("=" * 70)
print("GROUND TRUTH MATCHING")
print("=" * 70)


for gt_index, gt_box in enumerate(
    ground_truth_boxes
):

    best_iou = 0.0
    best_prediction = None

    for pred_index, pred_box in enumerate(
        predicted_boxes
    ):

        if pred_index in matched_predictions:
            continue

        iou = calculate_iou(
            gt_box,
            pred_box
        )

        if iou > best_iou:

            best_iou = iou
            best_prediction = pred_index


    print()
    print(
        f"GT-{gt_index + 1}"
    )

    print(
        "Ground Truth Box:",
        [
            round(float(x), 1)
            for x in gt_box
        ]
    )


    if best_prediction is not None:

        print(
            "Best Prediction:",
            [
                round(float(x), 1)
                for x in predicted_boxes[
                    best_prediction
                ]
            ]
        )

        print(
            f"Prediction Confidence: "
            f"{predicted_scores[best_prediction]:.3f}"
        )

        print(
            f"IoU: {best_iou:.4f}"
        )


        if best_iou >= IOU_THRESHOLD:

            TP += 1

            matched_predictions.add(
                best_prediction
            )

            print(
                "RESULT: TRUE POSITIVE"
            )

        else:

            FN += 1

            print(
                "RESULT: FALSE NEGATIVE"
            )

    else:

        FN += 1

        print(
            "RESULT: FALSE NEGATIVE"
        )


# ============================================================
# FALSE POSITIVES
# ============================================================

FP = (
    len(predicted_boxes)
    - len(matched_predictions)
)


# ============================================================
# METRICS
# ============================================================

if TP + FP > 0:

    precision = (
        TP /
        (TP + FP)
    )

else:

    precision = 0.0


if TP + FN > 0:

    recall = (
        TP /
        (TP + FN)
    )

else:

    recall = 0.0


if precision + recall > 0:

    f1 = (
        2 *
        precision *
        recall /
        (precision + recall)
    )

else:

    f1 = 0.0


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL IoU RESULTS")
print("=" * 70)

print(
    f"Ground-truth objects : "
    f"{len(ground_truth_boxes)}"
)

print(
    f"Predicted objects    : "
    f"{len(predicted_boxes)}"
)

print(
    f"True Positives (TP)  : "
    f"{TP}"
)

print(
    f"False Positives (FP) : "
    f"{FP}"
)

print(
    f"False Negatives (FN) : "
    f"{FN}"
)

print(
    f"Precision            : "
    f"{precision:.4f}"
)

print(
    f"Recall               : "
    f"{recall:.4f}"
)

print(
    f"F1 Score             : "
    f"{f1:.4f}"
)

print("=" * 70)