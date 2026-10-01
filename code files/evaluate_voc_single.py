import torch


# ============================================================
# GROUND-TRUTH BOXES
# VOC 2007 - 000004.jpg
# ============================================================

ground_truth_boxes = torch.tensor([
    [13, 311, 84, 362],
    [362, 330, 500, 389],
    [235, 328, 334, 375],
    [175, 327, 252, 364],
    [139, 320, 189, 359],
    [108, 325, 150, 353],
    [84, 323, 121, 350]
], dtype=torch.float32)


# ============================================================
# FASTER R-CNN PREDICTIONS
# ============================================================

predicted_boxes = torch.tensor([
    [231.1, 328.4, 333.9, 374.9],
    [363.3, 329.5, 498.8, 387.4],
    [108.0, 326.6, 146.0, 352.4],
    [83.7, 326.9, 115.4, 349.5],
    [172.5, 327.8, 247.1, 362.9],
    [137.3, 323.3, 187.0, 356.8],
    [0.1, 324.5, 17.2, 342.9]
], dtype=torch.float32)


predicted_scores = torch.tensor([
    0.999,
    0.999,
    0.995,
    0.995,
    0.990,
    0.969,
    0.968
])


IOU_THRESHOLD = 0.50


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

    area1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    union = (
        area1 +
        area2 -
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


print("=" * 70)
print("VOC 2007 - SINGLE IMAGE EVALUATION")
print("=" * 70)

print("Image: 000004.jpg")
print(f"Ground-truth cars: {len(ground_truth_boxes)}")
print(f"Predicted cars: {len(predicted_boxes)}")
print(f"IoU threshold: {IOU_THRESHOLD}")


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
        "Ground truth:",
        [
            round(float(x), 1)
            for x in gt_box
        ]
    )


    print(
        f"Best prediction: "
        f"P-{best_prediction + 1}"
    )

    print(
        f"Confidence: "
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
            "Result: TRUE POSITIVE"
        )

    else:

        FN += 1

        print(
            "Result: FALSE NEGATIVE"
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

precision = (
    TP /
    (TP + FP)
)

recall = (
    TP /
    (TP + FN)
)

f1 = (
    2 * precision * recall /
    (precision + recall)
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    f"Ground-truth cars : {len(ground_truth_boxes)}"
)

print(
    f"Predicted cars    : {len(predicted_boxes)}"
)

print(
    f"True Positives    : {TP}"
)

print(
    f"False Positives   : {FP}"
)

print(
    f"False Negatives   : {FN}"
)

print(
    f"Precision         : {precision:.4f}"
)

print(
    f"Recall            : {recall:.4f}"
)

print(
    f"F1 Score          : {f1:.4f}"
)

print("=" * 70)