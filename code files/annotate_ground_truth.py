import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image
import torch


# ============================================================
# SETTINGS
# ============================================================

IMAGE_PATH = "images/car_001.jpg"
OUTPUT_PATH = "ground_truth_boxes.pt"

NUMBER_OF_CARS = 5

boxes = []


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")

print("=" * 70)
print("GROUND-TRUTH BOUNDING BOX ANNOTATION")
print("=" * 70)

print(f"Image: {IMAGE_PATH}")
print(f"Image size: {image.size}")
print()
print("Instructions:")
print("  1. Click and HOLD the left mouse button.")
print("  2. Drag around one complete car.")
print("  3. Release the mouse button.")
print("  4. Repeat for 5 cars.")
print()
print("Choose large, clearly visible cars.")
print("When finished, close the image window.")
print("=" * 70)


# ============================================================
# DISPLAY IMAGE
# ============================================================

fig, ax = plt.subplots(figsize=(14, 8))

ax.imshow(image)

ax.set_title(
    "Drag a box around each car | "
    f"0/{NUMBER_OF_CARS} completed"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")


# ============================================================
# CALLBACK
# ============================================================

def on_select(eclick, erelease):

    global boxes

    if len(boxes) >= NUMBER_OF_CARS:
        return


    x1 = eclick.xdata
    y1 = eclick.ydata

    x2 = erelease.xdata
    y2 = erelease.ydata


    # Check valid coordinates

    if (
        x1 is None or
        y1 is None or
        x2 is None or
        y2 is None
    ):
        return


    # Arrange coordinates correctly

    x_min = min(x1, x2)
    y_min = min(y1, y2)

    x_max = max(x1, x2)
    y_max = max(y1, y2)


    # Ignore extremely small boxes

    if (
        x_max - x_min < 5 or
        y_max - y_min < 5
    ):
        print("Box too small. Try again.")
        return


    box = [
        x_min,
        y_min,
        x_max,
        y_max
    ]

    boxes.append(box)


    # Draw permanent rectangle

    rectangle = plt.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        fill=False,
        linewidth=2
    )

    ax.add_patch(rectangle)


    # Add GT label

    ax.text(
        x_min,
        y_min,
        f"GT-{len(boxes)}",
        fontsize=10,
        bbox=dict(
            facecolor="white",
            alpha=0.8
        )
    )


    # Update title

    ax.set_title(
        "Drag a box around each car | "
        f"{len(boxes)}/{NUMBER_OF_CARS} completed"
    )


    fig.canvas.draw_idle()


# ============================================================
# RECTANGLE SELECTOR
# ============================================================

selector = RectangleSelector(
    ax,
    on_select,
    useblit=True,
    button=[1],
    minspanx=5,
    minspany=5,
    spancoords="pixels",
    interactive=False
)


# ============================================================
# SHOW WINDOW
# ============================================================

plt.show()


# ============================================================
# CHECK NUMBER OF BOXES
# ============================================================

if len(boxes) != NUMBER_OF_CARS:

    print()
    print(
        f"Annotation stopped with "
        f"{len(boxes)} boxes."
    )

    print(
        f"Expected {NUMBER_OF_CARS} boxes."
    )

    raise SystemExit


# ============================================================
# SAVE BOXES
# ============================================================

boxes_tensor = torch.tensor(
    boxes,
    dtype=torch.float32
)

torch.save(
    boxes_tensor,
    OUTPUT_PATH
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("GROUND-TRUTH BOXES CREATED")
print("=" * 70)

for i, box in enumerate(boxes_tensor):

    print(
        f"GT-{i + 1}: "
        f"["
        f"{box[0]:.1f}, "
        f"{box[1]:.1f}, "
        f"{box[2]:.1f}, "
        f"{box[3]:.1f}"
        f"]"
    )

print()
print(f"Saved to: {OUTPUT_PATH}")
print()
print("Annotation completed successfully!")