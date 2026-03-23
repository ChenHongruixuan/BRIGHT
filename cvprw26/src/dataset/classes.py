"""Category definitions for BRIGHT building damage instance segmentation."""

# Category ID -> name mapping (excludes background, matches annotation category_id)
CATEGORIES = {
    1: "intact",
    2: "damaged",
    3: "destroyed",
}

# Total number of classes for Mask R-CNN (includes background)
NUM_CLASSES = len(CATEGORIES) + 1  # 4
