"""
PPE Compliance Detection System
Detects PPE items (helmet, vest, gloves, boots, goggles) and personnel using
a fine-tuned YOLOv8m model, then applies geometric rule-based logic to flag
missing safety gear per worker.
"""

import os
import glob
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "models/best.pt"
TEST_IMAGES_DIR = "data/test_images"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# REGION-OVERLAP COMPLIANCE LOGIC
# ============================================================
def check_region(person_box, item_boxes, region_top_pct, region_bottom_pct):
    """
    Checks if any item's center point falls within a specific vertical
    region of a person's bounding box (e.g. head, torso, feet).
    """
    px1, py1, px2, py2 = person_box
    h = py2 - py1
    region = (px1, py1 + h * region_top_pct, px2, py1 + h * region_bottom_pct)

    for ix1, iy1, ix2, iy2 in item_boxes:
        icx, icy = (ix1 + ix2) / 2, (iy1 + iy2) / 2
        if region[0] <= icx <= region[2] and region[1] <= icy <= region[3]:
            return True
    return False


def check_compliance(results):
    """
    Given YOLO inference results for one image, returns a per-person
    compliance report listing any missing PPE items.
    """
    boxes = results[0].boxes
    names = results[0].names

    persons, helmets, gloves, vests, boots_list, goggles = [], [], [], [], [], []

    for box, cls in zip(boxes.xyxy.tolist(), boxes.cls.tolist()):
        label = names[int(cls)]
        if label == "Person":
            persons.append(box)
        elif label == "helmet":
            helmets.append(box)
        elif label == "gloves":
            gloves.append(box)
        elif label == "vest":
            vests.append(box)
        elif label == "boots":
            boots_list.append(box)
        elif label == "goggles":
            goggles.append(box)

    report = []
    for i, p in enumerate(persons):
        violations = []
        if not check_region(p, helmets, 0.0, 0.25):
            violations.append("missing helmet")
        if not check_region(p, goggles, 0.0, 0.25):
            violations.append("missing goggles")
        if not check_region(p, vests, 0.25, 0.70):
            violations.append("missing vest")
        if not check_region(p, gloves, 0.20, 0.60):
            violations.append("missing gloves")
        if not check_region(p, boots_list, 0.80, 1.0):
            violations.append("missing boots")

        report.append({
            "person": i,
            "violations": violations,
            "compliant": len(violations) == 0
        })

    return report


# ============================================================
# MAIN — run inference + compliance check on all test images
# ============================================================
def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH)

    image_paths = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png"))

    if not image_paths:
        print(f"No images found in {TEST_IMAGES_DIR}. Add some test images first.")
        return

    for img_path in image_paths:
        print(f"\n--- Processing {img_path} ---")
        results = model(img_path)

        # Save annotated image with bounding boxes
        img_name = os.path.basename(img_path)
        output_path = os.path.join(OUTPUT_DIR, f"annotated_{img_name}")
        results[0].save(output_path)
        print(f"Saved annotated image to {output_path}")

        # Run compliance check
        report = check_compliance(results)
        if not report:
            print("No persons detected in this image.")
            continue

        for r in report:
            status = "COMPLIANT" if r["compliant"] else "NON-COMPLIANT"
            print(f"Person {r['person']}: {status}")
            if r["violations"]:
                print(f"  Violations: {', '.join(r['violations'])}")


if __name__ == "__main__":
    main()