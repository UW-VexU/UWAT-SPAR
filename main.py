import os, glob, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import SAM

BOXES_MODE = False

device = "cuda" if torch.cuda.is_available() else "cpu"

# Text -> boxes
name = "IDEA-Research/grounding-dino-base"
# Proc turns things into tensors, dino gets the objects
proc = AutoProcessor.from_pretrained(name) 
dino = AutoModelForZeroShotObjectDetection.from_pretrained(name).to(device)

# Boxes -> outlines
sam = SAM("sam2_b.pt")

os.makedirs("labels", exist_ok=True)
print("Program Started...")
for path in sorted(glob.glob("frames/*.jpg")):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    
    # Find robots
    inputs = proc(images=img, text="A Vex Robot.", return_tensors="pt").to(device)
    with torch.no_grad():
        out = dino(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, inputs.input_ids,
        threshold=0.23, text_threshold=0.35,
        target_sizes=[(h, w)],
    )[0]
    boxes = res["boxes"].cpu().tolist()

    # Write label file (empty file = no robots)
    label_path = "labels/" + os.path.basename(path).replace(".jpg", ".txt")
    lines = []
    if boxes:
        if not BOXES_MODE:
          masks = sam(path, bboxes=boxes, verbose=False)[0].masks
          for poly in masks.xyn:  # normalized outline points
              if len(poly) < 3:
                  continue
              coords = " ".join(f"{x:.5f} {y:.5f}" for x, y in poly)
              lines.append(f"0 {coords}")
        if BOXES_MODE:
          for box in boxes:
            x_1 = box[0] / w
            y_1 = box[1] / h
            x_2 = box[2] / w
            y_2 = box[3] / h
            coords = f"{x_1:.5f} {y_1:.5f} {x_2:.5f} {y_1:.5f} {x_2:.5f} {y_2:.5f} {x_1:.5f} {y_2:.5f}"
            lines.append(f"0 {coords}")
    with open(label_path, "w") as f:
        f.write("\n".join(lines))
    print(f"{os.path.basename(path)}: {len(lines)} robots")
