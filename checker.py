import cv2, os, glob
import numpy as np

BLACK_OUT_MODE = True

os.makedirs("checked", exist_ok=True)

for path in sorted(glob.glob("frames/*.jpg")):
    name = os.path.basename(path)
    img = cv2.imread(path)
    h, w = img.shape[:2]

    if BLACK_OUT_MODE:
      mask = np.zeros((h, w), dtype=np.uint8)

    label_path = "labels/" + name.replace(".jpg", ".txt")
    count = 0

    with open(label_path) as f:
        for line in f:
            nums = list(map(float, line.split()[1:]))  # skip class ID
            pts = (np.array(nums).reshape(-1, 2) * [w, h]).astype("int32")

            if BLACK_OUT_MODE:
              cv2.fillPoly(mask, [pts], 255)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            count += 1
    if BLACK_OUT_MODE:
      img = cv2.bitwise_and(img, img, mask=mask)
    cv2.putText(img, f"{count} robots", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imwrite(f"checked/{name}", img)