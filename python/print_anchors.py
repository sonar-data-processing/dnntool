import cv2
import numpy as np

input_size = 416
img_h, img_w = (422, 730)
grid_w = input_size/32
grid_h = input_size/32
cell_w = img_w / grid_w
cell_h = img_h / grid_h


anchors = [[1.53,3.87], [1.86,2.27], [2.49,3.81], [4.28,7.55], [6.35,4.15]]
scales = [16, 32, 48, 64]
# scales = [128, 256, 512]
img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

cy, cx = (img_h/2, img_w/2)

print cell_w, cell_h

for s in scales:
    for anchor in anchors:
        w, h = anchor
        w = w/2.0 * cell_w
        h = h/2.0 * cell_h
        x1 = int(cx - w)
        y1 = int(cy - h)
        x2 = int(cx + w)
        y2 = int(cy + h)
        cv2.rectangle(img, (x1, y1), (x2, y2), 255)

cv2.imshow("img", img)
cv2.waitKey()