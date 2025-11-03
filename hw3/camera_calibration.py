import cv2
import numpy as np
from glob import glob


img_dir = "hw3/calibration_images/*.JPEG"
img_paths = glob(img_dir)

pattern_size = (8, 6) # Number of inner corners

for img_path in img_paths:
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(img_gray, pattern_size, None)

    if found:
        print(corners)
        cv2.drawChessboardCorners(img, pattern_size, corners, found)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("not found")
