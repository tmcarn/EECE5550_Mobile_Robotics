import cv2
import numpy as np
from glob import glob
import os


img_dir = "calibration_images/*.JPEG"
img_paths = glob(img_dir)

pattern_size = (6, 8) # Number of inner corners

side_length = 0.01 # meters

# Define local coordinate system for chessboard
local_coords_shape = (8, 6, 3) # third dim added for x,y,z coord for each corner
local_coords = np.zeros(local_coords_shape, dtype=np.float32)
for j in range(pattern_size[1]):
    for i in range(pattern_size[0]):
        local_coords[j, i] = [i * side_length, j * side_length, 0.0] # z is constant 

local_coords = np.reshape(local_coords, (-1, 3)) # Reshape into (n,3) vector

obj_points = []
img_points = []

for img_path in img_paths:
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(img_gray, pattern_size, None)

    if found:
        # append object points and corresponding image points to running list
        obj_points.append(local_coords)
        img_points.append(corners)

        # visualize corner detection
        cv2.drawChessboardCorners(img, pattern_size, corners, found)
        img_name = os.path.basename(img_path)
        corner_path = os.path.join("corners", img_name)
        cv2.imwrite(corner_path, img)

        print(f"Corners found in image: {img_path}")

    else:
        print(f"No corners found in image:{img_path}")

# Batch callibration
print("Running Calibration")
ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, img_gray.shape, None, None)

if ret:
    # Save Camera Params for future use
    np.savez('calibration_params/camera_calibration.npz', 
            mtx=mtx,    # intrinsic matrix
            dist=dist)  # distortion coefficients
    
    # Saving Data in .txt for reporting
    with open('calibration_params/camera_calibration.txt', 'w') as f:
        f.write("Camera Calibration Parameters\n")
        f.write("="*50 + "\n\n")
        
        f.write("Intrinsic Calibration Matrix (K):\n")
        f.write(f"  [{mtx[0,0]:10.6f}, {mtx[0,1]:10.6f}, {mtx[0,2]:10.6f}]\n")
        f.write(f"  [{mtx[1,0]:10.6f}, {mtx[1,1]:10.6f}, {mtx[1,2]:10.6f}]\n")
        f.write(f"  [{mtx[2,0]:10.6f}, {mtx[2,1]:10.6f}, {mtx[2,2]:10.6f}]\n\n")
        
        f.write("Individual Parameters:\n")
        f.write(f"  fx (focal length x) = {mtx[0,0]:.6f} pixels\n")
        f.write(f"  fy (focal length y) = {mtx[1,1]:.6f} pixels\n")
        f.write(f"  px (principal point x) = {mtx[0,2]:.6f} pixels\n")
        f.write(f"  py (principal point y) = {mtx[1,2]:.6f} pixels\n\n")
        
        f.write("Distortion Coefficients:\n")
        f.write(f"  k1 = {dist[0,0]:10.6f} (radial)\n")
        f.write(f"  k2 = {dist[0,1]:10.6f} (radial)\n")
        f.write(f"  p1 = {dist[0,2]:10.6f} (tangential)\n")
        f.write(f"  p2 = {dist[0,3]:10.6f} (tangential)\n")
        f.write(f"  k3 = {dist[0,4]:10.6f} (radial)\n\n")
        
        f.write(f"RMS Reprojection Error: {ret:.6f} pixels\n")
        f.write(f"Image Size: {img_gray.shape[1]} x {img_gray.shape[0]} pixels\n")
        f.write(f"Number of Calibration Images: {len(obj_points)}\n")

    print("Calibration Parameters saved successfully")
    
    