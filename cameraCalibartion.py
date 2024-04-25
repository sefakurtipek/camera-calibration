import cv2
import numpy as np
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
checkerboard_dims = (10, 7)  # Adjusted dimensions based on your input
objp = np.zeros((np.prod(checkerboard_dims), 3), np.float32)
objp[:,:2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
objpoints = []
imgpoints = []
images = glob.glob('checkerBoardImages/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print("Failed to load", fname)
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
    print(fname, "Corners found:", ret)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw the corners
        cv2.drawChessboardCorners(img, checkerboard_dims, corners2, ret)
        # Save image
        save_path = 'calibrationDetections/' + fname.split('/')[-1]
        cv2.imwrite(save_path, img)
        print("Image saved to", save_path)
    else:
        # Optionally save images where detection failed for review
        fail_save_path = 'calibrationDetectionsFailed/' + fname.split('/')[-1]
        cv2.imwrite(fail_save_path, gray)
        print("Failed detection, image saved to", fail_save_path)

cv2.destroyAllWindows()

if len(objpoints) > 0 and len(imgpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:", dist)
    print("Rotation Vectors:", rvecs)
    print("Translation Vectors:", tvecs)
else:
    print("No corners were detected in any of the images.")