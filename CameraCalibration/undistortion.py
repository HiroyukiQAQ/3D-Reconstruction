import numpy as np
import cv2

parameters = np.load('parameters.npz')
mtx=parameters['mtx']
dist=parameters['dist']
img = cv2.imread('IMG_1286.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png', dst)
exit()