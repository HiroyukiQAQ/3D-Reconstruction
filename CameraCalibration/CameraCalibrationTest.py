import numpy as np
import cv2
import glob
import os

print("Start camera processing...")
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ImgPath = '/home/wmh/work/seqbuff/'
# ImgPath = '/home/wmh/work/seqbuff/usb-cam/'

# number of rows and  columns in the chess board
row = 7
col = 7

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((col * row, 3), np.float32)
objp[:, :2] = np.mgrid[0:row, 0:col].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


# images = glob.glob('*.png')
images = glob.glob('*.jpg')



# print (len(images)," images are found.")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray = img

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (row, col), None)
    # cv2.imwrite('g'+fname,gray)
    # print fname,": ",ret

    # If found, add object points, image points (after refining them)
    if ret is True:
        print(fname, "is available.")
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners, then save the result images.
        # cv2.drawChessboardCorners(img, (row, col), corners, ret)
        # print ImgPath + 'cross_' + fname.split('/')[-1]
        # cv2.imwrite(ImgPath+'cross_'+fname.split('/')[-1],img)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)
        # cv2.destroyAllWindows()
    #If not found, remove the bad image.
    else:
        os.remove(fname)

print("In",len(images),"images, find", len(objpoints), "available.")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print("Cam Matrix: ", mtx)
print("fx =",mtx[0,0])
print("fy =",mtx[1,1])
print("cx =",mtx[0,2])
print("cy =",mtx[1,2])

print("Distortion Factor: ", dist)
dist = dist.reshape(5,1)
print ("k1 =",dist[0])
print ("k2 =",dist[1])
print ("p1 =",dist[2])
print ("p2 =",dist[3])
print ("k3 =",dist[4])