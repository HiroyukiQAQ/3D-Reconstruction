import cv2
import numpy as np
import matplotlib.pyplot as plt

######  Input  #######

parameters = np.load('parameters.npz')
mtx = parameters['mtx']
dist = parameters['dist']

img1 = cv2.imread('KD_L.jpg', 0)
img2 = cv2.imread('KD_R.jpg', 0)

#######  Output  #########

out_fn = 'parameters_KD.npz'

#######  Code  ########
def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2



scaling_factor = 1.0
img1 = cv2.resize(img1, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

surf = cv2.xfeatures2d.SURF_create()
kp1, des1 = surf.detectAndCompute(img1, None)
# print(len(kp1))
kp2, des2 = surf.detectAndCompute(img2, None)
# print(len(kp2))

# sift = cv2.xfeatures2d.SIFT_create()
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)



# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)


pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3, 0.99)
E, mask_E = cv2.findEssentialMat(pts1, pts2, mtx, cv2.FM_RANSAC, 0.999, 1.0)

pts1_E = pts1[mask_E.ravel() == 1]
pts2_E = pts2[mask_E.ravel() == 1]

ret, R, T, mask_E = cv2.recoverPose(E, pts1_E, pts2_E, mtx)
print(R)
print(T)
print(E)
print(F)
# np.savez('parameters_VT.npz', R=R, T=T, E=E, F=F)
np.savez(out_fn, R=R, T=T, E=E, F=F)
# np.savez('parameters_KD.npz', R=R, T=T, E=E, F=F)
# select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]



# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
# lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
# lines1 = lines1.reshape(-1, 3)
# img_lines1, img_pits2 = drawlines(img1, img2, lines1, pts1, pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
# lines2 = lines2.reshape(-1,3)
# img_lines2, img_pits1 = drawlines(img2, img1, lines2, pts2, pts1)

# cv2.imshow('Epi lines on left', img_lines1)
# cv2.imshow('Feature points on right', img_pits2)
# cv2.imshow('Epi lines on right', img_lines2)
# cv2.imshow('Feature points on left', img_pits1)
# cv2.waitKey()
# cv2.destroyAllWindows()

# cv2.imwrite('img_left_lines.jpg', img_lines1)
# cv2.imwrite('img_right_lines.jpg', img_lines2)
# cv2.imwrite('img_left_points.jpg', img_pits1)
# cv2.imwrite('img_right_points.jpg', img_pits2)
exit()