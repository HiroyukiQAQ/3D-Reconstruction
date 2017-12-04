import cv2
import numpy as np
import matplotlib.pyplot as plt


parameters = np.load('parameters.npz')
mtx = parameters['mtx']
dist = parameters['dist']
rvecs = parameters['rvecs']
tvecs = parameters['tvecs']

parameters_RTE = np.load('parameters_RTE_KD.npz')
R = parameters_RTE['R']
T = parameters_RTE['T']
E = parameters_RTE['E']

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = vertices.reshape(-1, 3)
    vertices = np.hstack([vertices, colors])
    # vertices = vertices.astype(np.float16)
    with open(filename, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(vertices))).encode('utf-8'))
        np.savetxt(f, vertices, fmt='%f %f %f %d %d %d ')


imgL = cv2.imread('KDL.jpg')
imgR = cv2.imread('KDR.jpg')
# imgL = cv2.pyrDown(cv2.imread('Left.jpg'))
# imgR = cv2.pyrDown(cv2.imread('Left.jpg'))

window_size = 7  # wsize default 3; 5; 7 for SGBM reduced size image;
                  # 15 for SGBM full size image (1300px and above); 5 Works nicely
min_disp = 128
num_disp = 608-min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
    numDisparities=num_disp, # must be divisible by 16
    blockSize=5,  # must be an odd number >=1
    P1=8*3*window_size**2,
    P2=32*3*window_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=7,  # 5-15 range is good enough
    speckleWindowSize=200,  # Set it to 0 to disable speckle filtering.
                            # Otherwise, set it somewhere in the 50-200 range.
    speckleRange=1024,  # If you do speckle filtering, set the parameter to a positive value,
                      # it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
    mode=True
)

disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
print(disp.shape)
# ret, disp = cv2.threshold(disp, 230, 255, cv2.THRESH_TOZERO_INV)
# ret, disp = cv2.threshold(disp, 20, 255, cv2.THRESH_TRUNC)
# plt.imshow(disp)
# plt.show()
plt.imshow((disp-min_disp)/num_disp, plt.get_cmap('gray'))
# plt.imshow(disp, plt.get_cmap('gray'))
plt.show()


# h, w = imgL.shape[:2]
Q = np.float32([[1,  0,  0, -640],
                [0,  -1,  0, 640],  # turn points 180 deg around x-axis,
                [0,  0,  0,  -1460],  # so that y-axis looks up
                [0,  0,  1,    0]])

# R_Left, R_Right, P_Left, P_Right, Q, ROI1, ROI2 = cv2.stereoRectify(mtx, dist, mtx, dist, imgL.shape[:2], R, T)
# print(Q)

points_3D = cv2.reprojectImageTo3D(disp, Q)
colors_3D = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
mask_map = disp > disp.min()
out_points = points_3D[mask_map]
print(out_points.shape)
out_colors = colors_3D[mask_map]
print(out_colors.shape)
out_fn = 'KD_True.ply'
create_output(out_points, out_colors, out_fn)
print('%s saved' % out_fn)