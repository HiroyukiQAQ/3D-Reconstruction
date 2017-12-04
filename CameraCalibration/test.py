import numpy as np
import cv2

parameters = np.load('parameters.npz')
mtx=parameters['mtx']
dist=parameters['dist']

print(mtx)