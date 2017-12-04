import numpy as np
import cv2

surf = cv2.xfeatures2d.SURF_create()

a=['a1','a2','a3']
b=['b1','b2','b3']
c=['c1','c2','c3']
for a, b, c in zip(a, b, c):
    print(a, b, c)



exit()