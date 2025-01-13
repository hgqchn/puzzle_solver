import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
img=r'D:\Users\hgq\Pictures\test\3.jpg'

img=cv2.imread(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray   = cv2.medianBlur(gray, ksize=5)
thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.blur(thresh, ksize=(3, 3))

# Shi-Tomasi 角点检测
corners = cv2.goodFeaturesToTrack(thresh, maxCorners=15, qualityLevel=0.1, minDistance=10)
corners = corners.astype(np.int32)

# 在图像上绘制角点
for index in range(len(corners)):
    x, y = corners[index].ravel()
    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
    cv2.putText(img,str(index),(x,y+10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)

find_corners,score,rec_score=utils.corners_filter(corners)
if type(find_corners) != type(None):
    for corner in find_corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("test",img)
else:
    print("No corners found.")


cv2.imshow("corners", img)
#cv2.imshow("thresh", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()