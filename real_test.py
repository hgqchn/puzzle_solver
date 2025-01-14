import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils






if __name__=="__main__":

    # 从照片中提取拼图碎片的形状轮廓
    img=r'D:\Users\hgq\Pictures\test\real.jpeg'
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, ksize=5)
    thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.blur(thresh, ksize=(3, 3))
    #utils.show_img("thresh", thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    binary=np.zeros_like(thresh)
    cv2.drawContours(binary, contours, -1, (255, 255, 255), cv2.FILLED)

    pieces=[]

    binary_bgr=binary.copy()
    binary_bgr=cv2.cvtColor(binary_bgr, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        offset=10
        x=max(x-offset, 0)
        y=max(y-offset, 0)
        w=min(w+2*offset, binary_bgr.shape[1]-x)
        h=min(h+2*offset, binary_bgr.shape[0]-y)
        cv2.rectangle(binary_bgr, (x, y), (x + w, y + h), (0, 255, 0), 10)
        piece=binary[y:y+h,x:x+w]
        piece_obj=utils.Piece(piece)
        cv2.putText(binary_bgr, f"{piece_obj.id}", (x+w//2, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 5,thickness=10, color=(0,0,255))
        pieces.append(piece_obj)
    #utils.show_img("thresh1", binary_bgr)

    timer=utils.Timer()
    piece0=pieces[0]
    image=piece0.binary
    image = cv2.GaussianBlur(image, (5,5), 0)
    timer.start()
    corners = cv2.goodFeaturesToTrack(image, maxCorners=0, qualityLevel=0.03, minDistance=10,blockSize=5)
    corners = corners.astype(np.int32)
    timer.stop()
    print(f"corners detection time: {timer.get_last():.3f} seconds")
    timer.start()
    rect_corners, s1, s2 = utils.corners_filter(corners)
    timer.stop()
    print(f"corners filtering time: {timer.get_last():.3f} seconds")
    cv2.imshow("pieces", piece0.show_corners(corners))
    cv2.imshow("pieces1", piece0.show_corners(rect_corners))
    #piece0.get_corners()

    #cv2.imshow("pieces", piece0.show_corners(piece0.corners))



    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pass