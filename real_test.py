import os
import sys
import time

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

    pieces_bound=[]
    offset = 10
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        x=max(x-offset, 0)
        y=max(y-offset, 0)
        w=min(w+2*offset, binary_bgr.shape[1]-x)
        h=min(h+2*offset, binary_bgr.shape[0]-y)
        cv2.rectangle(binary_bgr, (x, y), (x + w, y + h), (0, 255, 0), 10)
        pieces_bound.append((x,y,w,h))
    # 对包围框排序，按照网格形式赋予序号
    sort_height_bias=0
    sort_height_delta=1000
    sort_width_bias=400
    sort_width_delta=1000
    pieces_bound.sort(key=lambda x: ((x[1]-sort_width_bias)//sort_height_delta, (x[0]-sort_width_bias)//sort_width_delta))
    for bound in pieces_bound:
        x,y,w,h=bound
        piece=binary[y:y+h,x:x+w]
        piece_obj=utils.Piece(piece)
        cv2.putText(binary_bgr, f"{piece_obj.id}", (x+w//2, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 5,thickness=10, color=(0,0,255))
        pieces.append(piece_obj)
    utils.show_img("all_pieces", binary_bgr)
    piece1=pieces[0]
    piece2=pieces[1]
    piece3=pieces[2]

    tik=time.perf_counter()
    piece1.get_rect_corners()
    print("get_rect_corners cost: {:.6f}s".format(time.perf_counter()-tik))
    piece1.piece_rotate_vertical()
    edge_label=piece1.get_edges()
    cv2.imshow("edge_label", edge_label)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pass