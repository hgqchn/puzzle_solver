import os
import sys
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm


if __name__=="__main__":
    img_save_path=r'./results'
    # 从照片中提取拼图碎片的形状轮廓
    pieces_num=3
    img=r'D:\Users\hgq\Pictures\test\real.jpeg'
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.medianBlur(gray, ksize=5)
    gray=cv2.bilateralFilter(gray, 15, 75,75)
    thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.blur(thresh, ksize=(5, 5))

    #handler1=utils.window_handler("thresh",thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area=500
    # 过滤掉较小的轮廓
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    if len(contours)!=pieces_num:
        raise ValueError("轮廓数量不等于碎片数量")
    binary=np.zeros_like(thresh)
    cv2.drawContours(binary, contours, -1, (255, 255, 255), cv2.FILLED)

    pieces=[]

    binary_bgr=binary.copy()
    binary_bgr=cv2.cvtColor(binary_bgr, cv2.COLOR_GRAY2BGR)

    pieces_bound=[]
    offset = 50
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
    #utils.show_img("all_pieces", binary_bgr)
    cv2.imwrite(os.path.join(img_save_path, "all_pieces.jpg"), binary_bgr)



    process_time=[]
    with tqdm(total=len(pieces),file=sys.stdout) as pbar:
        for piece in pieces:
            pbar.set_description("Processing")
            pbar.set_postfix_str(f"piece {piece.id}")

            start_time = time.perf_counter()
            try:
                piece.get_rect_corners()
                piece.piece_rotate_vertical()
                piece.get_edges()
                piece.process_edges()
                cv2.imwrite(os.path.join(img_save_path, f"piece_edge_{piece.id}.jpg"), piece.edges_bgr)
                utils.serialize(piece, img_save_path)
                end_time = time.perf_counter()
                process_time.append(end_time - start_time)

            except ValueError as e:
                print(f"{e}")
            pbar.update(1)
        pbar.set_postfix_str(f"cost time: {sum(process_time):.6f}s")


    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pass