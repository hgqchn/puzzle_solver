from enum import Enum

import cv2
import numpy as np
import math
import time

window_width=600
window_height=800

appro_piece_width=600
appro_piece_height=600
appro_edge_lenth=(appro_piece_width+appro_piece_height)/2

class Timer():
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        #self.tik = time.time()
        self.tik = time.perf_counter()

    # stop之后需要start
    def stop(self):
        #self.times.append(time.time() - self.tik)
        self.times.append(time.perf_counter() - self.tik)
        return self.times[-1]
    def get_last(self):
        return self.times[-1] if self.times else 0
    def avg(self):
        return sum(self.times) / len(self.times) if self.times else 0

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

    def reset(self):
        self.times.clear()


def show_img(winname,data):
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, window_width, window_height)
    cv2.imshow(winname, data)

def angle_between(p1, p2, p3):
    p1=p1.squeeze()
    p2=p2.squeeze()
    p3=p3.squeeze()
    # 计算 p1->p2 和 p2->p3 的夹角
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    cos_angle = dot / (mag1 * mag2)
    angle = math.acos(cos_angle) * 180 / math.pi  # 转换为角度
    return angle

def rectangle_score(points):
    # 按顺序计算四个点的夹角
    angles = [
        angle_between(points[i - 1], points[i], points[(i + 1) % 4])
        for i in range(4)
    ]
    # 计算矩形性评分（偏离90度的总误差越小越好）
    score = sum(abs(angle - 90) for angle in angles)
    return score

def area_score(points):

    area = cv2.contourArea(points)
    return area
def corners_filter(corners):
    find_corners=None
    nums=len(corners)
    max_score=0
    rec_score=0
    distance_threshold=appro_edge_lenth*0.6
    rec_threshold=40
    for i1 in range(nums-3):
        p1=corners[i1]
        for i2 in range(i1+1,nums-2):
            p2=corners[i2]
            if distance_point(p1.ravel(),p2.ravel())<distance_threshold:
                continue
            for i3 in range(i2+1,nums-1):
                p3=corners[i3]
                if distance_point(p2.ravel(),p3.ravel())<distance_threshold or distance_point(p1.ravel(),p3.ravel())<distance_threshold:
                    continue
                for i4 in range(i3+1,nums):
                    p4=corners[i4]
                    if distance_point(p3.ravel(),p4.ravel())<distance_threshold or distance_point(p2.ravel(),p4.ravel())<distance_threshold\
                        or distance_point(p1.ravel(),p4.ravel())<distance_threshold:
                        continue
                    points=np.array([p1,p2,p3,p4])
                    convexhull = cv2.convexHull(points)
                    if convexhull.shape[0]!=4:
                        continue
                    rec=rectangle_score(convexhull)
                    if rec>rec_threshold:
                        continue
                    area=area_score(convexhull)
                    if area>max_score:
                        find_corners=points
                        max_score=area
                        rec_score=rec
    return find_corners,max_score,rec_score

def distance_point(p1,p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
def find_puzzle_corners(img,show=True):
    img_copy=img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask=np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255,255,255), cv2.FILLED)

    # Shi-Tomasi 角点检测
    corners = cv2.goodFeaturesToTrack(mask, maxCorners=0, qualityLevel=0.01, minDistance=10)
    corners = corners.astype(np.int32)

    find_corners, score, rec_score = corners_filter(corners)


    if type(find_corners) == type(None):
        raise TypeError("no corners found")
    for corner in find_corners:
        x, y = corner.ravel()
        cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
    if show:
        cv2.imshow("corners", img_copy)

    return find_corners



def check_is_None(obj):
    if type(obj)==type(None):
        return True
    else:
        return False

class EdgeType(Enum):
    CONVEX=1
    CONCAVE=2
    FLAT=3

class PieceType(Enum):
    SINGLE_VEX=1
    DOUBLE_VEX_CAVE=2
    DOUBLE_VEX_FLAT=3
    DOUBLE_CAVE_FLAT=4
    TRIPLE_VEX=5
    QUAD_VEX=6
    QUAD_CAVE=7

class Piece:
    _id_counter=0
    def __init__(self,img):
        """

        :param img: binary image of the piece
        """
        Piece._id_counter+=1
        self.binary=img
        self.binary_bgr=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.id=Piece._id_counter
        self.edge_points=None
        self.corners=None
        self.edges=[]
        self.piece_type=None

    def get_rect_corners(self):
        # Shi-Tomasi 角点检测
        corners = cv2.goodFeaturesToTrack(self.binary, maxCorners=0, qualityLevel=0.05, minDistance=200)
        if check_is_None(corners):
            raise ValueError("No corners found")
        corners = corners.astype(np.int32)
        rect_corners, _, _ = corners_filter(corners)
        self.corners=rect_corners

    def show_corners(self,corners):
        if check_is_None(corners):
            raise ValueError("No corners")
        img = self.binary_bgr.copy()
        for index in range(len(corners)):
            x, y = corners[index].ravel()
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        return img

class Edge:
    def __init__(self,corner1,corner2,points,edge_type):
        self.corner1=corner1
        self.corner2=corner2
        self.edge_points=points
        self.edge_type=edge_type



