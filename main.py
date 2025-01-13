import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from enum import Enum
def angle_between(p1, p2, p3):
    p1 = p1.squeeze()
    p2 = p2.squeeze()
    p3 = p3.squeeze()
    # 计算 p1->p2 和 p2->p3 的夹角
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
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
    find_corners = None
    nums = len(corners)
    max_score = 0
    rec_score = 0
    for i1 in range(nums - 3):
        for i2 in range(i1 + 1, nums - 2):
            for i3 in range(i2 + 1, nums - 1):
                for i4 in range(i3 + 1, nums):
                    points = np.array([corners[i1], corners[i2], corners[i3], corners[i4]])
                    convexhull = cv2.convexHull(points)
                    if convexhull.shape[0] != 4:
                        continue
                    rec = rectangle_score(convexhull)
                    area = area_score(convexhull)
                    # print(area)
                    if area > max_score and rec < 20:
                        find_corners = convexhull
                        max_score = area
                        rec_score = rec
    return find_corners, max_score, rec_score


piece1 = r'D:\Users\hgq\Pictures\test\1.png'
piece2 = r'D:\Users\hgq\Pictures\test\2.png'

p1 = cv2.imread(piece1)
p2 = cv2.imread(piece2)
p1 = cv2.resize(p1, (700, 600))
p2 = cv2.resize(p2, (700, 600))

p1_gray = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)
p2_gray = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY)

p1_contours, _ = cv2.findContours(p1_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
p2_contours, _ = cv2.findContours(p2_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

p1_contour_bg = np.zeros_like(p1_gray)
p2_contour_bg = np.zeros_like(p2_gray)
cv2.drawContours(p1_contour_bg, p1_contours, -1, (255, 255, 255), cv2.FILLED)
cv2.drawContours(p2_contour_bg, p2_contours, -1, (255, 255, 255), cv2.FILLED)

p1_rgb = cv2.cvtColor(p1_contour_bg, cv2.COLOR_GRAY2BGR)
p2_rgb = cv2.cvtColor(p2_contour_bg, cv2.COLOR_GRAY2BGR)

# Shi-Tomasi 角点检测
p1_corners = cv2.goodFeaturesToTrack(p1_contour_bg, maxCorners=15, qualityLevel=0.1, minDistance=10)
p1_corners = p1_corners.astype(np.int32)
p2_corners = cv2.goodFeaturesToTrack(p2_contour_bg, maxCorners=15, qualityLevel=0.1, minDistance=10)
p2_corners = p2_corners.astype(np.int32)

# # 在图像上绘制角点
# for index in range(len(corners)):
#     x, y = corners[index].ravel()
#     cv2.circle(p1, (x, y), 5, (0, 255, 0), -1)
#     #cv2.putText(p1,str(index),(x,y+10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
p1_find_corners, score, rec_score = corners_filter(p1_corners)
p2_find_corners, _, _ = corners_filter(p2_corners)
p1_copy=p1.copy()
for corner in p1_find_corners:
    x, y = corner.ravel()
    cv2.circle(p1_copy, (x, y), 5, (0, 0, 255), -1)
    #cv2.putText(p1,f"{x},{y}",(x,y+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
p2_copy=p2.copy()
for corner in p2_find_corners:
    x, y = corner.ravel()
    cv2.circle(p2_copy, (x, y), 5, (0, 0, 255), -1)
cv2.imshow("p1_corners", p1_copy)
# cv2.imshow("p2_corners", p2_copy)

def sort_points(points):
    # points N,1,2
    # 先按照第一个坐标排序，再安装第二个坐标排序
    #sorted_indices=np.lexsort((points[:,0,1],points[:,0,0]))
    sorted_indices=np.argsort((points[:,0,0]))
    sorted_points=points[sorted_indices]
    return sorted_points

# 按凸包顺序排列的角点
p1_corners=p1_find_corners
p2_corners=p2_find_corners

def rotate_2_vertical(points,img):
    sorted=sort_points(points)
    left1=sorted[0].ravel()
    left2=sorted[1].ravel()
    # 计算两点连线的角度（弧度）
    delta_x = left2[0] - left1[0]
    delta_y = left2[1] - left1[1]
    angle_radians = math.atan2(delta_y, delta_x)
    # 将角度转换为度，并计算旋转角度
    angle_degrees = math.degrees(angle_radians)
    # 逆时针旋转角度
    if angle_degrees<0:
        rotation_angle = angle_degrees + 90
    else:
        rotation_angle = angle_degrees - 90
    # 获取图像的中心
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    # 构建旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # 旋转图像
    rotated_image = cv2.warpAffine(img, rotation_matrix, (w, h))

    points=points.reshape(4, 2)
    ones = np.ones((4, 1))
    points_homogeneous = np.hstack([points, ones])
    rotated_points = np.dot(points_homogeneous, rotation_matrix.T).astype(np.int32)
    #rotated_points_reshaped = rotated_points.reshape(-1, 1, 2)
    return rotated_image,rotated_points

p1_rotated,p1_rotated_points = rotate_2_vertical(p1_corners, p1)
p2_rotated,p2_rotated_points = rotate_2_vertical(p2_corners, p2)

#cv2.imshow("p1_rotated", p1_rotated)
p1_edge=cv2.Canny(p1_rotated, 50, 150)
cv2.imshow("p1_edge", p1_edge)

p1_edge_points=np.column_stack(np.where(p1_edge == 255))
p1_edge_points=p1_edge_points[...,::-1]

# points 坐标都是像素坐标x,y,x为横坐标，y为纵坐标
def edge_label(img,edge_points, corner_points):
    height=img.shape[0]
    width=img.shape[1]
    mask=np.zeros((height,width))
    for point in edge_points:
        distance01=cal_point_line_distance(corner_points[0],corner_points[1],point)
        distance12=cal_point_line_distance(corner_points[2],corner_points[1],point)
        distance23=cal_point_line_distance(corner_points[2],corner_points[3],point)
        distance30=cal_point_line_distance(corner_points[0],corner_points[3],point)

        min_distance=min(distance01,distance12,distance23,distance30)
        if distance01==min_distance:
            mask[point[1],point[0]]=1
        if distance12==min_distance:
            mask[point[1],point[0]]=2
        if distance23==min_distance:
            mask[point[1],point[0]]=3
        if distance30==min_distance:
            mask[point[1],point[0]]=4

    # show=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    # show[mask==1]=(255,0,0)
    # show[mask==2]=(0,255,0)
    # show[mask==3]=(0,0,255)
    # show[mask==4]=(255,255,0)
    #
    # cv2.imshow("edge_label", show)
    return mask

def cal_point_line_distance(line_point1,line_point2,point):
    x1, y1 = line_point1
    x2, y2 = line_point2
    x3, y3 = point
    a = (y2 - y1)
    b = (x1 - x2)
    c = (x2 * y1 - x1 * y2)
    d = abs(a * x3 + b * y3 + c) / math.sqrt(a * a + b * b)
    return d

p1_edge_label=edge_label(p1,p1_edge_points,p1_rotated_points)
p1_edge1=cv2.bitwise_and(p1_edge,p1_edge, mask=(p1_edge_label==1).astype(np.uint8))
p1_edge2=cv2.bitwise_and(p1_edge,p1_edge, mask=(p1_edge_label==2).astype(np.uint8))
p1_edge3=cv2.bitwise_and(p1_edge,p1_edge, mask=(p1_edge_label==3).astype(np.uint8))
p1_edge4=cv2.bitwise_and(p1_edge,p1_edge, mask=(p1_edge_label==4).astype(np.uint8))

cv2.imshow("p1_edge1", p1_edge1)
cv2.imshow("p1_edge2", p1_edge2)
cv2.imshow("p1_edge3", p1_edge3)
cv2.imshow("p1_edge4", p1_edge4)

class Piece:
    def __init__(self,img):
        self.img=img
        self.id=1
        self.edge_points=[]
        self.corner_points=[]
        self.edge1=[]

class Edge:
    def __init__(self,corner1,corner2,points,edge_type):
        self.corner1=corner1
        self.corner2=corner2
        self.points=points
        self.edge_type=edge_type

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




cv2.waitKey(0)
cv2.destroyAllWindows()
