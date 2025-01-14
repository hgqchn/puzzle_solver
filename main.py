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

p1_binary = np.zeros_like(p1_gray)
p2_binary = np.zeros_like(p2_gray)
cv2.drawContours(p1_binary, p1_contours, -1, (255, 255, 255), cv2.FILLED)
cv2.drawContours(p2_binary, p2_contours, -1, (255, 255, 255), cv2.FILLED)

p1_bgr = cv2.cvtColor(p1_binary, cv2.COLOR_GRAY2BGR)
p2_bgr = cv2.cvtColor(p2_binary, cv2.COLOR_GRAY2BGR)

# Shi-Tomasi 角点检测
p1_corners = cv2.goodFeaturesToTrack(p1_binary, maxCorners=15, qualityLevel=0.1, minDistance=10)
p1_corners = p1_corners.astype(np.int32)
p2_corners = cv2.goodFeaturesToTrack(p2_binary, maxCorners=15, qualityLevel=0.1, minDistance=10)
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
#cv2.imshow("p1_corners", p1_copy)
# cv2.imshow("p2_corners", p2_copy)

def sort_points_clockwise(points):
    """
    对给定的四个点进行排序，确保形成凸包并按顺时针顺序排列。
    起点为 x 最小的点（如果 x 相同，则 y 最小）。

    参数:
    points: np.ndarray - 形状为 (4, 2) 的数组，每行是一个点的 (x, y) 坐标。

    返回:
    sorted_points: np.ndarray - 排序后的点数组。
    """
    # 找到起始点：x 最小，如果 x 相同则 y 最小
    start_idx = np.lexsort((points[:, 1], points[:, 0]))[0]
    start = points[start_idx]

    # 删除起始点后计算其他点的极角
    remaining_points = np.delete(points, start_idx, axis=0)
    angles = np.arctan2(remaining_points[:, 1] - start[1], remaining_points[:, 0] - start[0])

    # # 计算距离以区分相同角度的点
    # distances = np.linalg.norm(remaining_points - start, axis=1)

    # 按角度和距离排序
    sorted_indices = np.argsort(angles)
    #sorted_indices = np.lexsort((distances, angles))
    sorted_remaining_points = remaining_points[sorted_indices]

    # 将起始点添加到排序结果的开头
    sorted_points = np.vstack((start, sorted_remaining_points))

    return np.expand_dims(sorted_points, axis=1)


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

# 顺时针排列，且起点为左上点
p1_corners=sort_points_clockwise(p1_corners.squeeze())
p2_corners=sort_points_clockwise(p2_corners.squeeze())
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

p1_rotated,p1_corners = rotate_2_vertical(p1_corners, p1_binary)
p2_rotated,p2_corners = rotate_2_vertical(p2_corners, p2_binary)

#cv2.imshow("p1_rotated", p1_rotated)
p1_edge=cv2.Canny(p1_rotated, 50, 150)
p2_edge=cv2.Canny(p2_rotated, 50, 150)
# cv2.imshow("p1_edge", p1_edge)
# cv2.imshow("p2_edge", p2_edge)

p1_edge_points=np.column_stack(np.where(p1_edge == 255))
p1_edge_points=p1_edge_points[...,::-1]

p2_edge_points=np.column_stack(np.where(p2_edge == 255))
p2_edge_points=p2_edge_points[...,::-1]

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

p1_edge_label=edge_label(p1,p1_edge_points,p1_corners)
p1_edge1=cv2.bitwise_and(p1_edge,p1_edge, mask=(p1_edge_label==1).astype(np.uint8))
p1_edge2=cv2.bitwise_and(p1_edge,p1_edge, mask=(p1_edge_label==2).astype(np.uint8))
p1_edge3=cv2.bitwise_and(p1_edge,p1_edge, mask=(p1_edge_label==3).astype(np.uint8))
p1_edge4=cv2.bitwise_and(p1_edge,p1_edge, mask=(p1_edge_label==4).astype(np.uint8))
p1_edges=[p1_edge1, p1_edge2, p1_edge3, p1_edge4]
#cv2.imshow("p1_edge1", p1_edge1)
#cv2.imshow("p1_edge2", p1_edge2)
# cv2.imshow("p1_edge3", p1_edge3)
# cv2.imshow("p1_edge4", p1_edge4)

p2_edge_label=edge_label(p2,p2_edge_points,p2_corners)
p2_edge1=cv2.bitwise_and(p2_edge,p2_edge, mask=(p2_edge_label==1).astype(np.uint8))
p2_edge2=cv2.bitwise_and(p2_edge,p2_edge, mask=(p2_edge_label==2).astype(np.uint8))
p2_edge3=cv2.bitwise_and(p2_edge,p2_edge, mask=(p2_edge_label==3).astype(np.uint8))
p2_edge4=cv2.bitwise_and(p2_edge,p2_edge, mask=(p2_edge_label==4).astype(np.uint8))
p2_edges=[p2_edge1, p2_edge2, p2_edge3, p2_edge4]
# cv2.imshow("p2_edge1", p2_edge1)
#cv2.imshow("p2_edge2", p2_edge2)
# cv2.imshow("p2_edge3", p2_edge3)
#cv2.imshow("p2_edge4", p2_edge4)

# cv2.imshow("p1_",p1_contour_bg)
# cv2.imshow("p2_",p2_contour_bg)

p1_edge2_points=np.column_stack(np.where(p1_edge2 == 255))
p1_edge2_points=p1_edge2_points[...,::-1]


p2_edge4_points=np.column_stack(np.where(p2_edge4 == 255))
p2_edge4_points=p2_edge4_points[...,::-1]


p1_edge2_points=np.expand_dims(p1_edge2_points, axis=1)
#cv2.imshow("p1_edge2",p1_edge2)
x, y, w, h = cv2.boundingRect(p1_edge2_points)
offset=20
x=max(x-offset, 0)
y=max(y-offset, 0)
w=min(w+2*offset, p1_bgr.shape[1]-x)
h=min(h+2*offset, p1_bgr.shape[0]-y)
p1_mask=np.zeros_like(p1_binary)
cv2.rectangle(p1_mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

p2_edge4_points=np.expand_dims(p2_edge4_points, axis=1)
x, y, w, h = cv2.boundingRect(p2_edge4_points)
offset=20
x=max(x-offset, 0)
y=max(y-offset, 0)
w=min(w+2*offset, p1_bgr.shape[1]-x)
h=min(h+2*offset, p1_bgr.shape[0]-y)
p2_mask=np.zeros_like(p2_binary)
cv2.rectangle(p2_mask, (x, y), (x + w, y + h), (255, 255, 255), -1)



orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(p1_rotated,p1_mask)
p2_rotated_inverted=cv2.bitwise_not(p2_rotated)
kp2, des2 = orb.detectAndCompute(p2_rotated_inverted,p2_mask)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 根据匹配的距离进行排序
matches = sorted(matches, key=lambda x: x.distance)


# 提取匹配点坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 添加角点作为匹配点
p1_corner_1=p1_corners[1].astype(np.float32).reshape(-1, 1, 2)
p1_corner_2=p1_corners[2].astype(np.float32).reshape(-1, 1, 2)
p2_corner_1=p2_corners[3].astype(np.float32).reshape(-1, 1, 2)
p2_corner_2=p2_corners[0].astype(np.float32).reshape(-1, 1, 2)
p1_edge_corners=[p1_corner_1,p1_corner_2]
p2_edge_corners=[p2_corner_1, p2_corner_2]
kp1=list(kp1)
kp2=list(kp2)
src_pts = np.vstack((src_pts, p1_corner_1))
dst_pts = np.vstack((dst_pts, p2_corner_2))
keypoint1 = cv2.KeyPoint(p1_corner_1[0][0][0], p1_corner_1[0][0][1], 5)
keypoint2 = cv2.KeyPoint(p2_corner_2[0][0][0], p2_corner_2[0][0][1], 5)
manual_match = cv2.DMatch(len(kp1), len(kp2), 0)
kp1.append(keypoint1)
kp2.append(keypoint2)
matches = matches + [manual_match]
src_pts = np.vstack((src_pts, p1_corner_2))
dst_pts = np.vstack((dst_pts, p2_corner_1))
keypoint1 = cv2.KeyPoint(p1_corner_2[0][0][0], p1_corner_2[0][0][1], 5)
keypoint2 = cv2.KeyPoint(p2_corner_1[0][0][0], p2_corner_1[0][0][1], 5)
manual_match = cv2.DMatch(len(kp1), len(kp2), 0)
kp1.append(keypoint1)
kp2.append(keypoint2)
matches = matches + [manual_match]

# 绘制匹配结果
result = cv2.drawMatches(p1_binary, kp1, p2_rotated_inverted, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#cv2.imshow("result",result)
# 使用 RANSAC 计算单应性矩阵并剔除错误匹配点
#H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
# # 使用 RANSAC 计算仿射矩阵并剔除错误匹配点
H, mask=cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
matches_mask = mask.ravel().tolist()
filtered_matches = [matches[i] for i in range(len(matches)) if matches_mask[i]]

# 绘制剔除错误匹配点后的匹配结果
result = cv2.drawMatches(p1_binary, kp1, p2_rotated_inverted, kp2,filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#cv2.imshow("result1",result)

# 提取匹配点坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)


height=p2_binary.shape[0]
width=p2_binary.shape[1]

T = np.array([[1, 0, -H[0,2]], [0, 1, -H[1,2]]], dtype=np.float32)
new_H= T @ np.vstack([H, [0, 0, 1]])
# warped_image = cv2.warpPerspective(p1_binary, H, (width * 2, height))
p1_warped_image = cv2.warpAffine(p1_rotated, new_H, (width*2, height*2))
#cv2.imshow("warped_image", p1_warped_image)
p2_warped_image = cv2.warpAffine(p2_rotated, T, (width * 2, height * 2))
#cv2.imshow("p2_warped_image", p2_warped_image)
p1_warped_image[p2_warped_image==255]=255

src_trans=cv2.transform(src_pts, new_H)
dst_trans=cv2.transform(dst_pts, T)

dis=0
for index in range(len(src_trans)):
    p1=src_trans[index].ravel()
    p2=dst_trans[index].ravel()
    distance=np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
    dis=dis+distance


cv2.imshow("whole_image",p1_warped_image)

# # 将第二张图像拼接到透视变换后的图像上
# warped_image[0:height, 0:width] = p2_rotated_inverted
# cv2.imshow("result", warped_image)
# res=match_and_stich(p1_edges[1],p2_edges[3],p1_contour_bg,p2_contour_bg)
# cv2.imshow('res',res)
#
# result_list=[]
# for img1 in p1_edges:
#     for img2 in p2_edges:
#         result=match_and_stich(img1,img2)
#         result_list.append(result)
#
# fig,axes=plt.subplots(4,4,figsize=(200,100))
# for i in range(16):
#     ax=axes[i//4,i%4]
#     ax.axis("off")
#     ax.imshow(result_list[i],cmap='gray')
# plt.savefig("res.png")
# plt.show(block=True)




cv2.waitKey(0)
cv2.destroyAllWindows()
