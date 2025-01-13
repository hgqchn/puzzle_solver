import cv2
import numpy as np
import math

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
    for i1 in range(nums-3):
        for i2 in range(i1+1,nums-2):
            for i3 in range(i2+1,nums-1):
                for i4 in range(i3+1,nums):
                    points=np.array([corners[i1], corners[i2], corners[i3], corners[i4]])
                    convexhull = cv2.convexHull(points)
                    if convexhull.shape[0]!=4:
                        continue
                    rec=rectangle_score(convexhull)
                    area=area_score(convexhull)
                    #print(area)
                    if area>max_score and rec<20:
                        find_corners=points
                        max_score=area
                        rec_score=rec
    return find_corners,max_score,rec_score

def find_puzzle_corners(img,show=True):
    img_copy=img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask=np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255,255,255), cv2.FILLED)

    # Shi-Tomasi 角点检测
    corners = cv2.goodFeaturesToTrack(mask, maxCorners=15, qualityLevel=0.1, minDistance=10)
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

