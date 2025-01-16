import os
import sys
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm

from utils import area_score

feature_detector= cv2.ORB_create(
    #nfeatures=500,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=20, #31
    firstLevel=0,
    WTA_K=4,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=51, #31
     )
#feature_detector=cv2.SIFT_create(nOctaveLayers=4,edgeThreshold=5)
#bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
def get_edge_mask(edge):
    edge_binary=edge.edge_binary
    edge_points=edge.edge_points
    edge_points=np.expand_dims(edge_points, axis=1)
    x, y, w, h = cv2.boundingRect(edge_points)
    offset = 6
    x = max(x - offset, 0)
    y = max(y - offset, 0)
    w = min(w + 2 * offset, edge_binary.shape[1] - x)
    h = min(h + 2 * offset, edge_binary.shape[0] - y)
    # mask=np.zeros_like(edge_binary)
    # cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return x,y,w,h
def match_two_edges(edge1,edge2):
    x1,y1,w1,h1=get_edge_mask(edge1)
    x2,y2,w2,h2=get_edge_mask(edge2)
    binary1=edge1.piece.binary
    binary2=edge2.piece.binary

    binary1=cv2.bilateralFilter(binary1, 30, 75,50)
    binary2=cv2.bilateralFilter(binary2, 30, 75,50)

    # binary1_bgr=edge1.piece.binary_bgr
    # binary2_bgr=edge2.piece.binary_bgr
    # binary2_bgr_inverted=cv2.bitwise_not(binary2_bgr)
    # cv2.rectangle(binary1_bgr, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 5)
    # cv2.rectangle(binary2_bgr_inverted, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 5)
    # cv2.imshow("binary1_bgr", binary1_bgr)
    # cv2.imshow("binary2_bgr", binary2_bgr_inverted)
    mask1=np.zeros_like(binary1)
    cv2.rectangle(mask1, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 255), -1)
    mask2=np.zeros_like(binary2)
    cv2.rectangle(mask2, (x2, y2), (x2 + w2, y2 + h2), (255, 255, 255), -1)
    kp1, des1 = feature_detector.detectAndCompute(binary1, mask1)
    binary2_inverted = cv2.bitwise_not(binary2)
    kp2, des2 = feature_detector.detectAndCompute(binary2_inverted, mask2)
    matches = bf.match(des1, des2)
    # 根据匹配的距离进行排序
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 4:
        return False
    # 提取匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # # 使用 RANSAC 计算仿射矩阵并剔除错误匹配点
    #H, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC,ransacReprojThreshold=20,maxIters=500,confidence=0.75,refineIters=0)
    H, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,ransacReprojThreshold=30,maxIters=500,confidence=0.7,refineIters=0)
    # 绘制剔除错误匹配点后的匹配结果
    matches_mask = mask.ravel().tolist()
    filtered_matches = [matches[i] for i in range(len(matches)) if matches_mask[i]]
    # 绘制匹配结果
    # result = cv2.drawMatches(binary1, kp1, binary2_inverted, kp2, matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # result_ = cv2.drawMatches(binary1, kp1, binary2_inverted, kp2, filtered_matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("match_result", result)
    # cv2.imshow("match_result_filtered", result_)

    # 提取线性部分
    linear_part = H[:2, :2]
    # 计算行列式
    scale = np.linalg.det(linear_part)
    scale_threshold = 0.65
    if abs(abs(scale) - 1) > scale_threshold:
        return False
    # 提取匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)

    height = binary2.shape[0]+binary1.shape[0]
    width =binary2.shape[1]+binary1.shape[1]

    # 构建平移变换矩阵
    param1=-H[0, 2] if H[0,2] <0 else 0
    param2=-H[1,2] if H[1,2] <0 else 0
    T = np.array([[1, 0, param1], [0, 1, param2]], dtype=np.float32)
    new_H = T @ np.vstack([H, [0, 0, 1]])

    p1_to_p2 = cv2.warpAffine(binary1, new_H, (width, height))
    # cv2.imshow("warped_image_1", edge1_to_edge2)

    p2_transformed = cv2.warpAffine(binary2, T, (width, height))
    # cv2.imshow("warped_image_2", edge2_transformed)

    p1_to_p2[p2_transformed == 255] = 255

    src_pts_trans = cv2.transform(src_pts, new_H)
    dst_pts_trans = cv2.transform(dst_pts, T)

    dis_score = 0
    for index in range(len(src_pts_trans )):
        p1 = src_pts_trans[index].ravel()
        p2 = dst_pts_trans[index].ravel()
        distance = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        dis_score = dis_score + distance
    dis_score = dis_score / len(src_pts_trans)
    area_score=0
    area1 = np.count_nonzero(binary1)
    area1_scaled=area1*scale
    area2 = np.count_nonzero(binary2)
    area_=np.count_nonzero(p1_to_p2)
    area_score=np.abs(area1_scaled+area2-area_)
    if area_score>150000:
        return False
    #area_score=area_/(area1+area2)
    return p1_to_p2, area_score,dis_score




