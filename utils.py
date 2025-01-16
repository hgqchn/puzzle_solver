from enum import Enum

import cv2
import numpy as np
import math
import time
import pickle
import os

from contourpy.array import concat_points_or_none

window_width = 600
window_height = 800

appro_piece_width = 600
appro_piece_height = 600
appro_edge_lenth = (appro_piece_width + appro_piece_height) / 2


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        # self.tik = time.time()
        self.tik = time.perf_counter()

    # stop之后需要start
    def stop(self):
        # self.times.append(time.time() - self.tik)
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


class window_handler:
    def __init__(self, winname, image):
        self.winname = winname
        self.image = image
        self.scale = 1.0
        self.max_scale = 5

        # 偏移量（用于拖动）
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.drag_start = (0, 0)

        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname, window_width, window_height)
        cv2.setMouseCallback(winname, self.mouse_callback)
        cv2.imshow(winname, self.image)

    def show_image(self):
        """根据缩放和偏移量显示图像"""
        h, w = self.image.shape[:2]
        display = np.zeros_like(self.image)  # 创建一个空白画布，尺寸与图像尺寸一致
        center_y = h // 2
        center_x = w // 2

        # 缩放图像
        resized_image = cv2.resize(
            self.image,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_LINEAR,
        )
        h_resized = resized_image.shape[0]
        w_resized = resized_image.shape[1]

        # 图像在画布上的坐标
        img_x1 = center_x - w_resized // 2
        img_y1 = center_y - h_resized // 2
        img_x1 = img_x1 + self.offset_x
        img_y1 = img_y1 + self.offset_y
        img_x2 = img_x1 + w_resized
        img_y2 = img_y1 + h_resized

        # 画布上显示区域的坐标
        x1 = 0
        x2 = w
        y1 = 0
        y2 = h

        x1_in_img = 0
        y1_in_img = 0
        x2_in_img = 0
        y2_in_img = 0
        # 显示区域在图像上的坐标
        if img_x1 < 0 and img_x2 > w:
            x1_in_img = -img_x1
            x2_in_img = x1_in_img + w
            x1 = 0
            x2 = w
        elif img_x1 < 0 and 0 < img_x2 <= w:
            x1_in_img = -img_x1
            x2_in_img = x1_in_img + img_x2
            x1 = 0
            x2 = img_x2
        elif 0 <= img_x1 < w:
            x1_in_img = 0
            x2_in_img = w - img_x1
            x1 = img_x1
            x2 = w
        else:
            x1_in_img = 0
            x2_in_img = 0

        if img_y1 < 0 and img_y2 > h:
            y1_in_img = -img_y1
            y2_in_img = y1_in_img + h
            y1 = 0
            y2 = h
        elif img_y1 < 0 and 0 < img_y2 <= h:
            y1_in_img = -img_y1
            y2_in_img = y1_in_img + img_y2
            y1 = 0
            y2 = img_y2
        elif 0 <= img_y1 < h:
            y1_in_img = 0
            y2_in_img = h - img_y1
            y1 = img_y1
            y2 = h
        else:
            y1_in_img = 0
            y2_in_img = 0

        # 将放大后的图像显示到窗口
        if x1_in_img > 0 or x2_in_img > 0:
            display[y1:y2, x1:x2] = resized_image[y1_in_img:y2_in_img, x1_in_img:x2_in_img]
        cv2.imshow(self.winname, display)

    def mouse_callback(self, event, x, y, flags, param):

        # 检测滚轮事件
        if event == cv2.EVENT_MOUSEWHEEL:

            if flags > 0:  # 滚轮向上
                self.scale *= 1.1  # 放大
            else:
                self.scale /= 1.1  # 缩小
            # 限制缩放比例范围
            self.scale = max(1.0, min(self.max_scale, self.scale))
            # 显示缩放后的图像
            self.show_image()

        elif event == cv2.EVENT_LBUTTONDOWN:
            # 左键按下：开始拖动
            self.dragging = True
            self.drag_start = (x, y)
            self.show_image()
        elif event == cv2.EVENT_MOUSEMOVE:
            # 鼠标移动：更新偏移量
            if self.dragging:
                dx, dy = x - self.drag_start[0], y - self.drag_start[1]
                self.offset_x += dx
                self.offset_y += dy
                self.drag_start = (x, y)  # 更新拖动起点
                self.show_image()

        elif event == cv2.EVENT_LBUTTONUP:
            # 左键释放：停止拖动
            self.dragging = False


def show_img(winname, data):
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, window_width, window_height)
    cv2.imshow(winname, data)


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
    distance_threshold = appro_edge_lenth * 0.6
    rec_threshold = 30
    for i1 in range(nums - 3):
        p1 = corners[i1]
        for i2 in range(i1 + 1, nums - 2):
            p2 = corners[i2]
            if distance_point(p1.ravel(), p2.ravel()) < distance_threshold:
                continue
            for i3 in range(i2 + 1, nums - 1):
                p3 = corners[i3]
                if distance_point(p2.ravel(), p3.ravel()) < distance_threshold or distance_point(p1.ravel(),
                                                                                                 p3.ravel()) < distance_threshold:
                    continue
                for i4 in range(i3 + 1, nums):
                    p4 = corners[i4]
                    if distance_point(p3.ravel(), p4.ravel()) < distance_threshold or distance_point(p2.ravel(),
                                                                                                     p4.ravel()) < distance_threshold \
                            or distance_point(p1.ravel(), p4.ravel()) < distance_threshold:
                        continue
                    points = np.array([p1, p2, p3, p4])
                    convexhull = cv2.convexHull(points)
                    if convexhull.shape[0] != 4:
                        continue
                    rec = rectangle_score(convexhull)
                    if rec > rec_threshold:
                        continue
                    area = area_score(convexhull)
                    if area > max_score:
                        find_corners = points
                        max_score = area
                        rec_score = rec
    return find_corners, max_score, rec_score


def distance_point(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_puzzle_corners(img, show=True):
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), cv2.FILLED)

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


def sort_points_clockwise(points):
    """
    对给定的四个点进行排序，确保形成凸包并按顺时针顺序排列。
    起点为 x 最小的点（如果 x 相同，则 y 最小）。

    参数:
    points: np.ndarray - 形状为 (4, 2) 的数组，每行是一个点的 (x, y) 坐标。

    返回:
    sorted_points: np.ndarray - 排序后的点数组。
    """
    # # 找到起始点：x 最小，如果 x 相同则 y 最小
    # start_idx = np.lexsort((points[:, 1], points[:, 0]))[0]
    # start = points[start_idx]
    distances = np.linalg.norm(points, axis=1)
    start_idx = np.argmin(distances)
    start = points[start_idx]

    # 删除起始点后计算其他点的极角
    remaining_points = np.delete(points, start_idx, axis=0)
    angles = np.arctan2(remaining_points[:, 1] - start[1], remaining_points[:, 0] - start[0])

    # # 计算距离以区分相同角度的点
    # distances = np.linalg.norm(remaining_points - start, axis=1)

    # 按角度和距离排序
    sorted_indices = np.argsort(angles)
    # sorted_indices = np.lexsort((distances, angles))
    sorted_remaining_points = remaining_points[sorted_indices]

    # 将起始点添加到排序结果的开头
    sorted_points = np.vstack((start, sorted_remaining_points))

    return sorted_points


def sort_points(points):
    # points N,2
    # 先按照第一个坐标排序，再安装第二个坐标排序
    # sorted_indices=np.lexsort((points[:,0,1],points[:,0,0]))
    sorted_indices = np.argsort((points[:, 0]))
    sorted_points = points[sorted_indices]
    return sorted_points


def rotate_2_vertical(rect_points, img):
    # rect_points 4,2
    sorted = sort_points(rect_points)
    left1 = sorted[0]
    left2 = sorted[1]
    # 计算两点连线的角度（弧度）
    delta_x = left2[0] - left1[0]
    delta_y = left2[1] - left1[1]
    angle_radians = math.atan2(delta_y, delta_x)
    # 将角度转换为度，并计算旋转角度
    angle_degrees = math.degrees(angle_radians)
    # 逆时针旋转角度
    if angle_degrees < 0:
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

    rect_points = rect_points.reshape(4, 2)
    ones = np.ones((4, 1))
    points_homogeneous = np.hstack([rect_points, ones])
    rotated_points = np.dot(points_homogeneous, rotation_matrix.T).astype(np.int32)
    # rotated_points_reshaped = rotated_points.reshape(-1, 1, 2)
    return rotated_image, rotated_points, rotation_matrix


# points 坐标都是像素坐标x,y,x为横坐标，y为纵坐标
# 根据距离两角点连线的距离判断点是否属于两角点的这条边
def get_edge_label(img, edge_points, corner_points):
    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros((height, width), dtype=np.uint8)
    for point in edge_points:

        distance01 = cal_point_line_distance(corner_points[0], corner_points[1], point)
        distance12 = cal_point_line_distance(corner_points[2], corner_points[1], point)
        distance23 = cal_point_line_distance(corner_points[2], corner_points[3], point)
        distance30 = cal_point_line_distance(corner_points[0], corner_points[3], point)

        point_info=[
            {"edge_id":1,"distance":distance01,"corners":(corner_points[0],corner_points[1])},
            {"edge_id":2,"distance":distance12,"corners":(corner_points[1],corner_points[2])},
            {"edge_id":3,"distance":distance23,"corners":(corner_points[2],corner_points[3])},
            {"edge_id":4,"distance":distance30,"corners":(corner_points[3],corner_points[0])}  # 4 for the last corner point of the rectangle
        ]
        point_info.sort(key=lambda x: x["distance"])
        distance_threshold = 0.1*appro_piece_width
        mask[point[1], point[0]] = point_info[0]["edge_id"]  # assign the edge id to the point with the smallest distance]
        for i,info in enumerate(point_info):
            # 距离两角点连线太远的点暂不标记
            if info["distance"] > distance_threshold:
                mask[point[1], point[0]] = 5
                break
            corner1=info["corners"][0]
            corner2=info["corners"][1]
            p_x,p_y=point
            c1_x,c1_y=corner1
            c2_x,c2_y=corner2
            p_c1=np.array([c1_x-p_x,c1_y-p_y])
            c1_c2=np.array([c2_x-c1_x,c2_y-c1_y])
            c2_p=np.array([p_x-c2_x,p_y-c2_y])

            p_c1_len=np.linalg.norm(p_c1)
            c1_c2_len=np.linalg.norm(c1_c2)
            c2_p_len=np.linalg.norm(c2_p)
            if p_c1_len == 0 or c2_p_len==0:
                mask[point[1], point[0]] = info["edge_id"]
                break
            angle_pc1c2=np.arccos(np.clip(np.dot(-p_c1, c1_c2) / (p_c1_len * c1_c2_len),-1.0,1.0))
            angle_c1c2p=np.arccos(np.clip(np.dot(c1_c2, -c2_p) / (c1_c2_len * c2_p_len),-1.0,1.0))
            # 夹角均为锐角，认为是一条边，但仅对凸在外面的点有效，凹在里面的点还未考虑
            # 凹在里面的点只能考虑根据上文的距离阈值distance_threshold，先不做标记
            if angle_pc1c2 < np.pi / (180/90) and angle_c1c2p < np.pi / (180/90):
                mask[point[1], point[0]] = info["edge_id"]
                break
            else:
                if i==3:
                    mask[point[1], point[0]] = 5

    points_nolabel = np.column_stack(np.where(mask == 5)).tolist()
    while len(points_nolabel) > 0:
        copy = points_nolabel[:]
        for point in copy:
            skip_flag = False
            for i in range(-1, 2):
                for j in range(-1, 2):
                    new_row = min(max(point[0] + i, 0), mask.shape[0] - 1)
                    new_col = min(max(point[1] + j, 0), mask.shape[1] - 1)
                    new_point = [new_row, new_col]
                    if mask[new_point[0], new_point[1]] == 1:
                        mask[point[0], point[1]] = 1
                    elif mask[new_point[0], new_point[1]] == 2:
                        mask[point[0], point[1]] = 2
                    elif mask[new_point[0], new_point[1]] == 3:
                        mask[point[0], point[1]] = 3
                    elif mask[new_point[0], new_point[1]] == 4:
                        mask[point[0], point[1]] = 4
                    else:
                        continue
                    points_nolabel.remove(point)
                    skip_flag = True
                    if skip_flag:
                        break
                if skip_flag:
                    break
            if skip_flag:
                continue
        continue

    edges_corners=[corner_points[[0,1],:],corner_points[[1,2],:],corner_points[[2,3],:],corner_points[[3,0],:]]
    return mask,edges_corners




def show_edge_label(mask):
    show = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    show[mask == 1] = (255, 0, 0) #蓝色 1
    show[mask == 2] = (0, 255, 0) #绿色 2
    show[mask == 3] = (0, 0, 255) #红色 3
    show[mask == 4] = (0, 255, 255) #黄色 4
    show[mask == 5] = (255, 255, 255)  # 未标记为四条边之一
    return show


def cal_point_line_distance(line_point1, line_point2, point):
    x1, y1 = line_point1
    x2, y2 = line_point2
    x3, y3 = point
    a = (y2 - y1)
    b = (x1 - x2)
    c = (x2 * y1 - x1 * y2)
    d = abs(a * x3 + b * y3 + c) / math.sqrt(a * a + b * b)
    return d


def check_is_None(obj):
    if type(obj) == type(None):
        return True
    else:
        return False

def cal_edge_type(center_point,edge_points,corner_points):
    # 传入参数大小为 1,2  N,2  2,2
    corner_center=np.mean(corner_points,axis=0)
    corner_center_distance=np.linalg.norm(corner_center-center_point)
    edge_center=np.mean(edge_points,axis=0)
    edge_center_distance=np.linalg.norm(edge_center-center_point)
    if corner_center_distance<edge_center_distance:
        return EdgeType.CONVEX
    elif corner_center_distance>edge_center_distance:
        return EdgeType.CONCAVE
    else:
        return EdgeType.OTHER

class EdgeType(Enum):
    CONVEX = 1 # 凸的
    CONCAVE = 2
    OTHER = 3 # 未明显判定


class PieceType(Enum):
    SINGLE_VEX_TRIPLE_CAVE = 1
    DOUBLE_VEX_DOUBLE_CAVE = 2
    TRIPLE_VEX_SINGLE_CAVE = 3
    QUAD_VEX = 4
    QUAD_CAVE = 5
    OTHER=6


class Piece:
    _id_counter = 0

    def __init__(self, img=None):
        """

        :param img: binary image of the piece
        """
        Piece._id_counter += 1
        self.binary = img
        self.binary_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.id = Piece._id_counter
        self.edges_binary = None
        self.edges_bgr=None
        self.edge_points = None
        self.corners = None
        self.rect_corners = None
        self.edges = None
        self.piece_type = None
        self.unmatched_convex_edges=[]
        self.unmatched_concave_edges=[]
        self.unmatched_other_edges=[]
        self.convex_edges=[]
        self.concave_edges=[]
        self.other_edges=[]
    def get_corners(self):
        image = cv2.GaussianBlur(self.binary, (5, 5), 0)
        # Shi-Tomasi 角点检测
        corners = cv2.goodFeaturesToTrack(image, maxCorners=0, qualityLevel=0.05, minDistance=10, blockSize=5)
        if check_is_None(corners):
            raise ValueError("No corners found")
        corners = corners.astype(np.int32)
        self.corners = corners

    def get_rect_corners(self):
        if check_is_None(self.corners):
            self.get_corners()
        rect_corners, _, _ = corners_filter(self.corners)
        if check_is_None(rect_corners):
            raise ValueError("No rect_corners found")
        # 顺时针排序，左上点在第一个位置
        rect_corners = sort_points_clockwise(rect_corners.squeeze())
        self.corners = self.corners.squeeze()
        self.rect_corners = rect_corners

    # 旋转图像，并更新binary binary_bgr图像,rect_corners
    def piece_rotate_vertical(self):
        rotated, corners, m = rotate_2_vertical(self.rect_corners, self.binary)
        self.binary = rotated
        self.rect_corners = corners
        self.binary_bgr = cv2.cvtColor(self.binary, cv2.COLOR_GRAY2BGR)

    def get_edges(self):
        edges = cv2.Canny(self.binary, 50, 150)
        self.edges_binary = edges
        edge_points = np.column_stack(np.where(edges == 255))
        # 先x后y
        self.edge_points = edge_points[..., ::-1]
        edge_label,edges_corners = get_edge_label(self.binary, self.edge_points, self.rect_corners)
        #先x后y
        center_point = np.mean(self.rect_corners,axis=0)
        self.edges={}
        for label in range(1,5):
            binary=cv2.bitwise_and(self.edges_binary, self.edges_binary, mask=(edge_label == label).astype(np.uint8))
            #cv2.imshow(f"{label}",binary)
            # 先x后y
            points=np.column_stack(np.where(binary == 255))[..., ::-1]
            # 先x后y
            # 按顺时针顺序的角点
            corners=edges_corners[label-1]
            type=cal_edge_type(center_point,points,corners)
            #print(f"edge {label}: {type}")
            edge=Edge(id=label,piece_id=self.id,piece=self,corner1=corners[0], corner2=corners[1], points=points, edge_type=type,edge_binary=binary)
            self.edges[label]=edge
        visualized=show_edge_label(edge_label)
        self.edges_bgr=visualized
    def process_edges(self):
        self.unmatched_convex_edges=[]
        self.unmatched_concave_edges=[]
        self.unmatched_other_edges=[]
        self.convex_edges=[]
        self.concave_edges=[]
        self.other_edges=[]
        for edge in self.edges.values():
            if edge.edge_type==EdgeType.CONVEX:
                self.convex_edges.append(edge)
                if not edge.is_matched:
                    self.unmatched_convex_edges.append(edge)
            elif edge.edge_type==EdgeType.CONCAVE:
                self.concave_edges.append(edge)
                if not edge.is_matched:
                    self.unmatched_concave_edges.append(edge)
            else:
                self.other_edges.append(edge)
                if not edge.is_matched:
                    self.unmatched_other_edges.append(edge)
        convex=len(self.convex_edges)
        concave=len(self.concave_edges)
        other=len(self.other_edges)
        if convex+concave!=4:
            self.piece_type=PieceType.OTHER
            return
        if convex==0 and concave==4:
            self.piece_type=PieceType.QUAD_CAVE
        elif convex==1 and concave==3:
            self.piece_type=PieceType.TRIPLE_VEX_SINGLE_CAVE
        elif convex==2 and concave==2:
            self.piece_type=PieceType.DOUBLE_VEX_DOUBLE_CAVE
        elif convex==3 and concave==1:
            self.piece_type=PieceType.SINGLE_VEX_TRIPLE_CAVE
        elif convex==4 and concave==0:
            self.piece_type=PieceType.QUAD_VEX
        else:
            raise ValueError("error")

    def get_edge_from_id(self, edge_id):
        return self.edges[edge_id]
    def show_rect_corners(self):
        return self.show_corners(self.rect_corners)

    def show_all_corners(self):
        return self.show_corners(self.corners)

    def show_corners(self, corners):
        if check_is_None(corners):
            raise ValueError("No corners")
        img = self.binary_bgr.copy()
        for index in range(len(corners)):
            x, y = corners[index].ravel()
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        return img

def serialize(obj,save_path):
        filepath=os.path.join(save_path, f"{obj.id}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)

def deserialize(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

class Edge:
    def __init__(self, id,piece_id,piece,corner1, corner2, points, edge_type,edge_binary):
        self.id=id
        self.piece_id = piece_id
        self.piece=piece
        self.corner1 = corner1
        self.corner2 = corner2
        self.edge_points = points
        self.edge_type = edge_type
        self.edge_binary=edge_binary
        self.is_matched=False
        self.matched_edge=None

    def match_edge(self,piece_id, edge_id):
        self.is_matched=True
        self.matched_edge=(piece_id, edge_id)

if __name__ == "__main__":
    # edge_label()
    pass
