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

class Timer:
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

# points 坐标都是像素坐标x,y,x为横坐标，y为纵坐标
def get_edge_label(img,edge_points, corner_points):
    height=img.shape[0]
    width=img.shape[1]
    mask=np.zeros((height,width),dtype=np.uint8)
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
    return mask

def show_edge_label(mask):
    show=np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8)
    show[mask==1]=(255,0,0)
    show[mask==2]=(0,255,0)
    show[mask==3]=(0,0,255)
    show[mask==4]=(255,255,0)
    return show



def cal_point_line_distance(line_point1,line_point2,point):
    x1, y1 = line_point1
    x2, y2 = line_point2
    x3, y3 = point
    a = (y2 - y1)
    b = (x1 - x2)
    c = (x2 * y1 - x1 * y2)
    d = abs(a * x3 + b * y3 + c) / math.sqrt(a * a + b * b)
    return d

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
        self.edges_binary=None
        self.edge_points=None
        self.corners=None
        self.rect_corners=None
        self.edges=[]
        self.piece_type=None

    def get_corners(self):
        image = cv2.GaussianBlur(self.binary, (5, 5), 0)
        # Shi-Tomasi 角点检测
        corners = cv2.goodFeaturesToTrack(image, maxCorners=0, qualityLevel=0.05, minDistance=10,blockSize=5)
        if check_is_None(corners):
            raise ValueError("No corners found")
        corners = corners.astype(np.int32)
        self.corners=corners

    def get_rect_corners(self):
        if check_is_None(self.corners):
            self.get_corners()
        rect_corners, _, _ = corners_filter(self.corners)
        if check_is_None(rect_corners):
            raise ValueError("No rect_corners found")
        # 顺时针排序，左上点在第一个位置
        rect_corners=sort_points_clockwise(rect_corners.squeeze())
        self.rect_corners=rect_corners

    def piece_rotate_vertical(self):
        rotated, corners = rotate_2_vertical(self.rect_corners, self.binary)
        self.binary=rotated
        self.rect_corners=corners

    def get_edges(self):
        edges = cv2.Canny(self.binary, 50, 150)
        self.edges_binary=edges
        edge_points=np.column_stack(np.where(edges == 255))
        # 先x后y
        self.edge_points=edge_points[...,::-1]
        edge_label=get_edge_label(self.binary,self.edge_points,self.rect_corners)
        edge1_binary=cv2.bitwise_and(self.edges_binary,self.edges_binary, mask=(edge_label==1).astype(np.uint8))
        edge2_binary = cv2.bitwise_and(self.edges_binary, self.edges_binary, mask=(edge_label == 2).astype(np.uint8))
        edge3_binary = cv2.bitwise_and(self.edges_binary, self.edges_binary, mask=(edge_label == 3).astype(np.uint8))
        edge4_binary = cv2.bitwise_and(self.edges_binary, self.edges_binary, mask=(edge_label == 4).astype(np.uint8))
        return edge_label
    def show_rect_corners(self):
        return self.show_corners(self.rect_corners)
    def show_all_corners(self):
        return self.show_corners(self.corners)

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



if __name__=="__main__":
    #edge_label()
    pass