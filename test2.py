import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import pickle
import time
from tqdm import tqdm

from utils import cal_edge_type

with open("piece1.pkl", "rb") as f:
    piece1 = pickle.load(f)



#cv2.imshow("rect_corners_after", piece1.show_rect_corners())
#mask,edges_corners= utils.get_edge_label(piece1.binary, piece1.edge_points, piece1.rect_corners)
mask=piece1.get_edges()
#cv2.imshow("edge",piece1.edges_binary)
#cv2.imshow("rect_corners", piece1.show_rect_corners())
#cv2.imshow("mask", utils.show_edge_label(mask))
cv2.waitKey(0)
cv2.destroyAllWindows()
