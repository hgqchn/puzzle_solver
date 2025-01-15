import os
import sys
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
from collections import defaultdict

dir=r'E:\codes\puzzle_solver\results'

files = os.listdir(dir)
# 过滤出指定后缀的文件
pkl_files = [f for f in files if f.endswith(".pkl")]

pieces=[]
pieces_dict=defaultdict(list)
for file in pkl_files:
    piece = utils.deserialize(os.path.join(dir, file))
    pieces.append(piece)
    if piece.piece_type not in pieces_dict.keys():
        pieces_dict[piece.piece_type].append(piece)
    else:
        pieces_dict[piece.piece_type].append(piece)

