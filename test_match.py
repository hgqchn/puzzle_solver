import os
import sys
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
from match_utils import *
from tqdm import tqdm
from collections import defaultdict

dir=r'E:\codes\puzzle_solver\results'
match_res_dir=r'E:\codes\puzzle_solver\results\match_results'
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

vex2cave2_count=len(pieces_dict[utils.PieceType.DOUBLE_VEX_DOUBLE_CAVE])

match_results={}
for i1 in range(vex2cave2_count):
    piece1=pieces_dict[utils.PieceType.DOUBLE_VEX_DOUBLE_CAVE][i1]
    match_results[piece1.id]={}
    piece1_results=defaultdict(list)
    for i2 in range(i1+1, vex2cave2_count):
        piece2=pieces_dict[utils.PieceType.DOUBLE_VEX_DOUBLE_CAVE][i2]
        for edge1 in piece1.unmatched_convex_edges:
            for edge2 in piece2.unmatched_concave_edges:
                result=match_two_edges(edge1, edge2)
                if result:
                    p1_to_p2, area_score,dis_score=result
                    print(f"piece {piece1.id} edge {edge1.id} matched with piece {piece2.id} edge {edge2.id} with area_score {area_score:.0f} distance_score {dis_score:.0f}")
                    piece1_results[edge1.id].append({"matched_piece":piece2.id,"matched_edge":edge2.id,"binary":p1_to_p2,"area_score":area_score})
                    plt.imshow(p1_to_p2, cmap='gray')
                    plt.savefig(os.path.join(match_res_dir, f"{piece1.id}_{edge1.id}_{piece2.id}_{edge2.id}_{area_score:0f}.png"))



piece1=pieces_dict[utils.PieceType.DOUBLE_VEX_DOUBLE_CAVE][0]
piece2=pieces_dict[utils.PieceType.DOUBLE_VEX_DOUBLE_CAVE][1]
piece3=pieces_dict[utils.PieceType.DOUBLE_VEX_DOUBLE_CAVE][2]
# fig,axes=plt.subplots(1,3)
# for ax in axes:
#     ax.axis("off")
# axes[0].imshow(piece1.binary, cmap='gray')
# axes[0].set_title(f"piece {piece1.id}")
# axes[1].imshow(piece2.binary, cmap='gray')
# axes[1].set_title(f"piece {piece2.id}")
# axes[2].imshow(piece3.binary, cmap='gray')
# axes[2].set_title(f"piece {piece3.id}")
# fig.savefig(os.path.join(dir, "pieces.png"))
# fig.show()

# p1e3=piece1.edges[3]
# p2e1=piece2.edges[1]
# result=match_two_edges(p1e3, p2e1)
# print(np.count_nonzero(piece1.binary)) #329970
# print(np.count_nonzero(piece2.binary)) #344446
# print(np.count_nonzero(result[0])) #594261
# print(result[1])
# plt.figure()
# plt.imshow(result[0], cmap='gray')
# plt.show()

# e1=piece2.edges[2]
# e2=piece3.edges[4]
# result=match_two_edges(e1, e2)
# print(np.count_nonzero(piece1.binary)) #329970
# print(np.count_nonzero(piece2.binary)) #344446
# print(np.count_nonzero(result[0])) #594261
# print(result[1])
# plt.figure()
# plt.imshow(result[0], cmap='gray')
# plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()