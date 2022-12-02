import cv2
from CustomFunction.PixelMapper import *
from CustomFunction.mathforcow import find_distance
import math


def plot_minimap(xc, yc, minimap, color):
    ### xc, yc 를 미니맵 기준의 좌표로 변환해주는 코드
    # corr1, corr2 = pm_1.pixel_to_lonlat((xc, yc))[0]
    corr1 = int(xc)
    corr2 = int(yc)

    ### 변환된 좌표로 미니맵에 Dot 찍는 코드
    cv2.line(minimap, (corr1, corr2), (corr1, corr2), color, 50)


def plot_cow(x, xc, yc, track_id, img, color):
    c1, c2 = (int(x.tlbr[0]), int(x.tlbr[1])), (int(x.tlbr[2]), int(x.tlbr[3]))

    ### 소 Center에 Dot 생성 코드
    cv2.line(img, (xc, yc), (xc, yc), color, 10)
    ### Bbox 생성 코드
    cv2.rectangle(img, c1, c2, color, thickness=5, lineType=cv2.LINE_AA)
    ### Cow 번호 생성 코드
    cv2.putText(img, str(track_id) , (c1[0],c1[1]), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 4)


