import argparse
import torch
import cv2
import os
import numpy as np


from CustomFunction.plots import *
from CustomFunction.VariableGroup import colors
from CustomFunction.PixelMapper import pm_1, pm_2, pm_3
from CustomFunction.mathforcow import *


from yolox.tracker.byte_tracker import BYTETracker
from shapely.geometry import Point, Polygon


def Tracker():

    cap = cv2.VideoCapture('/home/ubuntu/Track_sample/videos/1/realmonitor_20221118_155620.mp4')

    result_video = list()

    tracker = BYTETracker(opt, frame_rate=30)

    fcc = cv2.VideoWriter_fourcc(*"mp4v")

    width = int(1280) # 가로 길이 가져오기 
    height = int(720) # 세로 길이 가져오기

    out_size = (width , height)

    fps = 30

    out_video = cv2.VideoWriter('/home/ubuntu/video2.mp4', fcc,fps, out_size)
    # out_map = cv2.VideoWriter('/home/ubuntu/minimap1.mp4', fcc,fps, out_size)

    coordinate = np.empty((0,5))

    past_frame_num = None

    # ch2 = np.array([1280, 0, 1280, 0, 0])
    # ch3 = np.array([2560, 0, 2560, 0, 0])

    

    with open('/home/ubuntu/Track_sample/ch1-1_tiny.txt', 'r') as f:
        while True:
            data = f.readline().split(',')
            
            frame_num = data[0]
            # channel = data[1]

            if len(data) > 1:
                if past_frame_num != frame_num:
                    # cams = [capture.retrieve()[1] for capture in cap if capture.grab()]
                    if len(coordinate) != 0:
                        # img = origin_video.read()[1]

                        # images
                        canvas = cv2.imread('Track_sample/minimap_그림판.jpg')
                        # cam1, cam2, cam3 = cams
                        # img = np.hstack((cam1, cam2, cam3))

                        img = cap.read()[1]

                        # Tracker
                        tracked_targets = tracker.update(coordinate, img.shape)

                        # variable
                        dict_img = dict()

                        dict_minimap = dict()

                        remove_set = set()

                        # 각 객체
                        for obj in tracked_targets:

                            # Bbox center
                            xm, ym, xM, yM = obj.tlbr 

                            center = make_center([xm, ym, xM, yM])

                            center = [int(i) for i in center]

                            # Bbox Overlay
                            plot_cow(obj, center[0], center[1], obj.track_id, img, colors[obj.track_id % len(colors)])

                            # distance 사용하여 중복 Track 객체 삭제 알고리즘 사용하기 위한 변수 value 저장
                            dict_minimap[obj.track_id] = pm_1.pixel_to_lonlat([center[0], center[1]])
                            dict_img[obj.track_id] = center

                        ### distance를 이용하여 중복 Track된 객체를 삭제하는 코드
                        # dict_minimap안에 값들을 2중 for문을 활용하여 각각의 track_id에 대응되지 않는 값의 distance를 비교하여
                        # 해당 distance가 100 미만이라면 동일 객체라 선정
                        for k1, v1 in dict_minimap.items():
                            for k2, v2 in dict_minimap.items():
                                if k1 == k2:
                                    continue
                                else:
                                    # distance가 100 미만 일 경우 == 동일 객체
                                    if distance(v1, v2) < 100:
                                        remove_set.add(k1)

                        # 선정된 track_id를 사용하여 dict_minimap안의 대응 원소를 삭제
                        for remove_id in remove_set:
                            dict_minimap.pop(remove_id)

                        # 각 프레임 마다 Minimap Dot Overlay 생성
                        for key, value in dict_minimap.items():

                            plot_minimap(value[0][0], value[0][1], canvas, colors[key % len(colors)])
                    
                                
                        # # video write
                        # result_map.append(canvas)
                        # out_map.write(result_map.pop())

                        # result_video.append(img)
                        # out_video.write(result_video.pop())


                        # out_map.write(canvas)
                        out_video.write(img)

                        # 새로운 좌표 저장
                        coordinate = np.empty((0,5))
                        coordinate = np.append(coordinate, np.array([[int(i) for i in [float(i) for i in data[2:6]]] + [float(data[6])]]), axis=0)
                        print(f'start   obj : 0{len(coordinate)} frame : {frame_num}')
                else:
                    # 좌표 저장
                
                    coordinate = np.append(coordinate, np.array([[int(i) for i in [float(i) for i in data[2:6]]] + [float(data[6])]]), axis=0)

                    if len(coordinate) < 10:
                        coor = '0' + str(len(coordinate))
                    else:
                        coor = len(coordinate)
                    print(f'ok      obj : {coor} frame : {frame_num}')

                    # if int(frame_num) > 200:
                    #     print(f'frame_num : {frame_num}')
                    #     break
            else: 
                print("<<<<<<<<<<Finish>>>>>>>>>>")
                break
            past_frame_num = frame_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='Track_weigths/yolov7_p5_tiny_ver01.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='/home/ubuntu/yolov7/test_sample_9min.mp4', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    # parser.add_argument('--save-txt', action='store_true', default=True, help='save results to *.txt')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        # if opt.update:  # update all models (to fix SourceChangeWarning)
        #     for opt.weights in ['yolov7.pt']:
        #         Tracker()
        #         strip_optimizer(opt.weights)
        # else:
        #     Tracker()
        Tracker()