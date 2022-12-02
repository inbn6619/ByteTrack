import argparse
import torch
import cv2
import os
import numpy as np
import pandas as pd


from CustomFunction.plots import *
from CustomFunction.VariableGroup import colors
from CustomFunction.PixelMapper import pm_1, pm_2, pm_3
from CustomFunction.mathforcow import *


from yolox.tracker.byte_tracker import BYTETracker
from shapely.geometry import Point, Polygon



def Tracker():

    vid_path = [
        '/home/ubuntu/Track_sample/videos/1/realmonitor_20221118_155620.mp4',
        '/home/ubuntu/Track_sample/videos/1/realmonitor_20221118_155621.mp4',
        '/home/ubuntu/Track_sample/videos/1/realmonitor_20221118_155622.mp4'
    ]


    cap = list()
    for path in vid_path:
        cap.append(cv2.VideoCapture(path))

    # result_video = list()
    # result_map = list()
    # origin_video = cv2.VideoCapture('/home/ubuntu/Track_sample/videos/1/realmonitor_20221118_155620.mp4')


    tracker = BYTETracker(opt, frame_rate=30)

    fcc = cv2.VideoWriter_fourcc(*"mp4v")

    width = int(1280) # 가로 길이 가져오기 
    height = int(720) # 세로 길이 가져오기

    out_size = (width * 3, height)

    fps = 30

    out_video = cv2.VideoWriter('/home/ubuntu/video.mp4', fcc,fps, out_size)
    out_video_and_mini = cv2.VideoWriter('/home/ubuntu/video_and_mini.mp4', fcc,fps, (3840, 1440))
    out_map = cv2.VideoWriter('/home/ubuntu/minimap.mp4', fcc,fps, out_size)

    past_frame_num = None

    ch2 = np.array([1280, 0, 1280, 0, 0])
    ch3 = np.array([2560, 0, 2560, 0, 0])

    ch1_coor = np.empty((0,5))
    ch2_coor = np.empty((0,5))
    ch3_coor = np.empty((0,5))

                            # image = np.vstack((img, canvas))

    with open('/home/ubuntu/Track_sample/datas/1/dataframe1.csv', 'r') as f:
        while True:
            data = f.readline().split(',')[1:]
            frame_num = data[0]
            channel_name = data[1]

            

            # print(data)

            if len(data) > 1:
                if past_frame_num != frame_num:

                    try:
                        if type(int(frame_num)) == type(int()):
                            cams = [capture.retrieve()[1] for capture in cap if capture.grab()]
                            if len(cams) == 3:
                                # images
                                canvas = cv2.imread('Track_sample/minimap_그림판.jpg')
                                cam1, cam2, cam3 = cams
                                img = np.hstack((cam1, cam2, cam3))

                                # Tracker
                                # tracked_targets = tracker.update(coordinate, img.shape)

                                # variable
                                dict_img = dict()

                                dict_minimap = dict()

                                remove_set = set()
                                if max(len(ch1_coor), len(ch2_coor), len(ch3_coor)) != 0:
                                    ch1_target = tracker.update(ch1_coor, img.shape)

                                    for target1 in ch1_target:
                                        xm, ym, xM, yM = target1.tlbr 

                                        center = make_center([xm, ym, xM, yM])

                                        center = [int(i) for i in center]

                                        # Bbox Overlay
                                        plot_cow(target1, center[0], center[1], target1.track_id, img, colors[target1.track_id % len(colors)])

                                        # distance 사용하여 중복 Track 객체 삭제 알고리즘 사용하기 위한 변수 value 저장
                                        dict_minimap[target1.track_id] = pm_1.pixel_to_lonlat([center[0], center[1]])
                                        dict_img[target1.track_id] = center

                                    ch2_target = tracker.update(ch2_coor, img.shape)

                                    for target2 in ch2_target:
                                        xm, ym, xM, yM = target2.tlbr 

                                        center = make_center([xm, ym, xM, yM])

                                        center = [int(i) for i in center]

                                        # Bbox Overlay
                                        plot_cow(target2, center[0], center[1], target2.track_id, img, colors[target2.track_id % len(colors)])

                                        # distance 사용하여 중복 Track 객체 삭제 알고리즘 사용하기 위한 변수 value 저장
                                        dict_minimap[target2.track_id] = pm_2.pixel_to_lonlat([center[0], center[1]])
                                        dict_img[target2.track_id] = center

                                    ch3_target = tracker.update(ch3_coor, img.shape)

                                    for target3 in ch3_target:
                                        xm, ym, xM, yM = target3.tlbr 

                                        center = make_center([xm, ym, xM, yM])

                                        center = [int(i) for i in center]

                                        # Bbox Overlay
                                        plot_cow(target3, center[0], center[1], target3.track_id, img, colors[target3.track_id % len(colors)])

                                        # distance 사용하여 중복 Track 객체 삭제 알고리즘 사용하기 위한 변수 value 저장
                                        dict_minimap[target3.track_id] = pm_3.pixel_to_lonlat([center[0], center[1]])
                                        dict_img[target3.track_id] = center



                                    # ### distance를 이용하여 중복 Track된 객체를 삭제하는 코드
                                    # # dict_minimap안에 값들을 2중 for문을 활용하여 각각의 track_id에 대응되지 않는 값의 distance를 비교하여
                                    # # 해당 distance가 100 미만이라면 동일 객체라 선정
                                    # for k1, v1 in dict_minimap.items():
                                    #     for k2, v2 in dict_minimap.items():
                                    #         if k1 == k2:
                                    #             continue
                                    #         else:
                                    #             # distance가 100 미만 일 경우 == 동일 객체
                                    #             if distance(v1, v2) < 100:
                                    #                 remove_set.add(k1)

                                    # # 선정된 track_id를 사용하여 dict_minimap안의 대응 원소를 삭제
                                    # for remove_id in remove_set:
                                    #     dict_minimap.pop(remove_id)

                                    # 각 프레임 마다 Minimap Dot Overlay 생성


                                    for key, value in dict_minimap.items():

                                        plot_minimap(value[0][0], value[0][1], canvas, colors[key % len(colors)])




                                    out_map.write(canvas)
                                    out_video.write(img)

                                    image = np.vstack((img, canvas))

                                    out_video_and_mini.write(image)


                                ch1_coor = np.empty((0,5))
                                ch2_coor = np.empty((0,5))
                                ch3_coor = np.empty((0,5))

                                # 새로운 좌표 저장
                                if channel_name == 'ch1':
                                    ch1_coor = np.append(ch1_coor, np.array([[int(i) for i in [float(i) for i in data[2:6]]] + [float(data[6])]]), axis=0)
                                elif channel_name == 'ch2':
                                    ch2_coor = np.append(ch2_coor, np.array([[int(i) for i in [float(i) for i in data[2:6]]] + [float(data[6])]]) + ch2, axis=0)
                                else:
                                    ch3_coor = np.append(ch3_coor, np.array([[int(i) for i in [float(i) for i in data[2:6]]] + [float(data[6])]]) + ch3, axis=0)
                                objtrack = len(ch1_coor) + len(ch2_coor) + len(ch3_coor)
                                print(f'start   obj : 0{objtrack} frame : {frame_num}')
                            else:
                                print('can not capture video')
                                break
                    except ValueError:
                        pass
                    
                else:
                    if channel_name == 'ch1':
                        ch1_coor = np.append(ch1_coor, np.array([[int(i) for i in [float(i) for i in data[2:6]]] + [float(data[6])]]), axis=0)
                    elif channel_name == 'ch2':
                        ch2_coor = np.append(ch2_coor, np.array([[int(i) for i in [float(i) for i in data[2:6]]] + [float(data[6])]]) + ch2, axis=0)
                    else:
                        ch3_coor = np.append(ch3_coor, np.array([[int(i) for i in [float(i) for i in data[2:6]]] + [float(data[6])]]) + ch3, axis=0)
                    objtrack = len(ch1_coor) + len(ch2_coor) + len(ch3_coor)

                    if objtrack < 10:
                        print(f'ok      obj : {str(0) + str(objtrack)} frame : {frame_num}')
                    else:
                        print(f'ok      obj : {objtrack} frame : {frame_num}')


                    if int(frame_num) > 200:
                        print(f'frame_num : {frame_num}')
                        break


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