import argparse
import torch
import cv2
import os
import numpy as np
# import torch.backends.cudnn as cudnn

from CustomFunction.plots import *
from CustomFunction.mathforcow import make_center
from CustomFunction.VariableGroup import colors
from yolox.tracker.byte_tracker import BYTETracker
from shapely.geometry import Point, Polygon


def Tracker():

    result_video = list()
    result_map = list()


    tracker = BYTETracker(opt, frame_rate=30)

    out_video = cv2.VideoWriter('/home/ubuntu/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'),30, (1280, 720))
    out_map = cv2.VideoWriter('/home/ubuntu/minimap.mp4', cv2.VideoWriter_fourcc(*'mp4v'),30, (3840, 720))

    # origin_video = cv2.imread('Track_sample/sample_ch1.mp4')

    origin_video = cv2.VideoCapture('/home/ubuntu/Track_sample/sample_ch1.mp4')

    

    frames = np.empty((0,5))

    past = None

    with open('Track_sample/ch1-1_tiny.txt', 'r') as f:
        while True:
            data = f.readline().split(',')
            
            
            if len(data) > 1:
                # t0 = time.time()
                # print(data)
                if past != data[0]:
                    
                    if len(frames) != 0:
                        canvas = cv2.imread('Track_sample/minimap_그림판.jpg')
                    
                        img = origin_video.read()[1]


                        tracked_targets = tracker.update(frames, img.shape)


                        for frame in tracked_targets:

                            xm, ym, xM, yM = frame.tlbr
                            center = make_center([xm, ym, xM, yM])

                            center = [int(i) for i in center]

                            # plot_minimap(center[0], center[1], canvas, colors[frame.track_id % len(colors)])

                            plot_cow(frame, center[0], center[1], frame.track_id, img, colors[frame.track_id % len(colors)])

                        # result_map.append(canvas)
                        # out_map.write(result_map.pop())

                        result_video.append(img)
                        out_video.write(result_video.pop())
                        # out_map.write(canvas)


                        frames = np.empty((0,5))
                        frames = np.append(frames, np.array([[int(i) for i in [float(i) for i in data[2:6]]] + [float(data[6])]]), axis=0)
                        print('start', len(frames), data[0])




                else:
                    frames = np.append(frames, np.array([[int(i) for i in [float(i) for i in data[2:6]]] + [float(data[6])]]), axis=0)
                    print('ok   ', len(frames), data[0])

            else: 
                print("<<<<<<<<<<Finish>>>>>>>>>>")
                break
            past = data[0]
            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        img.release()
        out_video.release()
        out_map.release()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='Track_weigths/yolov7_p5_tiny_ver01.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='/home/ubuntu/yolov7/test_sample_9min.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
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