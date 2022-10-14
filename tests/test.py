import os; import sys; abspath = os.path.dirname(os.path.abspath(__file__)); sys.path.append(f'{abspath}\\..\\src')
import numpy as np
from yolov5_onnx_cv import YOLOv5_ONNX_CV
import cv2


class YOLOv5(YOLOv5_ONNX_CV):
    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
        'bus', 'train', 'truck', 'boat', 'traffic light', 
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 
        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 
        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
        'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 
        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    class_colors = [(int(i[0]), int(i[1]), int(i[2])) for i in np.random.randint(256, size=(len(class_names), 3))]


def load_capture():
    capture = cv2.VideoCapture(1)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return capture


if __name__ == '__main__':
    capture = load_capture()
    model = YOLOv5('tests/yolov5s.onnx', (640, 640), 0.5)

    while True:
        success, frame = capture.read()
        if not success:
            print('Open camera fail.')
            break

        class_ids, confs, boxes = model(frame)
        model.show_label_boxes() 
        
        # img = model.get_label_boxes_image()

        # Any key to quit.
        if cv2.waitKey(1) > -1:
            print('finished by user')
            break

    capture.release()
