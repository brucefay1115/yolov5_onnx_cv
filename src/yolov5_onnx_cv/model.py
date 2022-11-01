import platform
import os
if 'Windows' in platform.system():
    # Improve OpenCV startup speed
    os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'
import cv2
import numpy as np
from yolov5_onnx_cv import datadefine as df


class YOLOv5_ONNX_CV:
    def __init__(self, checkpoint, input_size=(640, 640), conf=0.25, iou=0.45, draw_size=3, is_cuda=False):
        assert self.class_colors or self.class_names, 'You must inherit YOLOv5_ONNX_CV and define class_colors and class_names'
        self.input_size = input_size
        self.conf = conf
        self.iou = iou
        self.draw_size = draw_size
        self.model = self._build_model(checkpoint, is_cuda)

    def _build_model(self, checkpoint, is_cuda=False):
        net = cv2.dnn.readNetFromONNX(checkpoint)
        if is_cuda:
            print('Attempty to use CUDA')
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            print('Running on CPU')
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def _detect(self, input, input_size, model):
        blob = cv2.dnn.blobFromImage(input, 1/255.0, input_size, swapRB=True, crop=False)
        model.setInput(blob)
        preds = model.forward()
        return preds

    def _letterbox(self, img, new_shape=(640, 640)):
        '''
            Resize image to new_sahpe size and move image to center location.
            For example your input image size as 1920x1080
            The image will resize to 640x640 and padding other space.
            ex:                             
                                            (xxxx means padding area)
                1920x1080           to      640x640
                oooooooooooooo              xxxxxxxxxxxxxx
                oooooooooooooo              oooooooooooooo
                oooooooooooooo              oooooooooooooo
                oooooooooooooo              oooooooooooooo
                oooooooooooooo              oooooooooooooo
                oooooooooooooo              xxxxxxxxxxxxxx
        '''
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return img, ratio, (dw, dh)

    def _xywh_to_box(self, x, y, w, h, ratio, dwh):
        left = int(((x - 0.5 * w) - dwh[0]) / ratio[0])
        top = int(((y - 0.5 * h) - dwh[1]) / ratio[1])
        width = int(w / ratio[0])
        height = int(h / ratio[1])
        box = np.array([left, top, width, height])
        return box

    def _nms_boxes(self, ids, confs, boxes):
        result = []
        indexes = cv2.dnn.NMSBoxes(boxes, confs, self.conf, self.iou) 
        for i in indexes:
            i = int(i)
            id = ids[i]
            result.append(df.ClassInfo(id, self.class_names[id], confs[i], boxes[i]))
        return result

    def _wrap_detection(self, output, ratio, dwh):
        ids, confs, boxes = [], [], []
        rows = output.shape[0]
        for r in range(rows):
            row = output[r]
            conf = row[4]
            if conf >= self.conf:
                scores = row[5:]
                id = scores.argmax()
                if (scores[id] >= self.conf):
                    confs.append(float(conf))
                    ids.append(int(id))
                    box = self._xywh_to_box(
                        int(row[0].item()), 
                        int(row[1].item()), 
                        int(row[2].item()), 
                        int(row[3].item()), 
                        ratio, dwh)
                    boxes.append(box)
        return self._nms_boxes(ids, confs, boxes)

    def _draw_box(self, img, label, box, color, line_width):
        lw = line_width or max(round(sum(img.shape) / 2 * 0.003), 2)
        p1 = (int(box[0]), int(box[1]))
        p2 = (p1[0] + int(box[2]), p1[1] + int(box[3]))
        cv2.rectangle(img, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        tf = max(lw - 1, 1)
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(img,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    (255, 255, 255),
                    thickness=tf,
                    lineType=cv2.LINE_AA)

    def _label_boxes(self, hide_conf):
        img = self.img
        for c_info in self.class_infos:
            color = self.class_colors[int(c_info.id) % len(self.class_colors)]
            label = self.class_names[c_info.id]
            if not hide_conf:
                label += f':{c_info.conf:.2f}'
            self._draw_box(img, label, c_info.box, color, self.draw_size)
        return img

    def show_label_boxes(self, hide_conf=True):
        img = self._label_boxes(hide_conf)
        cv2.namedWindow('show_boxes', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('show_boxes', img)

    def get_label_boxes_image(self, hide_conf=True):
        return self._label_boxes(hide_conf)

    def __call__(self, input):
        self.img = input.copy()
        input, ratio, dwh = self._letterbox(input, self.input_size)
        outputs = self._detect(input, self.input_size, self.model)
        self.class_infos = self._wrap_detection(outputs[0], ratio, dwh)
        return self.class_infos
