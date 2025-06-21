import torch
import cv2
import numpy as np
import os

class YOLODetector:
    def __init__(self, weights_path=None, model_type='yolov5'):
        """
        Initialize YOLO detector with pre-trained weights.
        If weights_path is missing, use the standard 'yolov5s' model from Ultralytics.
        model_type: 'yolov5' or 'yolov7' (yolov7 support is placeholder)
        """
        if model_type == 'yolov5':
            if weights_path and os.path.exists(weights_path):
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
            else:
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
        elif model_type == 'yolov7':
            # Placeholder: yolov7 loading would require a different repo or ONNX
            raise NotImplementedError('YOLOv7 loading not implemented in this template.')
        else:
            raise ValueError('Unsupported model type')

    def detect(self, image):
        """
        Run inference on an image (numpy array, BGR).
        Returns: list of bounding boxes [(x1, y1, x2, y2)], confidences, and class names.
        """
        results = self.model(image)
        bboxes = []
        confidences = []
        class_names = []
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = map(int, xyxy)
            bboxes.append((x1, y1, x2, y2))
            confidences.append(float(conf))
            class_names.append(self.model.names[int(cls)])
        return bboxes, confidences, class_names 