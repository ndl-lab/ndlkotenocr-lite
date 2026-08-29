from PIL import Image, ImageDraw
import time
import yaml
import onnxruntime
import numpy as np
from typing import Tuple, List
from onnxruntime_utils import create_inference_session, normalize_device

class RTMDet:
    def __init__(self,
                 model_path: str,
                 class_mapping_path: str,
                 original_size: Tuple[int, int] = (1280,1280),
                 score_threshold: float = 0.1,
                 conf_thresold: float = 0.1,
                 iou_threshold: float = 0.4,
                 device: str = "CPU") -> None:
        self.model_path = model_path
        self.class_mapping_path = class_mapping_path
        self.image_width, self.image_height = original_size
        self.device = normalize_device(device)
        self.score_threshold = score_threshold
        self.conf_thresold = conf_thresold
        self.iou_threshold = iou_threshold
        self.create_session()

    def create_session(self) -> None:
        opt_session = onnxruntime.SessionOptions()
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = create_inference_session(
            self.model_path,
            device=self.device,
            session_options=opt_session,
        )
        self.session = session
        self.model_inputs = self.session.get_inputs()
        self.input_names = [self.model_inputs[i].name for i in range(len(self.model_inputs))]
        self.input_shape = self.model_inputs[0].shape
        self.model_output = self.session.get_outputs()
        self.output_names = [self.model_output[i].name for i in range(len(self.model_output))]
        self.input_height, self.input_width = self.input_shape[2:]

        if self.class_mapping_path is not None:
            with open(self.class_mapping_path, 'r') as file:
                yaml_file = yaml.safe_load(file)
                self.classes = yaml_file['names']
                self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        max_wh=max(img.shape[0],img.shape[1])
        paddedimg=np.zeros((max_wh,max_wh,3)).astype(np.uint8)
        paddedimg[:img.shape[0],:img.shape[1],:]=img.copy()
        pil_image = Image.fromarray(paddedimg)
        self.image_width,self.image_height = pil_image.size
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #resized = cv2.resize(img, (self.input_width, self.input_height))
        pil_resized = pil_image.resize((self.input_width, self.input_height))
        resized=np.array(pil_resized)
        resized=resized[:,:,::-1]
        # Scale input pixel value to 0 to 1
        mean = np.array([103.53,116.28,123.675], dtype=np.float32)
        std = np.array([57.375,57.12,58.395], dtype=np.float32)
        input_image= (resized-mean) / std
        #input_image = resized / 255.0
        input_image = input_image.transpose(2,0,1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor
    
    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y 
    
    def postprocess(self, outputs):
        bboxes,class_ids=outputs
        #print(class_ids)
        class_ids=np.squeeze(class_ids)
        predictions = np.squeeze(bboxes)
        #print(predictions)
        #scores = np.max(predictions[:, 4:], axis=1)
        scores=predictions[:,4]
        #print(scores)
        predictions = predictions[scores > self.conf_thresold, :]
        scores = scores[scores > self.conf_thresold]
        #class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Rescale box
        boxes = predictions[:, :4]
        
        #input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        #boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes/=self.input_width
        #print(boxes)
        boxes *= np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        new_boxes=[]
        for box in boxes:
            delta_h=(box[3]-box[1])*0.02
            new_boxes.append([box[0],box[1]-delta_h,box[2],box[3]+delta_h])

        boxes=np.array(new_boxes)
        boxes = boxes.astype(np.int32)
        #indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.score_threshold, nms_threshold=self.iou_threshold)
        detections = []
        for bbox, score, label in zip(boxes, scores, class_ids):
            detections.append({
                "class_index": 1,
                "confidence": score,
                "box": bbox,
                "class_name": "line_main"
            })
        return detections
    
    def get_label_name(self, class_id: int) -> str:
        return self.classes[class_id]
        
    def detect(self, img: np.ndarray) -> List:
        input_tensor = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return self.postprocess(outputs)
    
    
    def draw_detections(self, npimg: np.ndarray, detections: List):
        pil_image = Image.fromarray(npimg)
        draw = ImageDraw.Draw(pil_image)
        for detection in detections:
            # バウンディングボックスの座標を抽出
            x1, y1, x2, y2 = detection['box']
            class_id = detection['class_index']
            confidence = detection['confidence']
            # クラスIDに対する色を取得
            color = (0, 0, 255)  # RGB形式で青色
            # 画像にバウンディングボックスを描画
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        return pil_image
            
