from PIL import Image
import time
import yaml
import onnxruntime
import numpy as np
from typing import Tuple, List

class PARSEQ:
    def __init__(self,
                 model_path: str,
                 charlist: [str],
                 original_size: Tuple[int, int] = (384, 32),
                 device: str = "CPU") -> None:
        self.model_path = model_path
        self.charlist = charlist

        self.device = device
        self.image_width, self.image_height = original_size
        self.create_session()

    def create_session(self) -> None:
        opt_session = onnxruntime.SessionOptions()
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers = ['CPUExecutionProvider']
        if self.device.casefold() != "cpu":
            providers.append("CUDAExecutionProvider")
        session = onnxruntime.InferenceSession(self.model_path, providers=providers)
        self.session = session
        self.model_inputs = self.session.get_inputs()
        self.input_names = [self.model_inputs[i].name for i in range(len(self.model_inputs))]
        self.input_shape = self.model_inputs[0].shape
        self.model_output = self.session.get_outputs()
        self.output_names = [self.model_output[i].name for i in range(len(self.model_output))]
        self.input_height, self.input_width = self.input_shape[2:]

    def postprocess(self, outputs):
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_thresold, :]
        scores = scores[scores > self.conf_thresold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(img)
        if pil_image.height>pil_image.width:
            pil_image =pil_image.transpose(Image.ROTATE_90)
        #image_rgb=img
        #image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #resized = cv2.resize(image_rgb, (self.input_width, self.input_height),interpolation=cv2.INTER_CUBIC)
        pil_resized = pil_image.resize((self.input_width, self.input_height))
        resized = np.array(pil_resized)
        resized = resized[:,:,::-1]
        input_image = resized / 255.0
        input_image = 2.0*(input_image-0.5)
        input_image = input_image.transpose(2,0,1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor
    
    def read(self, img: np.ndarray) -> List:
        if img is None:
            return None
        input_tensor = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        resstr=""
        resval=[]
        befw=""
        #print(outputs[0].shape)
        for idx in np.argmax(outputs,axis=2)[0]:
            if idx==0:
                break
            resstr+=self.charlist[idx-1]
            resval.append(idx)
        #print(resstr,resval)
        return resstr
    
