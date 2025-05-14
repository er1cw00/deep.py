#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import argparse
import imageio
import numpy as np
import onnxruntime
from deepfake.facefusion.utils.affine import warp_face_by_translation

class FaceLandmark_2dFan:
    def __init__(self, model_path, providers):
        self.session  = onnxruntime.InferenceSession(model_path, providers=providers)
        inputs = self.session.get_inputs()
        for input in inputs:
            print(f'input name: {input.name}, shape: {input.shape}')
            if input.name == 'input':
                self.input_name = input.name
                self.input_size = (inputs[0].shape[2], inputs[0].shape[3]) #CHW
        self.affine = False

    def pre_process(self, image, bbox):
        scale = 195 / np.subtract(bbox[2:], bbox[:2]).max()
        translation = (256 - np.add(bbox[2:], bbox[:2]) * scale) * 0.5
        cropped, affine_matrix = warp_face_by_translation(image, translation, scale, (256, 256))
        cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2Lab)
        if np.mean(cropped[:, :, 0]) < 30:
            cropped[:, :, 0] = cv2.createCLAHE(clipLimit = 2).apply(cropped[:, :, 0])
        cropped = cv2.cvtColor(cropped, cv2.COLOR_Lab2RGB)
        cropped = cropped.transpose(2, 0, 1).astype(np.float32) / 255.0
        return cropped, affine_matrix
        

    def post_process(self, face_landmark_68, face_heatmap, affine):
        face_landmark_68 = face_landmark_68[:, :, :2][0] / 64
        face_landmark_68 = face_landmark_68.reshape(1, -1, 2) * 256
        face_landmark_68 = cv2.transform(face_landmark_68, cv2.invertAffineTransform(affine))
        face_landmark_68 = face_landmark_68.reshape(-1, 2)
        face_landmark_68_score = np.amax(face_heatmap, axis = (2, 3))
        face_landmark_68_score = np.mean(face_landmark_68_score)
        return face_landmark_68, face_landmark_68_score
    
    def get(self, image, bbox):
        # Placeholder for face landmark detection logic
        img, affine = self.pre_process(image=image, bbox=bbox)
        face_landmark_68, face_heatmap = self.session.run(None, {self.input_name: [img]})
        return self.post_process(face_landmark_68, face_heatmap, affine)



      
if __name__ == "__main__":
    from .yoloface import YoloFace
    from .occluder import Occluder
    from rich.progress import track
    from deepfake.facefusion.utils.mask import overlay_mask_on_face
    from deepfake.utils.face import draw_landmarks, convert_face_landmark_68_to_5
    from deepfake.utils.timer import Timer
    import imageio
    import cv2
    import time
    import os
    

    os.environ["ORT_LOGGING_LEVEL"] = "VERBOSE"
    
    def test_image(yolo, landmark, input_path, output_path):
        image = cv2.imread(input_path)
        t1 = Timer()
        t2 = Timer()
        
        t1.tic()
        face_list = yolo.get(image=image, order='best-worst')
        t1.toc()
    
        color = (200,10,200)
        for face in face_list:
            
            x1, y1, x2, y2 = map(int, face.bbox)
            
            t2.tic()
            face_landmark_68, face_landmark_68_score = landmark.get(image, face.bbox)
            t2.toc()
            landmark_68_5 = convert_face_landmark_68_to_5(face_landmark_68)
            draw_landmarks(image, landmark_68_5, (200,10,200))
            draw_landmarks(image, face.landmark_5, (200,200,10))
            
            cv2.rectangle(image, (x1,y1), (x2,y2), color, 1)
            cv2.putText(image, f'{face.score:.4f}', (x1,y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(image, f'{face_landmark_68_score:.4f}', (x1,y2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imwrite(output_path, image)
        t1.show('face detect ')
        t2.show('face landmark ')
        
    yolo_path = "/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/yoloface_8n.onnx"
    fan2d_path = "/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/2dfan4.onnx"
    providers=['CoreMLExecutionProvider', 'CPUExecutionProvider', 'CUDAExecutionProvider']
   
    yolo = YoloFace(model_path=yolo_path, providers=providers, threshold=0.5)
    landmark = FaceLandmark_2dFan(model_path=fan2d_path, providers=providers)
#    input_path = "/Users/wadahana/workspace/AI/tbox.ai/data/deep/task/20250405/ec6ee635b4742b08e0fdea6c03769514/source.jpg"
    input_path = "/Users/wadahana/Desktop/sis/faceswap/test/mask/0d4fc3a041d6befd7a2ee218da2d820b/target.jpg"
    output_path = "/Users/wadahana/Desktop/output_face.jpg"   
     
    test_image(yolo=yolo, landmark=landmark, input_path=input_path, output_path=output_path)
  
#                https://x.com/LM62710/status/1920418524171784539
            