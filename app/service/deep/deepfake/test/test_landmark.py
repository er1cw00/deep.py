import os
import cv2
import numpy as np
from rich.progress import track
from deepfake.utils.face import draw_landmarks
from deepfake.facefusion.modules.yoloface import YoloFace
from deepfake.facefusion.modules.face_landmark import FaceLandmark_2dFan
from deepfake.facefusion.modules.retinaface import RetinaFace
from deepfake.facefusion.modules.face_analysis_diy import FaceAnalysisDIY
from deepfake.utils.face import expand_bbox
from deepfake.utils.timer import Timer
from deepfake.utils.video import get_video_writer
from .file import get_test_files


retina_path = '/home/eric/workspace/AI/sd/ComfyUI/facefusion/insightface'
#insightface_path = "/Users/wadahana/workspace/AI/sd/ComfyUI/models/insightface"


yolo_path = '/home/eric/workspace/AI/sd/ComfyUI/models/facefusion/yoloface_8n.onnx'
#yolo_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/yoloface_8n.onnx'

landmark_path = '/home/eric/workspace/AI/sd/ComfyUI/models/facefusion/2dfan4.onnx'
#landmark_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/2dfan4.onnx'

providers=['CUDAExecutionProvider', 'CPUExecutionProvider', 'CoreMLExecutionProvider']

yolo = YoloFace(model_path=yolo_path, providers=providers, threshold=0.5)
retina = RetinaFace(model_path=yolo_path, providers=providers, threshold=0.5)
landmark = FaceLandmark_2dFan(model_path=landmark_path, providers=providers)


photo_list, video_list = get_test_files()

print("Photo directories:")
for path in photo_list:
    input_path = os.path.join(path, 'target.jpg')
    output_path = os.path.join(path, 'output_mask2.png')
    print(path)
    test_image(yolo, xseg0, xseg1, input_path, output_path)

print("\nVideo directories:")
for path in video_list:
    input_path = os.path.join(path, 'target.mp4')
    output_path = os.path.join(path, 'output_mask2.mp4')
    print(path)
    print(input_path)
    print(output_path)
    test_video(yolo, xseg0, xseg1, input_path, output_path)