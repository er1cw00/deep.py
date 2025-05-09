import os
import cv2
import numpy as np
from rich.progress import track
from deepfake.utils.face import draw_landmarks
from deepfake.facefusion.modules.yoloface import YoloFace
from deepfake.facefusion.modules.face_analysis_diy import FaceAnalysisDIY
from deepfake.utils.face import expand_bbox
from deepfake.utils.timer import Timer
from deepfake.utils.video import get_video_writer
from .file import get_test_files
    
def get_one_face(det, image):
    face_list = det.get(image=image, conf=0.5, order='best-worst')
    if face_list != None and len(face_list) > 0:
        return face_list[0]
    return None

# def draw_bbox(image, face):
#     x1, y1, x2, y2 = map(int, face.bbox)
#     cv2.rectangle(image, (x1,y1), (x2,y2), (255, 0, 0), 1)
#     (new_x1, new_y1, new_x2, new_y2) = expand_bbox(face.bbox, 512)
#     face_cropped = image[new_y1:new_y2, new_x1:new_x2]
#     resized_face = cv2.resize(face_cropped, (512, 512))
#     return resized_face
    
def test_image(det1, det2, input_path, output_path):
    t1 = Timer()
    t2 = Timer()

    image = cv2.imread(input_path)
    t1.tic()
    face1 = get_one_face(det1, image)
    t1.toc()
    
    t2.tic()
    face2 = get_one_face(det2, image)
    t2.toc()
    
    color = (0, 0, 200)
    
    if face1 != None:
        output1 = draw_landmarks(image.copy(), face1.landmark_5)
        x1, y1, x2, y2 = map(int, face1.bbox)
        cv2.rectangle(output1, (x1,y1), (x2,y2), color, 2)
        cv2.putText(output1, f'{face2.score:.4f}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        output1 = np.zeros_like(image)
        
    if face2 != None:
        output2 = draw_landmarks(image.copy(), face2.landmark_5)  
        x1, y1, x2, y2 = map(int, face2.bbox)
        cv2.rectangle(output2, (x1,y1), (x2,y2), color, 2)
        cv2.putText(output2, f'{face2.score:.4f}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        output2 = np.zeros_like(image)
        
    combined = cv2.hconcat([image, output1, output2])
    cv2.imwrite(output_path, combined)
    t1.show('yoloface')
    t2.show('insightface')
        
def test_video(det1, det2, input_path, output_path):
    t1 = Timer()
    t2 = Timer()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
        
    print(f'video [{width}x{height}@{fps}] {total} frames!')
    
    writer = get_video_writer(output_path, fps)
    for i in track(range(total), description='Detecting....', transient=True):
        ret, frame = cap.read()
        if not ret:
            break
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        t1.tic()
        face1 = get_one_face(det1, frame)
        t1.toc()
        
        t2.tic()
        face2 = get_one_face(det2, frame)
        t2.toc()
        
        color = (10,200,10)
        if face1 == None:
            output1 = np.zeros_like(frame)
        else:
            output1 = draw_landmarks(frame.copy(), face1.landmark_5)
            x1, y1, x2, y2 = map(int, face1.bbox)
            cv2.rectangle(output1, (x1,y1), (x2,y2), color, 2)
            cv2.putText(output1, f'{face1.score:.4f}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
         
        if face2 == None:
            output2 = np.zeros_like(frame)
        else:
            output2 = draw_landmarks(frame.copy(), face2.landmark_5)
            x1, y1, x2, y2 = map(int, face2.bbox)
            cv2.rectangle(output2, (x1,y1), (x2,y2), color, 2)
            cv2.putText(output2, f'{face2.score:.4f}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                       
            combined = cv2.hconcat([frame, output1, output2])
            writer.append_data(combined[..., ::-1])

    cap.release()
    writer.close()
    
    t1.show('yoloface')
    t2.show('insightface')
    
insightface_path = '../../../models/insightface'
#insightface_path = "/Users/wadahana/workspace/AI/sd/ComfyUI/models/insightface"
providers=['CPUExecutionProvider', 'CoreMLExecutionProvider', 'CUDAExecutionProvider']

yolo_path = '../../../models/facefusion/yoloface_8n.onnx'
#yolo_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/yoloface_8n.onnx'
providers=['CPUExecutionProvider', 'CoreMLExecutionProvider', 'CUDAExecutionProvider']

yolo = YoloFace(model_path=yolo_path, providers=providers)

insight = FaceAnalysisDIY(name="buffalo_l", root=insightface_path, providers=providers)
insight.prepare(ctx_id=0, det_size=(512, 512), det_thresh=0.5)
insight.warmup()

# input_path = "/Users/wadahana/workspace/AI/tbox.ai/data/deep/task/20250405/ec6ee635b4742b08e0fdea6c03769514/source.jpg"
# output_path = "/Users/wadahana/Desktop/output_face.jpg"    
# test_image(yolo, insight, input_path=input_path, output_path=output_path)

# input_path = "/Users/wadahana/Desktop/sis/faceswap/test/mask/fbb6081fa3544ba51e4058b71660cfe3/target.mp4"
# output_path = "/Users/wadahana/Desktop/output_face.mp4"    
# test_video(yolo, insight, input_path=input_path, output_path=output_path)

photo_list, video_list = get_test_files()

print("Photo directories:")
for path in photo_list:
    input_path = os.path.join(path, 'target.jpg')
    output_path = os.path.join(path, 'output_face.png')
    print(path)
    test_image(yolo, yolo, insight, input_path, output_path)

print("\nVideo directories:")
for path in video_list:
    input_path = os.path.join(path, 'target.mp4')
    output_path = os.path.join(path, 'output_face.mp4')
    print(path)
    test_video(yolo, yolo, insight, input_path, output_path)
    
print('test face detect finished! ')
